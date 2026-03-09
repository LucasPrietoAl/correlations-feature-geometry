from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from datasets import DatasetDict as HFDatasetDict
from datasets import Dataset as HFDataset

from text_bows.constants import EXTRA_TOKENS
from text_bows.utils_io import iter_windows_npz


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


DEFAULT_CACHE_DIR = os.getenv("HF_HOME", None) or None
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_NLTK_DIR = "nltk_data"
NUMBERS_TO_KEEP: Set[str] = {str(i) for i in range(101)}

SUPPORTED_FORMATS = {"npz", "jsonl", "pt"}



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        logging.info("CUDA available. Seeding CUDA RNGs.")
        torch.cuda.manual_seed_all(seed)


def setup_nltk(download_dir: str) -> Set[str]:
    os.makedirs(download_dir, exist_ok=True)
    if download_dir not in nltk.data.path:
        nltk.data.path.append(download_dir)

    resources = {
        "stopwords": os.path.join("corpora", "stopwords", "english"),
        "punkt": os.path.join("tokenizers", "punkt", "english.pickle"),
    }

    for res_id, relative_path in resources.items():
        expected = os.path.join(download_dir, relative_path)
        if not os.path.exists(expected):
            logging.info("NLTK resource '%s' missing → downloading…", res_id)
            try:
                nltk.download(res_id, download_dir=download_dir, quiet=True)
            except Exception as exc:
                logging.error("Failed to download '%s': %s", res_id, exc)
                sys.exit(1)

    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except (LookupError, OSError):
        try:
            nltk.download("punkt_tab", download_dir=download_dir, quiet=True)
        except Exception:
            pass

    try:
        sw = set(stopwords.words("english"))
        logging.info("NLTK ready (punkt, stopwords).")
        return sw
    except Exception as exc:
        logging.error("NLTK load failed after download: %s", exc)
        sys.exit(1)


def load_wikitext(cache_dir: Optional[str]) -> HFDatasetDict:
    return load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)


def _split_from_single(ds: HFDataset, val: int, test: int, seed: int = 42) -> HFDatasetDict:
    ds_shuf = ds.shuffle(seed=seed)
    validation = ds_shuf.select(range(val))
    test_set = ds_shuf.select(range(val, val + test))
    train = ds_shuf.select(range(val + test, len(ds_shuf)))
    return HFDatasetDict({"train": train, "validation": validation, "test": test_set})


def load_openwebtext(cache_dir: Optional[str], val: int = 10_000, test: int = 10_000) -> HFDatasetDict:
    owt_train: HFDataset = load_dataset("Skylion007/openwebtext", cache_dir=cache_dir, split="train", revision="refs/convert/parquet",)
    return _split_from_single(owt_train, val, test)


def load_dataset_by_name(name: str, cache_dir: Optional[str]) -> Tuple[str, HFDatasetDict]:
    if name == "wikitext":
        return "wikitext", load_wikitext(cache_dir)
    if name == "openwebtext":
        return "openwebtext", load_openwebtext(cache_dir)
    raise ValueError(f"Unknown dataset '{name}'.")


def tokenize_text(
    text: str,
    stop_words: Set[str],
    numbers: Set[str],
    lowercase: bool = True,
) -> List[str]:
    if not isinstance(text, str):
        return []
    toks = word_tokenize(text)
    kept: List[str] = []
    for t in toks:
        if lowercase:
            tl = t.lower()
        else:
            tl = t
        if (tl.isalpha() and tl not in stop_words) or tl in numbers:
            kept.append(tl)
    return kept


def collect_tokens(
    ds_split: HFDataset,
    stop_words: Set[str],
    numbers: Set[str],
    min_len: int = 50,
    lowercase: bool = True,
) -> List[str]:
    tokens: List[str] = []
    from tqdm import tqdm

    for rec in tqdm(ds_split, desc="Collecting tokens"):
        txt = rec.get("text", "")
        if isinstance(txt, str) and len(txt) >= min_len:
            tokens.extend(tokenize_text(txt, stop_words, numbers, lowercase))
    return tokens


def build_vocab(tokens: Sequence[str], max_size: int, extra_tokens: Set[str]) -> Tuple[List[str], Dict[str, int]]:
    base_cnt = Counter(tokens)

    for tok in extra_tokens:
        base_cnt.pop(tok, None)

    vocab_list: List[str] = sorted(extra_tokens)

    for term, _ in base_cnt.most_common(max(0, max_size - len(vocab_list))):
        vocab_list.append(term)
        if len(vocab_list) >= max_size:
            break

    if len(vocab_list) > max_size:
        logging.warning(
            "Vocabulary exceeded requested size due to mandatory tokens (%d > %d).",
            len(vocab_list),
            max_size,
        )

    word2idx = {w: i for i, w in enumerate(vocab_list)}
    logging.info(
        "Vocabulary built: %d unique tokens (requested %d; mandatory %d).",
        len(vocab_list),
        max_size,
        len(extra_tokens),
    )
    return vocab_list, word2idx

def get_save_path(
    save_dir: str,
    dataset_name: str,
    split_name: str,
    vocab_size: int,
    group_size: int,
    stride: int,
    fmt: str,
) -> str:
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from {sorted(SUPPORTED_FORMATS)}")
    fname = f"{dataset_name}_{split_name}_v{vocab_size}_w{group_size}_s{stride}.{fmt}"
    return os.path.join(save_dir, fname)


def parse_path_details(filename: str) -> Optional[Tuple[int, int, int]]:
    """Extract (vocab_size, group_size, stride) from a filename."""
    m = re.search(r"_v(\d+)_w(\d+)_s(\d+)\.(npz|jsonl|pt)$", filename)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


@dataclass
class Meta:
    dataset: str
    split: str
    vocab_size: int
    group_size: int
    stride: int
    vocab_path: str
    extra_tokens: List[str]
    min_len: int
    lowercase: bool
    numbers_min: int
    numbers_max: int
    seed: int
    version: str = "0.1.0"
    storage_format: str = "npz"
    library_versions: Dict[str, str] = None


def write_metadata(path: str, meta: Meta) -> None:
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)

def canonical_dataset_name(dataset_name: str) -> str:
    return f"{dataset_name}-103-raw-v1" if dataset_name == "wikitext" else dataset_name

def _file_is_valid(path: str) -> bool:
    """Lightweight integrity check for cached dataset files."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".npz":
            if not zipfile.is_zipfile(path):
                return False
            with np.load(path, mmap_mode="r") as z:
                return "offsets" in z and "values" in z
        if ext in {".jsonl", ".pt"}:
            return os.path.getsize(path) > 0
    except Exception as exc:
        logging.warning("Validation failed for %s: %s", os.path.basename(path), exc)
        return False
    return True

def _target_matches(path: str, dataset_name: str, split: str, args: argparse.Namespace) -> bool:
    """Check if an existing target file matches requested parameters."""
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    expected = {
        "dataset": canonical_dataset_name(dataset_name),
        "split": split,
        "vocab_size": args.vocab_size,
        "group_size": args.group_size,
        "stride": args.stride,
        "storage_format": args.storage_format,
    }
    for k, v in expected.items():
        if meta.get(k) != v:
            return False
    if not _file_is_valid(path):
        logging.warning("Existing target %s looks corrupted; regenerating.", os.path.basename(path))
        return False
    return True

def _to_offsets_and_values(samples: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    lengths = [len(s) for s in samples]
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    if lengths:
        offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    values = np.empty(offsets[-1], dtype=np.int64)
    pos = 0
    for s in samples:
        if len(s):
            values[pos : pos + len(s)] = s
        pos += len(s)
    return offsets, values


def save_sparse(samples: Sequence[Sequence[int]], path: str, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == "npz":
        offsets, values = _to_offsets_and_values(samples)
        np.savez_compressed(path, offsets=offsets, values=values)
    elif fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    elif fmt == "pt":
        torch.save([list(map(int, s)) for s in samples], path)
    else:
        raise ValueError(f"Unsupported format '{fmt}'.")


def iter_sparse(path: str) -> Generator[List[int], None, None]:
    ext = Path(path).suffix.lower()
    if ext == ".npz":
        for window in iter_windows_npz(Path(path)):
            yield window
    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield list(map(int, json.loads(line)))
    elif ext == ".pt":
        data = torch.load(path)
        for s in data:
            yield list(map(int, s))
    else:
        raise ValueError(f"Unsupported file extension for '{path}'.")

def get_train_subset(train_ds: HFDataset, fraction: float, seed: int = 42) -> HFDataset:
    if not 0 < fraction <= 1:
        raise ValueError("vocab_fraction must be in (0, 1].")
    if fraction == 1:
        return train_ds
    n_total = len(train_ds)
    n_keep = max(1, int(round(n_total * fraction)))
    logging.info("Sampling %d/%d (%.2f%%) records for vocab.", n_keep, n_total, 100 * fraction)
    return train_ds.shuffle(seed=seed).select(range(n_keep))


def _doc_indices_factory(
    texts: Sequence[str],
    word2idx: Dict[str, int],
    stop_words: Set[str],
    numbers: Set[str],
    lowercase: bool,
    min_len: int,
):
    @lru_cache(maxsize=200_000)
    def doc_indices(i: int) -> List[int]:
        txt = texts[i]
        if not isinstance(txt, str) or len(txt) < min_len:
            return []
        toks = tokenize_text(txt, stop_words, numbers, lowercase)
        uniq = {t for t in toks if t in word2idx}
        return sorted(word2idx[t] for t in uniq)

    return doc_indices


def generate_representations_for_split(
    ds_split: HFDataset,
    word2idx: Dict[str, int],
    dataset_name: str,
    split_name: str,
    group_size: int,
    stride: int,
    stop_words: Set[str],
    numbers: Set[str],
    lowercase: bool,
    min_len: int,
) -> List[List[int]]:
    texts: List[str] = ds_split["text"]
    n = len(texts)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if group_size > n:
        logging.warning("%s: group_size %d > split size %d; returning empty list.", split_name, group_size, n)
        return []

    doc_indices = _doc_indices_factory(texts, word2idx, stop_words, numbers, lowercase, min_len)

    reps: List[List[int]] = []
    from tqdm import tqdm

    for start in tqdm(range(0, n - group_size + 1, stride), desc=f"{dataset_name}/{split_name} windows"):
        window_sets: List[List[int]] = [doc_indices(i) for i in range(start, start + group_size)]
        uniq: Set[int] = set()
        for s in window_sets:
            if s:
                uniq.update(s)
        reps.append(sorted(uniq))

    logging.info("Generated %d representations for %s/%s.", len(reps), dataset_name, split_name)
    return reps

def find_best_base_dataset(
    save_dir: str,
    dataset_name: str,
    split_name: str,
    target_vocab_size: int,
    target_group_size: int,
) -> Optional[Tuple[str, int, int]]:
    """Find a base file that can be derived without violating group-size constraints.

    Requirements:
    - base vocab >= target vocab
    - base group size <= target group size
    - target group size is a multiple of the base group size (otherwise derive_reps_from_existing will fail)
    """
    candidates: List[Tuple[str, int, int]] = []
    if not os.path.exists(save_dir):
        return None

    for f in os.listdir(save_dir):
        if not f.startswith(f"{dataset_name}_{split_name}_"):
            continue
        details = parse_path_details(f)
        if not details:
            continue
        if not _file_is_valid(os.path.join(save_dir, f)):
            continue
        v_size, g_size, _s = details
        if (
            v_size >= target_vocab_size
            and g_size <= target_group_size
            and target_group_size % g_size == 0
        ):
            candidates.append((os.path.join(save_dir, f), v_size, g_size))

    if not candidates:
        return None

    # Prefer the largest feasible group size (closest to target), then smaller vocab
    base_path, base_v, base_g = sorted(candidates, key=lambda x: (-x[2], x[1]))[0]
    if target_group_size % base_g != 0:
        return None
    logging.info("Found suitable base: %s", os.path.basename(base_path))
    return base_path, base_v, base_g


def derive_reps_from_existing(
    base_path: str,
    base_vocab_size: int,
    base_group_size: int,
    target_vocab_size: int,
    target_group_size: int,
    target_stride: int,
) -> List[List[int]]:
    """Create a new variant by trimming vocab and/or merging windows.

    - Vocab shrink: drop indices ≥ target_vocab_size
    - Group grow: requires target_group_size % base_group_size == 0; we slide
      with step = (target_group_size // base_group_size) * target_stride
    """
    factor = 1
    if target_group_size < base_group_size:
        raise ValueError("Target group size must be ≥ base group size when deriving.")
    if target_group_size > base_group_size:
        if target_group_size % base_group_size != 0:
            raise ValueError(
                f"Target group size ({target_group_size}) must be a multiple of base group size ({base_group_size})."
            )
        factor = target_group_size // base_group_size

    data_iter = iter_sparse(base_path)
    base_samples: List[List[int]] = list(data_iter)
    if not base_samples:
        return []

    reps: List[List[int]] = []
    step = factor * target_stride
    from tqdm import tqdm

    for i in tqdm(range(0, len(base_samples) - factor + 1, step), desc="Deriving reps"):
        window = base_samples[i : i + factor]
        uniq: Set[int] = set()
        for s in window:
            if s:
                if target_vocab_size < base_vocab_size:
                    uniq.update([x for x in s if x < target_vocab_size])
                else:
                    uniq.update(s)
        reps.append(sorted(uniq))

    logging.info("Derived %d representations from %s.", len(reps), os.path.basename(base_path))
    return reps

def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build vocabulary & sparse representations from WikiText-103 or OpenWebText. "
            "Can derive new variants (vocab/group/stride) from existing ones."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=["wikitext", "openwebtext"], default="wikitext", help="Which HF dataset to process.")
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory (None uses HF defaults).")
    p.add_argument("--save_dir", default="./data", help="Where to save the generated files.")


    p.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE, help="Maximum vocabulary size.")
    p.add_argument("--vocab_fraction", type=float, default=0.2, help="Fraction of TRAIN used to build the vocab.")
    p.add_argument("--min_len", type=int, default=50, help="Min char length of doc to include.")
    p.add_argument("--numbers_max", type=int, default=100, help="Keep numeric tokens in [0, numbers_max].")
    p.add_argument("--lowercase", action="store_true", help="Lowercase tokens before filtering.")

    p.add_argument("--group_size", type=int, default=4, help="Number of docs per window.")
    p.add_argument("--stride", type=int, default=1, help="Stride (step) between windows in docs.")

    p.add_argument("--storage_format", choices=sorted(SUPPORTED_FORMATS), default="npz", help="Output format for split files.")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")

    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    set_seed(args.seed)

    numbers = {str(i) for i in range(args.numbers_max + 1)}

    extra_tokens: Set[str] = (
        set(EXTRA_TOKENS)
    )

    dataset_name: str = args.dataset
    dataset: Optional[HFDatasetDict] = None
    stop_words: Optional[Set[str]] = None

    def ensure_dataset_loaded() -> None:
        nonlocal dataset_name, dataset
        if dataset is None:
            dataset_name, dataset = load_dataset_by_name(args.dataset, args.cache_dir)
            logging.info("Dataset '%s' loaded. Splits: %s", dataset_name, list(dataset.keys()))

    def ensure_stop_words() -> None:
        nonlocal stop_words
        if stop_words is None:
            nltk_dir = os.path.join(args.save_dir, DEFAULT_NLTK_DIR)
            stop_words = setup_nltk(nltk_dir)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = save_dir / f"{dataset_name}_vocab_v{args.vocab_size}.pt"
    if vocab_path.exists():
        logging.info("Loading existing vocabulary from %s", vocab_path)
        vocab_data = torch.load(vocab_path, weights_only=False)
        vocab_list = list(vocab_data["vocab_list"])
        word2idx: Dict[str, int] = {str(k): int(v) for k, v in vocab_data["word2idx"].items()}
    else:
        logging.info("Building vocabulary from scratch…")
        ensure_dataset_loaded()
        vocab_path = save_dir / f"{dataset_name}_vocab_v{args.vocab_size}.pt"
        ensure_stop_words()
        train_subset = get_train_subset(dataset["train"], args.vocab_fraction, seed=args.seed)
        train_tokens = collect_tokens(train_subset, stop_words, numbers, min_len=args.min_len, lowercase=args.lowercase)
        vocab_list, word2idx = build_vocab(train_tokens, args.vocab_size, extra_tokens)
        torch.save({"vocab_list": vocab_list, "word2idx": word2idx}, str(vocab_path))
        logging.info("Vocabulary saved → %s", vocab_path)

    for split in ["train", "validation", "test"]:
        target_path = get_save_path(
            str(save_dir), dataset_name, split, args.vocab_size, args.group_size, args.stride, args.storage_format
        )

        if os.path.exists(target_path):
            if _target_matches(target_path, dataset_name, split, args):
                logging.info("Target already exists, skipping: %s", os.path.basename(target_path))
                continue
            else:
                logging.warning("Existing target %s has incompatible metadata; regenerating.", os.path.basename(target_path))

        base_info = find_best_base_dataset(str(save_dir), dataset_name, split, args.vocab_size, args.group_size)

        if base_info:
            base_path, base_v, base_g = base_info
            reps = derive_reps_from_existing(
                base_path=base_path,
                base_vocab_size=base_v,
                base_group_size=base_g,
                target_vocab_size=args.vocab_size,
                target_group_size=args.group_size,
                target_stride=args.stride,
            )
        else:
            logging.info("No suitable base found for %s; generating from raw split.", split)
            ensure_dataset_loaded()
            ensure_stop_words()
            reps = generate_representations_for_split(
                ds_split=dataset[split],
                word2idx=word2idx,
                dataset_name=dataset_name,
                split_name=split,
                group_size=args.group_size,
                stride=args.stride,
                stop_words=stop_words,
                numbers=numbers,
                lowercase=args.lowercase,
                min_len=args.min_len,
            )

        save_sparse(reps, target_path, args.storage_format)

        meta = Meta(
            dataset=canonical_dataset_name(dataset_name),
            split=split,
            vocab_size=args.vocab_size,
            group_size=args.group_size,
            stride=args.stride,
            vocab_path=str(vocab_path),
            extra_tokens=sorted(list(extra_tokens)),
            min_len=args.min_len,
            lowercase=bool(args.lowercase),
            numbers_min=0,
            numbers_max=args.numbers_max,
            seed=args.seed,
            version="0.1.0",
            storage_format=args.storage_format,
            library_versions={
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": torch.__version__,
                "datasets": __import__("datasets").__version__,
                "nltk": nltk.__version__,
            },
        )
        write_metadata(target_path, meta)
        logging.info("Saved %s and metadata.", os.path.basename(target_path))

if __name__ == "__main__":
    main()
