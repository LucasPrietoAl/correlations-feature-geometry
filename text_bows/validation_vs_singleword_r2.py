from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from text_bows.train_autoencoder import Autoencoder, collate_indices, dense_target
from text_bows.utils_io import SparseNPZDataset, find_existing_path, meta_for, path_candidates


def load_split(
    data_dir: str,
    dataset: str,
    split: str,
    vocab_size: int,
    group_size: int,
    stride: Optional[int],
    batch_size: int,
) -> Tuple[Optional[DataLoader], Optional[Path], int]:
    path = find_existing_path(path_candidates(data_dir, dataset, split, vocab_size, group_size, stride))
    if path is None:
        return None, None, vocab_size

    dataset_obj = SparseNPZDataset(str(path))
    meta = meta_for(path)
    vocab = int(meta["vocab_size"]) if (meta and "vocab_size" in meta) else vocab_size
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, collate_fn=collate_indices)
    return loader, path, vocab


def load_idx_to_word(vocab_path: Path) -> Dict[int, str]:
    if not vocab_path.exists():
        return {}
    blob = torch.load(vocab_path, map_location="cpu", weights_only=False)
    if "word2idx" in blob:
        return {int(v): str(k) for k, v in blob["word2idx"].items()}
    vocab_list = list(blob.get("vocab_list", []))
    return {i: str(word) for i, word in enumerate(vocab_list)}


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[Autoencoder, int, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    if "enc.weight" not in state:
        raise KeyError(f"'enc.weight' missing in checkpoint: {ckpt_path}")

    hid, inp = state["enc.weight"].shape
    args = ckpt.get("args", {})
    model = Autoencoder(
        inp=inp,
        hid=hid,
        relu_latent=bool(args.get("relu_latent", False)),
        linear_out=bool(args.get("linear_out", False)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, inp, hid


def init_acc(vocab_size: int) -> Dict[str, np.ndarray | int]:
    return {
        "sum_y": np.zeros(vocab_size, dtype=np.float64),
        "sum_y2": np.zeros(vocab_size, dtype=np.float64),
        "ss_res": np.zeros(vocab_size, dtype=np.float64),
        "n": 0,
    }


def update_acc(acc: Dict[str, np.ndarray | int], y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    yt = y_true.detach().cpu().double().numpy()
    yp = y_pred.detach().cpu().double().numpy()
    acc["sum_y"] += yt.sum(axis=0)
    acc["sum_y2"] += (yt * yt).sum(axis=0)
    acc["ss_res"] += ((yt - yp) ** 2).sum(axis=0)
    acc["n"] += int(yt.shape[0])


def finalize_r2(acc: Dict[str, np.ndarray | int]) -> np.ndarray:
    n = max(1, int(acc["n"]))
    mean = acc["sum_y"] / n
    ss_tot = acc["sum_y2"] - acc["sum_y"] * mean
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_tot = np.where(ss_tot <= 1e-12, np.nan, ss_tot)
        r2 = 1.0 - (acc["ss_res"] / safe_tot)
    return np.where(np.isfinite(r2), r2, np.nan)


@torch.no_grad()
def compute_r2_and_frequency_from_loaders(
    model: Autoencoder,
    loaders: Iterable[DataLoader],
    vocab_size: int,
    device: torch.device,
    max_batches: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    acc = init_acc(vocab_size)
    processed_batches = 0
    for loader in loaders:
        for xs, cnts in loader:
            if max_batches is not None and processed_batches >= max_batches:
                break
            target = dense_target(xs, cnts, vocab_size, device=device, dtype=torch.float32)
            pred = model(xs, cnts).to(dtype=torch.float32)
            update_acc(acc, target, pred)
            processed_batches += 1
        if max_batches is not None and processed_batches >= max_batches:
            break
    return finalize_r2(acc), acc["sum_y"].copy(), int(acc["n"])


@torch.no_grad()
def compute_single_word_r2(
    model: Autoencoder,
    vocab_size: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    acc = init_acc(vocab_size)
    for start in range(0, vocab_size, batch_size):
        end = min(vocab_size, start + batch_size)
        xs = [torch.tensor([idx], dtype=torch.long) for idx in range(start, end)]
        cnts = torch.ones(end - start, dtype=torch.long)
        target = dense_target(xs, cnts, vocab_size, device=device, dtype=torch.float32)
        pred = model(xs, cnts).to(dtype=torch.float32)
        update_acc(acc, target, pred)
    return finalize_r2(acc)


def save_table(
    out_csv: Path,
    idx_to_word: Dict[int, str],
    freq_count: np.ndarray,
    freq_rate: np.ndarray,
    r2_val: np.ndarray,
    r2_single: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "word",
                "freq_count",
                "freq_rate",
                "r2_validation",
                "r2_single_word",
                "r2_delta_single_minus_val",
            ]
        )
        for idx in range(len(r2_val)):
            val = r2_val[idx]
            single = r2_single[idx]
            writer.writerow(
                [
                    idx,
                    idx_to_word.get(idx, f"<unk-{idx}>"),
                    float(freq_count[idx]),
                    float(freq_rate[idx]),
                    float(val) if np.isfinite(val) else "",
                    float(single) if np.isfinite(single) else "",
                    float(single - val) if np.isfinite(val) and np.isfinite(single) else "",
                ]
            )


def save_gap_table(
    out_csv: Path,
    idx_to_word: Dict[int, str],
    freq_count: np.ndarray,
    r2_val: np.ndarray,
    r2_single: np.ndarray,
) -> None:
    valid_rows = []
    for idx in range(len(r2_val)):
        rv = r2_val[idx]
        rs = r2_single[idx]
        freq = freq_count[idx]
        if not (np.isfinite(rv) and np.isfinite(rs) and np.isfinite(freq)):
            continue
        valid_rows.append(
            {
                "word": idx_to_word.get(idx, f"<unk-{idx}>"),
                "freq_count": float(freq),
                "r2_valtest": float(rv),
                "r2_onehot": float(rs),
                "delta_val_minus_onehot": float(rv - rs),
            }
        )

    valid_rows.sort(key=lambda row: (-row["delta_val_minus_onehot"], -row["freq_count"], row["word"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "freq_count", "r2_valtest", "r2_onehot", "delta_val_minus_onehot"])
        for rank, row in enumerate(valid_rows, start=1):
            writer.writerow(
                [
                    rank,
                    row["word"],
                    row["freq_count"],
                    row["r2_valtest"],
                    row["r2_onehot"],
                    row["delta_val_minus_onehot"],
                ]
            )


def save_scatter(
    out_png: Path,
    freq_count: np.ndarray,
    r2_val: np.ndarray,
    r2_single: np.ndarray,
    ls: int,
    eval_label: str,
) -> None:
    mask = (freq_count > 0) & np.isfinite(r2_val) & np.isfinite(r2_single)
    x = freq_count[mask]
    y_val = r2_val[mask]
    y_single = r2_single[mask]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.scatter(x, y_val, s=10, alpha=0.35, color="#1f77b4", label=f"{eval_label} R2")
    ax.scatter(x, y_single, s=10, alpha=0.35, color="#ff7f0e", label="Single-word-input R2")
    ax.set_xscale("log")
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"Word frequency in {eval_label.lower()} windows (count)")
    ax.set_ylabel("Per-word R2")
    ax.set_title(f"Per-word R2 vs frequency ({eval_label}, latent size = {ls})")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def split_tag(split: str) -> str:
    return "valtest" if split == "val_test" else split


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-word R2 on dataset windows vs one-word inputs for a trained AE checkpoint."
    )
    parser.add_argument("--ls", type=int, required=True, help="Latent size (used in checkpoint template).")
    parser.add_argument(
        "--checkpoint_template",
        default="./text_bows/models/ae_wikitext_v10000_w20_seed1_L{ls}_mse_wd1.0_seed1.pt",
        help="Checkpoint template; should contain '{ls}'.",
    )
    parser.add_argument("--data_dir", default="./text_bows/data")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--group_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--split", default="validation", choices=["validation", "test", "train", "val_test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--single_word_batch_size", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="", help="cpu, cuda, cuda:0, ... (default: cuda if available)")
    parser.add_argument("--out_dir", default="./text_bows/figures")
    args = parser.parse_args(argv)

    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    ckpt_path = Path(args.checkpoint_template.format(ls=args.ls))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    model, model_vocab, model_hid = load_model_from_checkpoint(ckpt_path, device=device)
    print(f"Loaded model: vocab={model_vocab}, latent={model_hid}, device={device}")

    loaders: List[DataLoader] = []
    used_splits: List[str] = []
    used_paths: List[Path] = []
    if args.split == "val_test":
        split_order = ["validation", "test"]
        for split in split_order:
            loader, path, vocab = load_split(
                data_dir=args.data_dir,
                dataset=args.dataset,
                split=split,
                vocab_size=args.vocab_size,
                group_size=args.group_size,
                stride=args.stride,
                batch_size=args.batch_size,
            )
            if loader is None or path is None:
                raise FileNotFoundError(f"Could not find required split '{split}' for combined val_test mode.")
            if vocab != model_vocab:
                raise ValueError(
                    f"Vocab mismatch on split '{split}': dataset vocab={vocab}, model vocab={model_vocab}."
                )
            loaders.append(loader)
            used_splits.append(split)
            used_paths.append(path)
    else:
        split_order = [args.split] + [s for s in ["validation", "test", "train"] if s != args.split]
        for split in split_order:
            loader, path, vocab = load_split(
                data_dir=args.data_dir,
                dataset=args.dataset,
                split=split,
                vocab_size=args.vocab_size,
                group_size=args.group_size,
                stride=args.stride,
                batch_size=args.batch_size,
            )
            if loader is None or path is None:
                continue
            if vocab != model_vocab:
                raise ValueError(
                    f"Vocab mismatch on split '{split}': dataset vocab={vocab}, model vocab={model_vocab}."
                )
            loaders = [loader]
            used_splits = [split]
            used_paths = [path]
            break
        if not loaders:
            raise FileNotFoundError("Could not find any split file among validation/test/train for the given settings.")

    for split, path in zip(used_splits, used_paths):
        print(f"Using split '{split}' from {path}")

    eval_name = "Validation+Test" if args.split == "val_test" else used_splits[0].capitalize()
    print(f"Computing per-word {eval_name} R2 and frequencies...")
    r2_val, freq_count, n_samples = compute_r2_and_frequency_from_loaders(
        model=model,
        loaders=loaders,
        vocab_size=model_vocab,
        device=device,
        max_batches=args.max_batches,
    )
    freq_rate = freq_count / max(1, n_samples)

    print("Computing per-word R2 for one-word inputs...")
    r2_single = compute_single_word_r2(
        model=model,
        vocab_size=model_vocab,
        device=device,
        batch_size=args.single_word_batch_size,
    )

    out_dir = Path(args.out_dir)
    tag = split_tag(args.split)
    if args.split == "validation":
        stem = f"r2_validation_vs_singleword_ls{args.ls}"
    else:
        stem = f"r2_{tag}_vs_singleword_ls{args.ls}"
    out_csv = out_dir / f"{stem}.csv"
    out_png = out_dir / f"{stem}.png"
    gap_csv = out_dir / f"r2_gap_{tag}_minus_onehot_ls{args.ls}_sorted.csv"

    vocab_path = Path(args.data_dir) / f"{args.dataset}_vocab_v{model_vocab}.pt"
    idx_to_word = load_idx_to_word(vocab_path)
    save_table(out_csv, idx_to_word, freq_count, freq_rate, r2_val, r2_single)
    save_scatter(out_png, freq_count, r2_val, r2_single, args.ls, eval_label=eval_name)
    save_gap_table(gap_csv, idx_to_word, freq_count, r2_val, r2_single)

    valid_mask = (freq_count > 0) & np.isfinite(r2_val) & np.isfinite(r2_single)
    print(f"Evaluated {eval_name.lower()} samples: {n_samples}")
    print(f"Words with frequency>0 and finite R2 in both settings: {int(valid_mask.sum())}")
    print(f"Mean {eval_name.lower()} R2 (valid words): {np.nanmean(r2_val[valid_mask]):.6f}")
    print(f"Mean single-word R2 (valid words): {np.nanmean(r2_single[valid_mask]):.6f}")
    print(f"Saved table: {out_csv}")
    print(f"Saved scatter: {out_png}")
    print(f"Saved gap table: {gap_csv}")


if __name__ == "__main__":
    main()
