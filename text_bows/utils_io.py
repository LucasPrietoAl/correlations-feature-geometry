from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def path_candidates(
    data_dir: str,
    dataset: str,
    split: str,
    vocab: int,
    group: int,
    stride: Optional[int],
) -> List[Path]:
    base = Path(data_dir)
    if stride is not None:
        return [base / f"{dataset}_{split}_v{vocab}_w{group}_s{stride}.npz"]
    return [base / f"{dataset}_{split}_v{vocab}_w{group}_s{s}.npz" for s in range(1, 65)]


def find_existing_path(cands: List[Path]) -> Optional[Path]:
    for path in cands:
        if path.exists():
            return path
    return None


def meta_for(path: Path) -> Optional[dict]:
    """Assumes sidecar metadata at <file>.meta.json."""
    meta_path = Path(str(path.with_suffix("")) + ".meta.json")
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_windows_npz(npz_path: Path) -> Generator[List[int], None, None]:
    data = np.load(str(npz_path), mmap_mode="r")
    offsets = data["offsets"]
    values = data["values"]
    n = int(offsets.shape[0]) - 1
    for i in range(n):
        a = int(offsets[i])
        b = int(offsets[i + 1])
        yield values[a:b].astype(np.int64).tolist()


class SparseNPZDataset(Dataset):
    def __init__(self, npz_path: str):
        self.path = Path(npz_path)
        self._npz = np.load(self.path, mmap_mode="r")
        self.offsets = self._npz["offsets"]
        self.values = self._npz["values"]

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def __getitem__(self, idx: int) -> List[int]:
        a = int(self.offsets[idx])
        b = int(self.offsets[idx + 1])
        return self.values[a:b].astype(np.int64).tolist()

    def vocab_size(self) -> int:
        if self.values.size == 0:
            return 0
        return int(self.values.max()) + 1


def load_vocab_map(vocab_path: str) -> Dict[str, int]:
    blob = torch.load(vocab_path, map_location="cpu", weights_only=False)
    if "word2idx" in blob:
        return {str(k): int(v) for k, v in blob["word2idx"].items()}
    vocab_list = list(blob.get("vocab_list", []))
    return {str(w): i for i, w in enumerate(vocab_list)}


def load_encoder_weight_matrix(model_path: str) -> np.ndarray:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    weight = state.get("enc.weight")
    if weight is None:
        raise KeyError(f"'enc.weight' missing in checkpoint: {model_path}")
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()
    return weight.T

