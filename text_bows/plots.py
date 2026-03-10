from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from text_bows.constants import SEMANTIC_CATEGORIES as _SEMANTIC_CATEGORIES
from text_bows.utils_io import (
    SparseNPZDataset,
    find_existing_path,
    iter_windows_npz,
    load_encoder_weight_matrix,
    load_vocab_map,
    meta_for,
    path_candidates,
)


RANDOM_SEED = 42
MAX_SAMPLES_FOR_PCA = 500_000
DEFAULT_TEXT_BOWS_DIR = Path("./text_bows")
DEFAULT_DATA_DIR = str(DEFAULT_TEXT_BOWS_DIR / "data")
DEFAULT_MODELS_DIR = str(DEFAULT_TEXT_BOWS_DIR / "models")
DEFAULT_FIGURES_DIR = str(DEFAULT_TEXT_BOWS_DIR / "figures")
DEFAULT_TEXT_DATASET = "wikitext"
DEFAULT_TEXT_VOCAB_SIZE = 10000
DEFAULT_TEXT_GROUP_SIZE = 20
DEFAULT_TEXT_STRIDE = 1

MONTHS_ORDER = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

DEFAULT_EXPERIMENTS = [
    {
        "label": "wikitext",
        "dataset": "wikitext",
        "vocab_size": 10000,
        "group_size": 20,
        "stride": 1,
        "vocab_path": str(Path(DEFAULT_DATA_DIR) / "wikitext_vocab_v10000.pt"),
        "ckpt_path": str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L1000_mse_wd4.0_seed1.pt"),
    },
]

DEFAULT_SEASONAL_LS = 1000
DEFAULT_SEASONAL_VOCAB_PATH = str(Path(DEFAULT_DATA_DIR) / "wikitext_vocab_v10000.pt")
DEFAULT_SEASONAL_MODEL_TEMPLATE = str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L{ls}_mse_wd4.0_seed1.pt")
DEFAULT_PLOTS_ROOT = DEFAULT_FIGURES_DIR

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_ABBREVS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_SEASONAL_WORDS = [
    "snow",
    "freezing",
    "Christmas",
    "summer",
    "spring",
    "Easter",
    "leaves",
    "Halloween",
    "Thanksgiving",
]

STRUCTURE_LATENT_SIZES = [50, 75, 100, 125, 150, 200, 250, 300, 400, 600, 800, 1000, 2000, 4000, 6000, 10000]
STRUCTURE_WORD_GROUPS = {
    "Months": {
        "words": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "color": "#1f77b4",
        "filename": "Months",
    },
    "Roman numerals": {
        "words": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"],
        "labels": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"],
        "color": "#ff7f0e",
        "filename": "Roman numerals",
    },
}
DEFAULT_STRUCTURE_MODEL_TEMPLATE = str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L{ls}_mse_wd1.0_seed1.pt")
DEFAULT_STRUCTURE_MULTI_SEED_TEMPLATES = [
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L{ls}_mse_wd1.0_seed1.pt"),
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed2_L{ls}_mse_wd1.0_seed2.pt"),
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed3_L{ls}_mse_wd1.0_seed3.pt"),
]
DEFAULT_STRUCTURE_METRICS_PATH = str(Path(DEFAULT_FIGURES_DIR) / "metrics_curves3.pdf")
DEFAULT_STRUCTURE_METRICS_FILL_PATH = str(Path(DEFAULT_FIGURES_DIR) / "metrics_curves_fill.pdf")


_MONTH_SEASON_COLORS = np.array(
    [
        [0.88, 0.93, 1.00],
        [0.88, 0.93, 1.00],
        [0.55, 0.85, 0.45],
        [0.35, 0.78, 0.30],
        [0.20, 0.72, 0.20],
        [1.00, 0.95, 0.25],
        [1.00, 0.88, 0.05],
        [1.00, 0.88, 0.15],
        [0.72, 0.42, 0.10],
        [0.62, 0.32, 0.06],
        [0.52, 0.25, 0.06],
        [0.88, 0.93, 1.00],
    ],
    dtype=float,
)

_WINTER = _MONTH_SEASON_COLORS[0]
_SPRING = _MONTH_SEASON_COLORS[3]
_SUMMER = _MONTH_SEASON_COLORS[6]
_AUTUMN = _MONTH_SEASON_COLORS[9]

_SEASONAL_WORD_COLORS: Dict[str, np.ndarray] = {
    "snow": _WINTER,
    "freezing": _WINTER,
    "Christmas": _WINTER,
    "summer": _SUMMER,
    "spring": _SPRING,
    "Easter": _SPRING,
    "leaves": _AUTUMN,
    "Halloween": _AUTUMN,
    "Thanksgiving": _AUTUMN,
}

_WINTER_MONTHS = {0, 1, 11}
_WINTER_WORDS = {"snow", "freezing", "Christmas"}
_WINTER_LABEL_COLOR = np.array([0.35, 0.35, 0.42])


def _triangle_color(i: int, j: int, alpha: float = 0.45) -> tuple:
    c = 0.5 * (_MONTH_SEASON_COLORS[i % 12] + _MONTH_SEASON_COLORS[j % 12])
    return (*c, alpha)


def _month_color(month_idx: int) -> np.ndarray:
    if month_idx in _WINTER_MONTHS:
        return _WINTER_LABEL_COLOR
    return np.clip(_MONTH_SEASON_COLORS[month_idx] * 0.65, 0, 1)


def _seasonal_word_color(word: str) -> np.ndarray:
    if word in _WINTER_WORDS:
        return _WINTER_LABEL_COLOR
    base = _SEASONAL_WORD_COLORS.get(word, np.array([0.2, 0.5, 0.2]))
    return np.clip(base * 0.65, 0, 1)


def _off_diag_fro_norm(mat: np.ndarray) -> float:
    return float(np.linalg.norm(mat - np.diag(np.diag(mat)), ord="fro"))


def _pca_to_2d(vecs: np.ndarray) -> Tuple[np.ndarray, float]:
    if vecs.shape[1] == 2:
        return vecs, 1.0
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(vecs)
    return coords, float(pca.explained_variance_ratio_.sum())


def _plot_structure_group(
    coords: np.ndarray,
    labels: List[str],
    out_path: Path,
    color: str,
    label_fontsize: int = 35,
    label_offset: float = 0.15,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    origin = np.zeros(2)
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        ax.fill(
            [origin[0], coords[i, 0], coords[j, 0]],
            [origin[1], coords[i, 1], coords[j, 1]],
            facecolor=color,
            alpha=0.15,
            edgecolor=color,
            linewidth=1.5,
            zorder=1,
        )

    for x, y in coords:
        ax.plot([0, x], [0, y], color="gray", lw=0.8, alpha=0.6, zorder=2)

    closed = np.vstack([coords, coords[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.5, alpha=0.6, zorder=3)
    ax.scatter(coords[:, 0], coords[:, 1], facecolor=color, edgecolor="white", linewidth=0.8, s=70, zorder=4)
    ax.scatter(0, 0, color="red", s=50, zorder=5)

    for (x_coord, y_coord), word in zip(coords, labels):
        dist = float(np.hypot(x_coord, y_coord))
        if dist > 1e-9:
            dx = (x_coord / dist) * label_offset
            dy = (y_coord / dist) * label_offset
        else:
            dx, dy = label_offset, 0.0
        ax.text(
            x_coord + dx,
            y_coord + dy,
            word,
            fontsize=label_fontsize,
            color="black",
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, ls="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)
    ax.margins(0.30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _build_structure_group_indices(w2i: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    group_idxs: Dict[str, np.ndarray] = {}
    display_labels: Dict[str, List[str]] = {}
    for group_name, cfg in STRUCTURE_WORD_GROUPS.items():
        present_idxs = []
        present_labels = []
        for word, label in zip(cfg["words"], cfg["labels"]):
            idx = lookup_ci(w2i, word)
            if idx is not None:
                present_idxs.append(idx)
                present_labels.append(label)
        if not present_idxs:
            raise RuntimeError(f"None of the words in group '{group_name}' were found in the vocabulary.")
        group_idxs[group_name] = np.array(present_idxs, dtype=np.int64)
        display_labels[group_name] = present_labels
    return group_idxs, display_labels


def run_structure(args: argparse.Namespace) -> None:
    vocab_path = Path(args.vocab_path)
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    w2i = load_vocab_map(str(vocab_path))
    group_idxs, display_labels = _build_structure_group_indices(w2i)

    latent_sizes = list(args.latent_sizes)
    metrics: Dict[str, List[float]] = {group: [] for group in STRUCTURE_WORD_GROUPS}
    runs: Dict[str, List[List[float]]] = {group: [[] for _ in latent_sizes] for group in STRUCTURE_WORD_GROUPS}
    out_root = Path(args.plots_root)

    for ls_i, ls in enumerate(latent_sizes):
        model_path = Path(args.model_path_template.format(ls=ls))
        if model_path.is_file():
            W = load_encoder_weight_matrix(str(model_path))
            for group_name, idxs in group_idxs.items():
                Wg = W[idxs, :]
                gram = Wg @ Wg.T
                metrics[group_name].append(_off_diag_fro_norm(gram))
                coords, _ = _pca_to_2d(Wg)
                out_name = f"{STRUCTURE_WORD_GROUPS[group_name]['filename']}_ls{ls}.pdf"
                _plot_structure_group(
                    coords=coords,
                    labels=display_labels[group_name],
                    out_path=out_root / out_name,
                    color=str(STRUCTURE_WORD_GROUPS[group_name]["color"]),
                )
        else:
            print(f"Warning: model not found → {model_path}")
            for group_name in STRUCTURE_WORD_GROUPS:
                metrics[group_name].append(np.nan)

        for ckpt_template in args.multi_seed_templates:
            ckpt_path = Path(ckpt_template.format(ls=ls))
            if not ckpt_path.is_file():
                continue
            W = load_encoder_weight_matrix(str(ckpt_path))
            for group_name, idxs in group_idxs.items():
                Wg = W[idxs, :]
                runs[group_name][ls_i].append(_off_diag_fro_norm(Wg @ Wg.T))

    mean_metrics = {group: [float(np.nanmean(xs)) if xs else np.nan for xs in group_runs] for group, group_runs in runs.items()}
    std_metrics = {
        group: [float(np.nanstd(xs, ddof=1)) if len(xs) > 1 else 0.0 for xs in group_runs]
        for group, group_runs in runs.items()
    }

    metrics_path = Path(args.metrics_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel("Latent dimension", fontsize=18)
    ax.set_xscale("log")
    ax.set_ylabel(r"$\|\mathbf{G} - \mathrm{diag}(\mathbf{G})\|_F$", fontsize=18)
    ax.grid(True, ls="--", alpha=0.4, zorder=0.5)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    for group_name in STRUCTURE_WORD_GROUPS:
        color = str(STRUCTURE_WORD_GROUPS[group_name]["color"])
        ax.plot(latent_sizes, metrics[group_name], marker="o", label=group_name, color=color, zorder=3, linewidth=3.0)
    for snap_lat, snap_color, snap_label in [(400, "green", "Snapshot 1"), (4000, "red", "Snapshot 2")]:
        if snap_lat in latent_sizes:
            ax.axvline(x=snap_lat, color=snap_color, linestyle=":", linewidth=6, label=snap_label, zorder=2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=12, frameon=True, framealpha=0.9)
    fig.tight_layout()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(metrics_path, bbox_inches="tight")
    plt.close(fig)

    metrics_fill_path = Path(args.metrics_fill_path)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlabel("Latent dimension", fontsize=18)
    ax.set_xscale("log")
    ax.set_ylabel(r"$\|\mathbf{G} - \mathrm{diag}(\mathbf{G})\|_F$", fontsize=18)
    ax.grid(True, ls="--", alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    for group_name in STRUCTURE_WORD_GROUPS:
        color = str(STRUCTURE_WORD_GROUPS[group_name]["color"])
        y = np.array(mean_metrics[group_name], dtype=np.float64)
        yerr = 2.0 * np.array(std_metrics[group_name], dtype=np.float64)
        ax.plot(latent_sizes, y, "o-", lw=2.0, color=color, label=group_name)
        ax.fill_between(latent_sizes, y - yerr, y + yerr, color=color, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, framealpha=0.9)
    fig.tight_layout()
    metrics_fill_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(metrics_fill_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {metrics_fill_path}")


def month_indices_from_vocab(vocab_map: dict) -> Tuple[List[str], List[int]]:
    names, idxs = [], []
    for month in MONTHS_ORDER:
        title = month.title()
        if title in vocab_map:
            names.append(title)
            idxs.append(vocab_map[title])
        elif month in vocab_map:
            names.append(month)
            idxs.append(vocab_map[month])
    return names, idxs


def abbreviate_months(names: List[str]) -> List[str]:
    return [name[:3].title() for name in names]


def build_month_activity_matrix(
    ds: SparseNPZDataset,
    month_idxs: List[int],
    max_samples: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    month_idx_set = set(month_idxs)
    rows: List[List[int]] = []
    for i in range(len(ds)):
        toks = ds[i]
        if month_idx_set.intersection(toks):
            toks_set = set(toks)
            rows.append([1 if m in toks_set else 0 for m in month_idxs])
    X = np.array(rows, dtype=np.float32)
    if max_samples is not None and X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
    return X


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    if X.shape[0] < 2:
        return np.full((X.shape[1], X.shape[1]), np.nan, dtype=np.float32)
    return np.corrcoef(X, rowvar=False).astype(np.float32)


def _draw_month_polygon(
    ax: plt.Axes,
    coords: np.ndarray,
    month_names: List[str],
    label_offset_frac: float = 0.25,
) -> None:
    base_blue = "#1f77b4"
    light_blue = "#9ecae1"
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        ax.fill(
            [0, coords[i, 0], coords[j, 0]],
            [0, coords[i, 1], coords[j, 1]],
            facecolor=light_blue,
            edgecolor="none",
            alpha=0.45,
            zorder=1,
        )
    for x, y in coords:
        ax.plot([0, x], [0, y], color=base_blue, lw=1.2, alpha=0.95, zorder=2)
    closed = np.vstack([coords, coords[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=base_blue, lw=1.2, zorder=2)
    ax.scatter(coords[:, 0], coords[:, 1], s=320, color=base_blue, edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter([0], [0], s=120, color="red", zorder=4)
    rmax = max(1e-8, np.linalg.norm(coords, axis=1).max())
    label_offset = label_offset_frac * rmax
    for (x, y), name in zip(coords, month_names):
        dist = np.sqrt(x**2 + y**2)
        if dist > 1e-9:
            dx, dy = (x / dist) * label_offset, (y / dist) * label_offset
        else:
            dx, dy = label_offset, 0
        ax.text(x + dx, y + dy, name, fontsize=46, fontweight="bold", color=base_blue, ha="center", va="center", zorder=5)


def plot_correlation_matrix(month_names: List[str], corr: np.ndarray, label: str, out_dir: Path) -> None:
    off_diag = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])
    vmax = float(np.percentile(off_diag, 95))
    vmax = max(0.3, min(0.6, vmax))

    fig = plt.figure(figsize=(7.5, 6.5))
    im = plt.imshow(corr, interpolation="nearest", cmap="coolwarm", vmin=0.0, vmax=vmax)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=26, fontsize=25)
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(range(len(month_names)), month_names, rotation=45, ha="right", fontsize=20)
    plt.yticks(range(len(month_names)), month_names, fontsize=20)
    plt.tight_layout()
    fig_path = out_dir / f"months_corr_matrix__{label}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def plot_month_pca_from_data(X: np.ndarray, month_names: List[str], label: str, out_dir: Path) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(X - X.mean(axis=1, keepdims=True))
    I = np.eye(len(month_names), dtype=np.float32)
    coords = (I - pca.mean_) @ pca.components_.T

    fig, ax = plt.subplots(figsize=(14, 14))
    _draw_month_polygon(ax, coords, month_names, label_offset_frac=0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    ax.grid(True, ls="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(0.25)
    fig_path = out_dir / f"months_pca_data__{label}.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def plot_month_pca_from_weights(
    ckpt_path: str,
    month_idxs: List[int],
    month_names: List[str],
    label: str,
    out_dir: Path,
) -> None:
    W = load_encoder_weight_matrix(ckpt_path)
    X = W[month_idxs, :]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(Xn)

    fig, ax = plt.subplots(figsize=(14, 14))
    _draw_month_polygon(ax, coords, month_names)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    ax.grid(True, ls="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(0.25)
    fig_path = out_dir / f"months_pca_weights__{label}.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def _compute_month_analysis(
    experiments: List[dict],
    base_data_dir: str,
    max_samples_for_pca: int,
    seed: int,
    figures_dir: Path,
    run_corr: bool,
    run_data_pca: bool,
) -> Dict[str, dict]:
    month_analysis: Dict[str, dict] = {}
    for cfg in experiments:
        label = cfg["label"]
        dataset = cfg["dataset"]
        print(f"\n=== {label} ({dataset}) ===")

        train_path = find_existing_path(
            path_candidates(base_data_dir, dataset, "train", cfg["vocab_size"], cfg["group_size"], cfg.get("stride"))
        )
        if train_path is None:
            print(f"  Warning: training NPZ not found – skipping {label}.")
            continue

        meta = meta_for(train_path)
        if meta and "vocab_path" in meta:
            vocab_map = load_vocab_map(meta["vocab_path"])
        elif "vocab_path" in cfg:
            vocab_map = load_vocab_map(cfg["vocab_path"])
        else:
            print(f"  Warning: no vocab_path for {label} – skipping.")
            continue

        month_full, month_idxs = month_indices_from_vocab(vocab_map)
        if len(month_idxs) != 12:
            print(f"  Warning: found {len(month_idxs)}/12 months – skipping.")
            continue

        month_names_abbrev = abbreviate_months(month_full)
        ds = SparseNPZDataset(str(train_path))
        X = build_month_activity_matrix(ds, month_idxs, max_samples=max_samples_for_pca, seed=seed)

        print(f"  Windows with ≥1 month: {X.shape[0]}")
        if X.shape[0] < 2:
            print("  Warning: not enough samples – skipping.")
            continue

        if run_corr or run_data_pca:
            corr = correlation_matrix(X)
            if run_corr:
                plot_correlation_matrix(month_names_abbrev, corr, label, figures_dir)
            if run_data_pca:
                plot_month_pca_from_data(X, month_names_abbrev, label, figures_dir)

        month_analysis[label] = {
            "config": dict(cfg),
            "month_idxs": month_idxs,
            "month_names": month_names_abbrev,
        }

    return month_analysis


def generate_seasonal_square(
    ls: int = DEFAULT_SEASONAL_LS,
    vocab_path: str = DEFAULT_SEASONAL_VOCAB_PATH,
    model_path_template: str = DEFAULT_SEASONAL_MODEL_TEMPLATE,
    plots_root: str = DEFAULT_PLOTS_ROOT,
    seasonal_words: Optional[List[str]] = None,
) -> None:
    seasonal_words = seasonal_words or list(DEFAULT_SEASONAL_WORDS)

    output_2d = os.path.join(plots_root, f"months_with_seasons_ls{ls}_square.pdf")

    print(f"Loading vocabulary → {vocab_path}")
    word2idx = load_vocab_map(vocab_path)
    print("Vocab size:", len(word2idx))

    month_idxs = np.array([word2idx[m] for m in MONTH_NAMES if m in word2idx])
    present_months = [m for m in MONTH_NAMES if m in word2idx]
    present_abbrevs = [MONTH_ABBREVS[MONTH_NAMES.index(m)] for m in present_months]
    present_month_indices = [MONTH_NAMES.index(m) for m in present_months]
    print(f"Months present: {present_months}")

    seasonal_present = [w for w in seasonal_words if w in word2idx]
    seasonal_missing = [w for w in seasonal_words if w not in word2idx]
    seasonal_idxs = np.array([word2idx[w] for w in seasonal_present])
    print(f"Seasonal words present ({len(seasonal_present)}): {seasonal_present}")
    if seasonal_missing:
        print(f"Seasonal words missing ({len(seasonal_missing)}): {seasonal_missing}")

    model_path = model_path_template.format(ls=ls)
    W = load_encoder_weight_matrix(model_path)
    print(f"Weight matrix shape: {W.shape}")

    W_months = W[month_idxs, :]
    W_seasonal = W[seasonal_idxs, :] if len(seasonal_idxs) else np.zeros((0, W.shape[1]), dtype=W.dtype)

    pca3 = PCA(n_components=3)
    month_coords_3d = pca3.fit_transform(W_months)
    seasonal_coords_3d = pca3.transform(W_seasonal) if len(seasonal_idxs) else np.zeros((0, 3), dtype=np.float32)
    var3 = pca3.explained_variance_ratio_
    print(f"PCA variance explained (3 PCs): {var3.sum():.2%}  ({var3[0]:.1%}, {var3[1]:.1%}, {var3[2]:.1%})")

    month_coords = month_coords_3d[:, :2]
    seasonal_coords = seasonal_coords_3d[:, :2]

    if len(seasonal_idxs):
        W_seasonal_centered = W_seasonal - pca3.mean_
        reconstructed_2d = seasonal_coords @ pca3.components_[:2]
        residual_2d = W_seasonal_centered - reconstructed_2d
        total_variance = np.sum(W_seasonal_centered**2, axis=1)
        residual_variance_2d = np.sum(residual_2d**2, axis=1)
        seasonal_var_ratios = 1 - (residual_variance_2d / (total_variance + 1e-10))
    else:
        seasonal_var_ratios = np.array([], dtype=np.float32)

    print("\nSeasonal words variance explained by month PCs (2D):")
    for word, value in zip(seasonal_present, seasonal_var_ratios):
        print(f"  {word}: {value:.2%}")

    os.makedirs(plots_root, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 14))

    origin = np.zeros(2)
    n = len(month_coords)
    for i in range(n):
        j = (i + 1) % n
        mi = present_month_indices[i]
        mj = present_month_indices[j]
        tri = np.array([origin, month_coords[i], month_coords[j]])
        fc = _triangle_color(mi, mj, alpha=0.45)
        patch = plt.Polygon(tri, facecolor=fc, edgecolor=(*fc[:3], 0.0), linewidth=0, zorder=1)
        ax.add_patch(patch)

    for (x, y), mi in zip(month_coords, present_month_indices):
        ax.plot([0, x], [0, y], color=_month_color(mi), lw=0.8, alpha=0.6, zorder=2)

    for i in range(n):
        j = (i + 1) % n
        mi = present_month_indices[i]
        mj = present_month_indices[j]
        seg_color = np.clip(0.5 * (_MONTH_SEASON_COLORS[mi] + _MONTH_SEASON_COLORS[mj]) * 0.75, 0, 1)
        ax.plot(
            [month_coords[i, 0], month_coords[j, 0]],
            [month_coords[i, 1], month_coords[j, 1]],
            color=seg_color,
            lw=3.0,
            alpha=0.8,
            zorder=3,
        )

    month_dot_colors = [_month_color(mi) for mi in present_month_indices]
    ax.scatter(
        month_coords[:, 0],
        month_coords[:, 1],
        c=month_dot_colors,
        edgecolor="white",
        linewidth=1.5,
        s=320,
        zorder=4,
        label="Months",
    )
    ax.scatter(0, 0, color="red", s=120, zorder=5)

    label_offset = 0.10
    for (x, y), label in zip(month_coords, present_abbrevs):
        dist = np.sqrt(x**2 + y**2)
        if dist > 1e-9:
            dx, dy = (x / dist) * label_offset, (y / dist) * label_offset
        else:
            dx, dy = label_offset, 0
        mi = present_month_indices[present_abbrevs.index(label)]
        ax.text(
            x + dx,
            y + dy,
            label,
            fontsize=46,
            color=_month_color(mi),
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    if len(seasonal_coords):
        seasonal_dot_colors = [_seasonal_word_color(w) for w in seasonal_present]
        ax.scatter(
            seasonal_coords[:, 0],
            seasonal_coords[:, 1],
            c=seasonal_dot_colors,
            edgecolor="white",
            linewidth=1.5,
            s=280,
            zorder=4,
            label="Seasonal words",
        )

        label_offset_seasonal = 0.15
        for (x, y), label, _vr in zip(seasonal_coords, seasonal_present, seasonal_var_ratios):
            dist = np.sqrt(x**2 + y**2)
            if dist > 1e-9:
                dx, dy = (x / dist) * label_offset_seasonal, (y / dist) * label_offset_seasonal
            else:
                dx, dy = label_offset_seasonal, 0
            ax.text(
                x + dx,
                y + dy,
                label,
                fontsize=36,
                color=_seasonal_word_color(label),
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
                alpha=0.9,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(0.25)

    plt.savefig(output_2d, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"\nSaved → {output_2d}")


_SEM_CATS = {k: list(v) for k, v in _SEMANTIC_CATEGORIES.items()}
if "Other" not in _SEM_CATS:
    _SEM_CATS["Other"] = []

_CAT_ORDER = ["Other"] + [c for c in list(_SEM_CATS.keys())[::-1] if c != "Other"]
_UMAP_KW = dict(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42)
_POINT_SIZE = 14
_FIGSIZE = (6, 6)
_AX_PAD_FRAC = 0.08
_LEGEND_FONTSIZE = 10
_MAX_OTHER = 6000


DEFAULT_UMAP_MODEL_PATHS = [
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L200_mse_wd0.0_seed1.pt"),
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L400_mse_wd0.0_seed1.pt"),
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L600_mse_wd0.0_seed1.pt"),
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L800_mse_wd0.0_seed1.pt"),
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L800_mse_wd1.0_seed1.pt"),
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L800_mse_wd4.0_seed1.pt"),
    str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L1000_mse_wd1.0_seed1.pt"),
]


def _build_umap_palette(categories: List[str]) -> Dict[str, tuple]:
    grey = (0.7, 0.7, 0.7, 1.0)
    tab20 = plt.get_cmap("tab20")
    dark2 = plt.get_cmap("Dark2")
    vivid = [tab20(i) for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] + [dark2(i) for i in range(2)]
    palette: Dict[str, tuple] = {}
    ci = 0
    for cat in categories:
        if cat.lower() == "other":
            palette[cat] = grey
        else:
            palette[cat] = vivid[ci % len(vivid)]
            ci += 1
    palette.setdefault("Other", grey)
    return palette


_CAT_COLORS: Optional[Dict[str, tuple]] = None


def _get_cat_colors() -> Dict[str, tuple]:
    global _CAT_COLORS
    if _CAT_COLORS is None:
        _CAT_COLORS = _build_umap_palette(_CAT_ORDER)
    return _CAT_COLORS


def _build_token_lists(w2i: Dict[str, int]) -> Tuple[np.ndarray, List[str], List[str]]:
    rng = random.Random(42)
    seen = set()
    idxs: List[int] = []
    words: List[str] = []
    cats: List[str] = []

    lower2idx = {}
    for word, idx in w2i.items():
        lower2idx.setdefault(word.lower(), idx)

    for cat in [c for c in _CAT_ORDER if c != "Other"]:
        for tok in _SEM_CATS.get(cat, []):
            idx = lower2idx.get(tok.lower())
            if idx is not None and idx not in seen:
                seen.add(idx)
                idxs.append(idx)
                words.append(tok)
                cats.append(cat)

    all_cat_lowers = {t.lower() for vals in _SEM_CATS.values() for t in vals}
    candidates = [(idx, word) for word, idx in w2i.items() if idx not in seen and word.lower() not in all_cat_lowers]
    rng.shuffle(candidates)
    for idx, word in candidates[:_MAX_OTHER]:
        idxs.append(idx)
        words.append(word)
        cats.append("Other")

    return np.array(idxs, dtype=np.int64), words, cats


def _plot_umap_for_model(model_path: str, w2i: Dict[str, int], present_cats: Set[str], out_dir: Path) -> None:
    import umap.umap_ as umap

    W = load_encoder_weight_matrix(model_path)
    idxs, _words, cats = _build_token_lists(w2i)
    X = W[idxs, :]
    emb = umap.UMAP(**_UMAP_KW).fit_transform(X)
    present_cats.update(cats)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    cat_colors = _get_cat_colors()
    for cat in ["Other"] + [c for c in _CAT_ORDER if c != "Other"]:
        pts = [i for i, c in enumerate(cats) if c == cat]
        if not pts:
            continue
        ax.scatter(
            emb[pts, 0],
            emb[pts, 1],
            s=_POINT_SIZE,
            color=cat_colors[cat],
            rasterized=True,
            alpha=0.4 if cat == "Other" else 0.8,
            linewidths=0,
        )

    xmn, xmx = emb[:, 0].min(), emb[:, 0].max()
    ymn, ymx = emb[:, 1].min(), emb[:, 1].max()
    cx, cy = (xmn + xmx) / 2, (ymn + ymx) / 2
    hs = 0.5 * max(xmx - xmn, ymx - ymn) * (1 + _AX_PAD_FRAC)
    ax.set_xlim(cx - hs, cx + hs)
    ax.set_ylim(cy - hs, cy + hs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    stem = Path(model_path).stem
    for suffix in (".pdf", ".png"):
        fig.savefig(out_dir / f"umap_semcats__{stem}{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_dir / f'umap_semcats__{stem}.pdf'}")


def _save_umap_legend(present_cats: List[str], out_dir: Path) -> None:
    cat_colors = _get_cat_colors()
    order = [c for c in _CAT_ORDER if c in present_cats]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(2.5, max(2.0, 0.28 * len(order))), dpi=200)
    ax.axis("off")
    patches = [mpatches.Patch(color=cat_colors[c], label=c) for c in order]
    fig.legend(handles=patches, loc="center", frameon=False, fontsize=_LEGEND_FONTSIZE)
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_path = out_dir / "legend_semantic_categories.pdf"
    fig.savefig(legend_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved legend: {legend_path}")


R2_DISPLAY_WORDS = [
    ("Christmas", "seasonal", "Christmas"),
    ("mbar", "atmospheric", "mbar"),
    ("inHg", "atmospheric", "inHg"),
    ("shear", "atmospheric", "shear"),
    ("convection", "atmospheric", "convection"),
    ("Rumble", "wrestling", "Rumble"),
    ("McMahon", "wrestling", "McMahon"),
    ("Cena", "wrestling", "Cena"),
]

THREE_TOWER_CONTEXTS: Dict[str, Tuple[List[str], List[str], List[str]]] = {
    "Beatles": (
        ["Lennon", "McCartney", "Harrison", "Starr"],
        ["orange", "milk", "rice", "fish"],
        ["Lennon", "McCartney", "Harrison", "Starr"],
    ),
    "Christmas": (
        ["December", "gift", "Day", "snow"],
        ["orange", "milk", "rice", "fish"],
        ["December", "gift", "Day", "snow"],
    ),
    "mbar": (
        ["pressure", "barometric", "inHg", "hPa"],
        ["orange", "milk", "rice", "fish"],
        ["pressure", "barometric", "inHg", "hPa"],
    ),
}

R2_EXTRA_WORDS: Dict[str, List[Tuple[str, str, str]]] = {
    "Beatles": [
        ("Beatles", "music", "Beatles"),
        ("Lennon", "music", "Lennon"),
        ("McCartney", "music", "McCartney"),
        ("Harrison", "music", "Harrison"),
        ("Starr", "music", "Starr"),
    ],
    "Christmas": [
        ("Christmas", "seasonal", "Christmas"),
        ("December", "seasonal", "December"),
        ("January", "seasonal", "January"),
        ("snow", "seasonal", "snow"),
        ("Halloween", "seasonal", "Halloween"),
    ],
    "mbar": [
        ("mbar", "atmospheric", "mbar"),
        ("pressure", "atmospheric", "pressure"),
        ("barometric", "atmospheric", "barometric"),
        ("inHg", "atmospheric", "inHg"),
        ("hPa", "atmospheric", "hPa"),
    ],
}

CATEGORY_COLOR = {
    "seasonal": "#2ca02c",
    "atmospheric": "#1f77b4",
    "wrestling": "#ff7f0e",
    "music": "#1f77b4",
}

WORD_COLOR_OVERRIDES: Dict[str, str] = {
    "Lennon": "#27a040",
    "McCartney": "#3dba58",
    "Harrison": "#62d073",
    "Starr": "#8de09b",
}

FS_TICK = 15
FS_LABEL = 18
FS_TITLE = 15
FS_ANN = 16
FS_NUM = 12
FS_LEGEND = 13
FS_PANEL = 16


def load_weights_and_bias(ckpt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    enc_w = state["enc.weight"].detach().cpu().float().numpy().astype(np.float64)
    bias = state["bias"].detach().cpu().float().numpy().astype(np.float64)
    return enc_w, bias


def lookup_ci(w2i: Dict[str, int], word: str) -> Optional[int]:
    if word in w2i:
        return w2i[word]
    word_lower = word.lower()
    return next((v for k, v in w2i.items() if k.lower() == word_lower), None)


def load_r2_lookup(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    result: Dict[str, Tuple[float, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "")
            try:
                r_val = float(row.get("r2_valtest", row.get("r2_validation", "nan")))
                r_onehot = float(row.get("r2_onehot", row.get("r2_single_word", "nan")))
                result[word] = (r_val, r_onehot)
            except (ValueError, KeyError):
                continue
    return result


def pre_activation(enc_w: np.ndarray, bias: np.ndarray, target_idx: int, context: List[int]) -> float:
    W_v = enc_w[:, target_idx]
    lat = enc_w[:, context].sum(axis=1) if context else np.zeros(enc_w.shape[0])
    return float(W_v @ lat) + float(bias[target_idx])


def pre_acts_for_windows_containing(
    enc_w: np.ndarray,
    bias: np.ndarray,
    target_idx: int,
    npz_paths: List[Path],
    max_windows: int = 50_000,
) -> Tuple[np.ndarray, List[List[int]]]:
    W_v = enc_w[:, target_idx]
    b_v = float(bias[target_idx])
    pre_acts, windows_found = [], []
    n_checked = 0
    for npz_path in npz_paths:
        for window in iter_windows_npz(npz_path):
            if n_checked >= max_windows:
                break
            n_checked += 1
            if target_idx not in window:
                continue
            lat = enc_w[:, window].sum(axis=1)
            pre_acts.append(float(W_v @ lat) + b_v)
            windows_found.append(window)
    return np.array(pre_acts), windows_found


def contribution_decomposition(
    enc_w: np.ndarray,
    bias: np.ndarray,
    target_idx: int,
    context_idxs: List[int],
    idx2word: Dict[int, str],
    top_k_context: int = 6,
) -> Tuple[List[str], List[float]]:
    W_v = enc_w[:, target_idx]
    b_v = float(bias[target_idx])
    self_c = float(W_v @ W_v)

    other = [(float(W_v @ enc_w[:, w]), idx2word.get(w, f"<{w}>")) for w in context_idxs if w != target_idx]
    other.sort(key=lambda x: -x[0])
    top = other[:top_k_context]

    labels = ["bias", "self"] + [lbl for _, lbl in top]
    values = [b_v, self_c] + [val for val, _ in top]
    return labels, values


def _bar_color(val: float) -> str:
    return "#2ca02c" if val >= 0 else "#d62728"


def _label_text_color(bg_hex: str) -> str:
    r, g, b = int(bg_hex[1:3], 16) / 255, int(bg_hex[3:5], 16) / 255, int(bg_hex[5:7], 16) / 255
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.40 else "#333333"


def panel_r2_bars(
    ax: plt.Axes,
    words: List[str],
    labels: List[str],
    r2_val: List[float],
    r2_onehot: List[float],
    categories: List[str],
    target_word: Optional[str] = None,
    word_color_map: Optional[Dict[str, str]] = None,
) -> None:
    h = 0.32
    y = np.arange(len(words))
    near_zero_thresh = 0.02

    for i, (rv, rs, cat, lbl) in enumerate(zip(r2_val, r2_onehot, categories, labels)):
        col = (word_color_map or {}).get(lbl) or CATEGORY_COLOR.get(cat, "#7f7f7f")
        is_target = target_word is not None and lbl == target_word
        ec = "black" if is_target else "none"
        elw = 1.5 if is_target else 0

        ax.barh(y[i] + h / 2, rv, h, color=col, alpha=0.85, edgecolor=ec, linewidth=elw)
        if rs >= near_zero_thresh:
            ax.barh(y[i] - h / 2, rs, h, color=col, alpha=0.85, hatch="///", edgecolor="white", linewidth=0.5)
            if is_target:
                ax.barh(y[i] - h / 2, rs, h, fill=False, edgecolor="black", linewidth=elw)
        else:
            ax.plot(0, y[i] - h / 2, "D", color=col, ms=5, alpha=0.75, zorder=4)
            ax.text(0.035, y[i] - h / 2, "0", va="center", ha="left", fontsize=FS_ANN - 1, color=col, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    if target_word:
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == target_word:
                tick_label.set_fontweight("bold")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.7, len(words) - 0.5)
    ax.set_xlabel("Per-word R²", fontsize=FS_LABEL)
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", ls=":", alpha=0.35)

    handles = []
    n_cats = len({c for c in categories})
    if n_cats > 1:
        for cat, col in CATEGORY_COLOR.items():
            if cat in categories:
                handles.append(mpatches.Patch(color=col, alpha=0.85, label=cat))
    solid = mpatches.Patch(facecolor="gray", alpha=0.85, label="val R²")
    hatched = mpatches.Patch(facecolor="gray", alpha=0.85, hatch="///", edgecolor="white", linewidth=0.5, label="one-hot R²")
    ax.legend(handles=handles + [solid, hatched], fontsize=FS_LEGEND, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, framealpha=0.9)


def panel_preact_histogram(ax: plt.Axes, pre_natural: np.ndarray, pre_onehot_val: float, target_word: str) -> None:
    bw = 0.10
    kde = gaussian_kde(pre_natural, bw_method=bw)
    pad = 0.25
    lo = min(pre_natural.min() - pad, pre_onehot_val - pad)
    hi = pre_natural.max() + pad
    xs = np.linspace(lo, hi, 600)
    ys = kde(xs)
    ymax = ys.max()

    pos_frac = (pre_natural > 0).mean()

    i0 = int(np.searchsorted(xs, 0.0))
    y0 = float(kde(0.0))
    xs_neg = np.append(xs[:i0], 0.0)
    ys_neg = np.append(ys[:i0], y0)
    xs_pos = np.concatenate([[0.0], xs[i0:]])
    ys_pos = np.concatenate([[y0], ys[i0:]])

    ax.fill_betweenx([0, ymax * 1.15], lo, 0, color="#d62728", alpha=0.07, zorder=0)
    ax.fill_betweenx([0, ymax * 1.15], 0, hi, color="#2ca02c", alpha=0.07, zorder=0)

    zero_pct = f" ({1 - pos_frac:.0%})" if pos_frac < 1.0 else ""
    label_zero = f"$\\hat{{x}}_{{{target_word}}} = 0${zero_pct}"
    ax.fill_between(xs_neg, ys_neg, alpha=0.30, color="#d62728", hatch="///", label=label_zero, zorder=2)
    ax.plot(xs_neg, ys_neg, color="#d62728", lw=1.8, ls="--", zorder=3)

    label_pos = f"$\\hat{{x}}_{{{target_word}}} > 0$ ({pos_frac:.0%})" if pos_frac < 1.0 else f"$\\hat{{x}}_{{{target_word}}} > 0$"
    ax.fill_between(xs_pos, ys_pos, alpha=0.40, color="#1f77b4", label=label_pos, zorder=2)
    ax.plot(xs_pos, ys_pos, color="#1f77b4", lw=1.8, zorder=3)

    ax.axvline(pre_onehot_val, color="#ff7f0e", lw=2.2, ls="-")
    ax.axvline(0, color="black", lw=1.4, ls="--", zorder=5)

    pos_mask = xs > 0
    peak_x = float(xs[pos_mask][np.argmax(ys[pos_mask])])
    arrow_y = ymax * 0.72
    arrow_color = "#2ca02c"
    ax.annotate(
        "",
        xy=(peak_x, arrow_y),
        xytext=(pre_onehot_val, arrow_y),
        arrowprops=dict(arrowstyle="<->", color=arrow_color, lw=1.8),
        annotation_clip=False,
    )
    ax.text(
        (peak_x + pre_onehot_val) / 2,
        arrow_y + ymax * 0.05,
        "Interference\nis beneficial",
        ha="center",
        va="bottom",
        color=arrow_color,
        fontsize=FS_ANN - 2,
        fontweight="bold",
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel(f"'{target_word}' decoder pre-activation", fontsize=FS_LABEL)
    ax.set_ylabel("Density", fontsize=FS_LABEL)

    fill_handles, _fill_labels = ax.get_legend_handles_labels()
    line_handles = [
        Line2D([0], [0], color="#ff7f0e", lw=2.2, ls="-", label=f"one-hot ({pre_onehot_val:+.3f})"),
        Line2D([0], [0], color="black", lw=1.4, ls="--", label="ReLU threshold"),
    ]
    ax.legend(handles=fill_handles + line_handles, fontsize=FS_LEGEND, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9)
    ax.grid(ls=":", alpha=0.35)


def panel_waterfall(ax: plt.Axes, labels: List[str], values: List[float], target_word: str, context_desc: str) -> None:
    running = np.cumsum([0.0] + values[:-1])
    cum = np.cumsum(values)

    colors = [_bar_color(v) for v in values]
    colors[0] = "#d62728"
    colors[1] = "#4393c3"

    n = len(labels)
    y = np.arange(n - 1, -1, -1)
    h = 0.52

    ax.axvspan(-999, 0, color="#d62728", alpha=0.07, zorder=0)
    ax.axvspan(0, 999, color="#2ca02c", alpha=0.07, zorder=0)

    ax.barh(y, values, left=running, height=h, color=colors, alpha=0.85, edgecolor="white", lw=0.6)

    for val, run, yi in zip(values, running, y):
        cx = run + val / 2
        txt_col = "white" if abs(val) > 0.025 else "black"
        ax.text(cx, yi, f"{val:+.3f}", va="center", ha="center", fontsize=FS_NUM, color=txt_col, fontweight="bold")

    xs_end = [running[i] + values[i] for i in range(n)]
    for yi_dot, end, c in zip(y, xs_end, cum):
        col = "#2ca02c" if c >= 0 else "#d62728"
        ax.plot(end, yi_dot, "D", color=col, ms=4.5, zorder=5)
    ax.plot(xs_end, y, color="gray", lw=0.8, ls="-", alpha=0.5, zorder=4)

    ax.axvline(0, color="black", lw=1.5, ls="--", zorder=6)
    ax.text(0.01, 0.03, "ReLU\nthreshold", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=FS_ANN, color="black")

    onehot_total = values[0] + values[1]
    ax.axvline(onehot_total, color="#ff7f0e", lw=1.5, ls=":", zorder=6)
    ax.text(onehot_total - 0.01, 0.03, f"one-hot\n({onehot_total:+.3f})", transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=FS_ANN, color="#ff7f0e")

    first_cross = next((i for i, c in enumerate(cum) if c >= 0), None)
    if first_cross is not None:
        cross_y = y[first_cross]
        ax.annotate(
            "threshold\ncrossed ↑",
            xy=(0.005, cross_y + 0.25),
            xytext=(0.08, cross_y + 1.0),
            fontsize=FS_ANN,
            color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0),
        )

    xs_all = list(running) + xs_end
    pad_x = (max(xs_all) - min(xs_all)) * 0.15
    ax.set_xlim(min(xs_all) - pad_x, max(xs_all) + pad_x)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("Cumulative pre-activation", fontsize=FS_LABEL)
    ax.grid(axis="x", ls=":", alpha=0.35)


def panel_three_towers(
    ax: plt.Axes,
    labels1: List[str],
    values1: List[float],
    labels2: List[str],
    values2: List[float],
    labels3: List[str],
    values3: List[float],
    target_word: str,
    bias_val: float,
    ctx1_desc: str = "",
    ctx2_desc: str = "",
    ctx3_desc: str = "",
) -> None:
    BAR_W = 0.60
    X1, X2, X3 = 0.0, 1.0, 2.0

    def draw_tower(x: float, labels: List[str], values: List[float]):
        running = bias_val + np.cumsum([0.0] + list(values[:-1]))
        colors = [_bar_color(v) for v in values]
        if labels[0] == target_word:
            colors[0] = "#4393c3"
        for i, lbl in enumerate(labels):
            if lbl in WORD_COLOR_OVERRIDES:
                colors[i] = WORD_COLOR_OVERRIDES[lbl]
        for val, run, col, lbl in zip(values, running, colors, labels):
            ax.bar(x, val, bottom=run, width=BAR_W, color=col, alpha=0.85, edgecolor="white", lw=0.5, zorder=3)
            cy = run + val / 2
            if abs(val) >= 0.035:
                txt_col = _label_text_color(col)
                ax.text(x, cy, lbl, va="center", ha="center", fontsize=FS_ANN - 3, color=txt_col, fontweight="bold", zorder=4)
        return running, bias_val + np.cumsum(np.array(values))

    _running1, cum1 = draw_tower(X1, labels1, values1)
    _running2, cum2 = draw_tower(X2, labels2, values2)
    _running3, cum3 = draw_tower(X3, labels3, values3)

    ctx_words = labels2[1:]
    ctx_vals = values2[1:]
    ctx_sum = sum(ctx_vals)
    ann_color = "#d62728"
    tower2_top = cum2[-1]
    box_dy = 0.075
    box_hw = 0.215
    box_hh = 0.033
    y_sum = tower2_top + 0.09
    gap_to_fish = 0.105

    for j, word in enumerate(reversed(ctx_words)):
        y_pos = y_sum + gap_to_fish + j * box_dy
        rect = mpatches.FancyBboxPatch(
            (X2 - box_hw, y_pos - box_hh),
            2 * box_hw,
            2 * box_hh,
            boxstyle="square,pad=0",
            edgecolor=ann_color,
            facecolor="none",
            linestyle=":",
            linewidth=1.2,
            zorder=6,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(X2, y_pos, word, fontsize=FS_NUM + 2, color=ann_color, va="center", ha="center", zorder=7)

    ax.text(X2, y_sum, f"$\\Sigma$ = {ctx_sum:+.3f}", fontsize=FS_NUM + 2, color=ann_color, va="bottom", ha="center", zorder=6)

    ax.annotate(
        "",
        xy=(X2, tower2_top + 0.005),
        xytext=(X2, y_sum - 0.01),
        arrowprops=dict(arrowstyle="->", color=ann_color, lw=1.2),
        zorder=6,
    )

    ax.axhspan(-999, 0, color="#d62728", alpha=0.07, zorder=0)
    ax.axhspan(0, 999, color="#2ca02c", alpha=0.07, zorder=0)

    ax.axhline(0, color="black", lw=1.5, ls="--", zorder=1)
    ax.axhline(bias_val, color="#ff7f0e", lw=2.5, ls=":", zorder=5)

    legend_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="ReLU threshold"),
        Line2D([0], [0], color="#ff7f0e", lw=2.5, ls=":", label=f"bias = {bias_val:+.3f}"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=FS_LEGEND, framealpha=0.9)

    all_y = list(cum1) + list(cum2) + list(cum3) + [bias_val, 0.0]
    ax.set_ylim(min(all_y) - 0.05, max(all_y) + 0.25)
    ax.set_xlim(X1 - BAR_W / 2 - 0.15, X3 + BAR_W / 2 + 0.15)

    ax.set_xticks([X1, X2, X3])
    ax.set_xticklabels([ctx1_desc, ctx2_desc, ctx3_desc], fontsize=FS_TICK - 1)
    ax.set_ylabel("Pre-activation", fontsize=FS_LABEL)
    ax.grid(axis="y", ls=":", alpha=0.35)


def panel_compact_two_towers(
    ax: plt.Axes,
    labels1: List[str],
    values1: List[float],
    labels2: List[str],
    values2: List[float],
    target_word: str,
    bias_val: float,
    ctx1_desc: str = "",
    ctx2_desc: str = "",
) -> None:
    bar_w = 0.42
    bias_lw = 3.2
    x1, x2 = 0.0, 0.58
    label_thresh = 0.012
    month_label_whitelist = {"December", "January", "July"}

    def draw_tower(
        x: float,
        labels: List[str],
        values: List[float],
        highlight_target: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
        running = bias_val + np.cumsum([0.0] + list(values[:-1]))
        colors = [_bar_color(v) for v in values]
        centers: Dict[str, Tuple[float, float]] = {}
        if highlight_target and labels and labels[0] == target_word:
            colors[0] = "#4393c3"
        for i, lbl in enumerate(labels):
            if lbl in WORD_COLOR_OVERRIDES:
                colors[i] = WORD_COLOR_OVERRIDES[lbl]
        for val, run, col, lbl in zip(values, running, colors, labels):
            ax.bar(x, val, bottom=run, width=bar_w, color=col, alpha=0.88, edgecolor="white", lw=0.5, zorder=3)
            if abs(val) >= label_thresh:
                cy = run + val / 2
                centers[lbl] = (x, cy)
                txt_col = _label_text_color(col)
                ax.text(x, cy, lbl, va="center", ha="center", fontsize=FS_ANN - 3, color=txt_col, fontweight="bold", zorder=4)
            else:
                centers[lbl] = (x, run + val / 2)
        return running, bias_val + np.cumsum(np.array(values)), centers

    def draw_month_tower_with_sum(
        x: float,
        labels: List[str],
        values: List[float],
    ) -> Tuple[List[float], List[float], float, float, float, Dict[str, Tuple[float, float]]]:
        pairs = list(zip(labels, values))
        pos_pairs = [(lbl, val) for lbl, val in pairs if val > 0]
        neg_pairs = [(lbl, val) for lbl, val in pairs if val < 0]
        # Put the strongest positive month at the top of the positive stack.
        pos_pairs = list(reversed(pos_pairs))

        tower_w = 0.40
        sum_w = 0.28
        gap_w = 0.08
        x_tower = x
        x_sum = x + (tower_w / 2 + gap_w + sum_w / 2)

        neg_base = bias_val
        pos_base = bias_val
        pos_points = [pos_base]
        neg_points = [bias_val]
        centers: Dict[str, Tuple[float, float]] = {}

        for lbl, val in neg_pairs:
            col = _bar_color(val)
            if lbl in WORD_COLOR_OVERRIDES:
                col = WORD_COLOR_OVERRIDES[lbl]
            bottom = neg_base
            ax.bar(x_tower, val, bottom=bottom, width=tower_w, color=col, alpha=0.88, edgecolor="white", lw=0.5, zorder=3)
            centers[lbl] = (x_tower, bottom + val / 2)
            if lbl in month_label_whitelist:
                cy = centers[lbl][1]
                txt_col = _label_text_color(col)
                ax.text(x_tower, cy, lbl, va="center", ha="center", fontsize=FS_ANN - 2, color=txt_col, fontweight="bold", zorder=4)
            neg_base += val
            neg_points.append(neg_base)

        for lbl, val in pos_pairs:
            col = _bar_color(val)
            if lbl in WORD_COLOR_OVERRIDES:
                col = WORD_COLOR_OVERRIDES[lbl]
            bottom = pos_base
            ax.bar(x_tower, val, bottom=bottom, width=tower_w, color=col, alpha=0.88, edgecolor="white", lw=0.5, zorder=3)
            centers[lbl] = (x_tower, bottom + val / 2)
            if lbl in month_label_whitelist:
                cy = centers[lbl][1]
                txt_col = _label_text_color(col)
                ax.text(x_tower, cy, lbl, va="center", ha="center", fontsize=FS_ANN - 2, color=txt_col, fontweight="bold", zorder=4)
            pos_base += val
            pos_points.append(pos_base)

        month_sum = float(np.sum(values))
        sum_color = "#2ca02c"
        ax.bar(
            x_sum,
            month_sum,
            bottom=bias_val,
            width=sum_w,
            color=sum_color,
            alpha=0.95,
            edgecolor="none",
            lw=0.0,
            zorder=4,
        )
        with matplotlib.rc_context({"hatch.linewidth": 2.8}):
            ax.bar(
                x_sum,
                month_sum,
                bottom=bias_val,
                width=sum_w,
                color="none",
                alpha=1.0,
                edgecolor="#d62728",
                hatch="///",
                lw=0.0,
                zorder=5,
            )

        return pos_points, neg_points, x_tower, x_sum, tower_w, sum_w, centers

    running1, cum1, left_centers = draw_tower(x1, labels1, values1, highlight_target=True)
    pos2, neg2, x_tower, x_sum, tower_w, sum_w, month_centers = draw_month_tower_with_sum(x2, labels2, values2)

    ax.axhspan(-999, 0, color="#d62728", alpha=0.07, zorder=0)
    ax.axhspan(0, 999, color="#2ca02c", alpha=0.07, zorder=0)
    ax.axhline(0, color="black", lw=1.5, ls="--", zorder=1)
    ax.axhline(bias_val, color="#ff7f0e", lw=bias_lw, ls=":", zorder=5)

    month_sum = float(np.sum(values2))
    top2 = float(max(pos2)) if len(pos2) > 0 else bias_val
    neg_bottom = float(min(neg2)) if len(neg2) > 0 else bias_val
    brace_gap = x_sum - x_tower - (tower_w / 2 + sum_w / 2)
    brace_x = x_tower + tower_w / 2 + brace_gap / 2
    brace_y = 0.5 * (top2 + neg_bottom)
    arrow_dx = tower_w * 0.62
    arrow_gap = 0.012
    red_arrow_len = max(0.05, min(0.085, bias_val - neg_bottom - 0.008))
    green_arrow_len = max(0.05, min(0.085, top2 - bias_val - 0.008))
    ax.annotate(
        "",
        xy=(x_tower - arrow_dx, bias_val - arrow_gap - red_arrow_len),
        xytext=(x_tower - arrow_dx, bias_val - arrow_gap),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#d62728",
            lw=3.0,
            mutation_scale=18,
            capstyle="butt",
            joinstyle="miter",
        ),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(x_tower - arrow_dx, bias_val + arrow_gap + green_arrow_len),
        xytext=(x_tower - arrow_dx, bias_val + arrow_gap),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#2ca02c",
            lw=3.0,
            mutation_scale=18,
            capstyle="butt",
            joinstyle="miter",
        ),
        zorder=6,
    )
    sum_top = bias_val + month_sum
    sum_mid = bias_val + month_sum / 2.0
    ax.text(
        brace_x,
        brace_y,
        "}",
        color="#000000",
        fontsize=38,
        va="center",
        ha="center",
        zorder=6,
    )
    ax.text(
        x_sum + 0.03,
        bias_val - 0.016,
        f"$\\mathbf{{\\Sigma}}$ = {month_sum:.3f}",
        color="#2ca02c",
        fontsize=FS_NUM + 2,
        va="top",
        ha="center",
        zorder=6,
        fontweight="bold",
    )
    ax.text(
        x_sum - 0.035,
        -0.05,
        "Interference\ncancels out",
        color="#d62728",
        fontsize=FS_NUM + 2,
        va="top",
        ha="center",
        zorder=6,
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(x_sum, sum_top),
        xytext=(x_sum, -0.155),
        arrowprops=dict(
            arrowstyle="->",
            color="#d62728",
            linestyle="dotted",
            lw=1.5,
            connectionstyle="arc3,rad=0",
            shrinkA=0,
            shrinkB=0,
        ),
    )
    if "December" in left_centers:
        dec_x, dec_y = left_centers["December"]
        ax.annotate(
            "Constructive\ninterference",
            xy=(dec_x + bar_w / 2, dec_y),
            xytext=(x2, 0.11),
            color="#2ca02c",
            fontsize=FS_NUM + 2,
            fontweight="bold",
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", linestyle="dotted", lw=1.5),
            zorder=7,
        )

    legend_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="ReLU\nthreshold"),
        Line2D([0], [0], color="#ff7f0e", lw=bias_lw, ls=":", label=f"bias = {bias_val:+.3f}"),
        mpatches.Patch(color="#2ca02c", alpha=0.88, label="positive\ninterference"),
        mpatches.Patch(color="#d62728", alpha=0.88, label="negative\ninterference"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=FS_LEGEND - 1, framealpha=0.9)

    all_y = [bias_val, 0.0] + list(running1) + list(cum1) + list(pos2) + list(neg2)
    right_xtick = 0.5 * (x_tower + x_sum)
    ax.set_ylim(min(all_y) - 0.05, 0.62)
    ax.set_xlim(x1 - bar_w / 2 - 0.08, x_sum + sum_w / 2 + 0.08)
    ax.set_xticks([x1, right_xtick])
    ax.set_xticklabels([ctx1_desc, ctx2_desc], fontsize=FS_TICK)
    ax.set_ylabel("Pre-activation", fontsize=FS_LABEL, labelpad=-2)
    ax.grid(axis="y", ls=":", alpha=0.35)


def month_interference_contributions(
    enc_w: np.ndarray,
    w2i: Dict[str, int],
    target_idx: int,
) -> Tuple[List[str], List[float]]:
    W_target = enc_w[:, target_idx]
    rows: List[Tuple[str, float]] = []
    for month in MONTH_NAMES:
        month_idx = lookup_ci(w2i, month)
        if month_idx is None or month_idx == target_idx:
            continue
        rows.append((month, float(W_target @ enc_w[:, month_idx])))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return [label for label, _ in rows], [value for _, value in rows]


def generate_mechanism_figure(
    ls: int,
    checkpoint_template: str,
    vocab_path: str,
    data_dir: str,
    r2_csv: str,
    out_dir: str,
    target: str,
    top_k_contrib: int,
    max_windows: int,
) -> None:
    ckpt_path = Path(checkpoint_template.format(ls=ls))
    vocab_path_p = Path(vocab_path)
    r2_csv_p = Path(r2_csv.format(ls=ls))

    data_dir_p = Path(data_dir)
    candidate_npzs = [
        data_dir_p / "wikitext_validation_v10000_w20_s1.npz",
        data_dir_p / "wikitext_val_v10000_w20_s1.npz",
        data_dir_p / "wikitext_test_v10000_w20_s1.npz",
    ]
    npz_paths = [p for p in candidate_npzs if p.exists()]
    if not npz_paths:
        raise FileNotFoundError(f"Could not find any val/test npz in {data_dir_p}")
    print(f"Using splits: {[p.name for p in npz_paths]}")

    print(f"Loading checkpoint: {ckpt_path}")
    enc_w, bias = load_weights_and_bias(ckpt_path)
    print(f"  enc.weight: {enc_w.shape}  bias: {bias.shape}")

    w2i = load_vocab_map(str(vocab_path_p))
    idx2word = {v: k for k, v in w2i.items()}

    target_idx = lookup_ci(w2i, target)
    if target_idx is None:
        raise KeyError(f"Target word not in vocabulary: {target!r}")
    print(f"Target: '{target}'  idx={target_idx}  ||enc||={np.linalg.norm(enc_w[:, target_idx]):.4f}")

    r2_lookup: Dict[str, Tuple[float, float]] = {}
    if r2_csv_p.exists():
        r2_lookup = load_r2_lookup(r2_csv_p)
        print(f"Loaded R² for {len(r2_lookup)} words from {r2_csv_p.name}")
    else:
        print(f"Warning: R² CSV not found at {r2_csv_p}. Panel A will be empty.")

    display_list = list(R2_DISPLAY_WORDS)
    if target in R2_EXTRA_WORDS:
        display_list = R2_EXTRA_WORDS[target]

    r2_words_data = []
    for word, cat, lbl in display_list:
        wi = lookup_ci(w2i, word)
        if wi is None:
            print(f"  skip '{word}': not in vocab")
            continue
        if word in r2_lookup:
            rv, rs = r2_lookup[word]
        elif word.lower() in r2_lookup:
            rv, rs = r2_lookup[word.lower()]
        else:
            print(f"  '{word}' not in R² CSV – skipping from panel A")
            continue
        r2_words_data.append((lbl, cat, rv, rs))
    print(f"Panel A words: {[x[0] for x in r2_words_data]}")

    print(f"\nComputing pre-activations from {[p.name for p in npz_paths]} ...")
    pre_natural, all_windows = pre_acts_for_windows_containing(enc_w, bias, target_idx, npz_paths, max_windows=max_windows)
    pre_onehot_val = pre_activation(enc_w, bias, target_idx, [target_idx])
    print(f"  natural windows containing target: {len(pre_natural)}")
    print(f"  one-hot pre-activation: {pre_onehot_val:.4f}")
    print(
        f"  natural pre-act: mean={pre_natural.mean():.4f}  median={np.median(pre_natural):.4f}  frac>0: {(pre_natural > 0).mean():.2%}"
    )

    W_target = enc_w[:, target_idx]
    b_v = float(bias[target_idx])
    self_c = float(W_target @ W_target)

    use_three_towers = target in THREE_TOWER_CONTEXTS

    if use_three_towers:
        sem_words, rnd_words, noctx_words = THREE_TOWER_CONTEXTS[target]

        def fixed_decomp(word_list: List[str], include_self: bool = True) -> Tuple[List[str], List[float]]:
            lbs: List[str] = [target] if include_self else []
            vals: List[float] = [self_c] if include_self else []
            for word in word_list:
                wi = lookup_ci(w2i, word)
                if wi is not None and wi != target_idx:
                    lbs.append(word)
                    vals.append(float(W_target @ enc_w[:, wi]))
                else:
                    print(f"  '{word}' not found in vocab – skipping")
            return lbs, vals

        decomp_labels1, decomp_values1 = fixed_decomp(sem_words)
        decomp_labels2, decomp_values2 = fixed_decomp(rnd_words)
        decomp_labels3, decomp_values3 = fixed_decomp(noctx_words, include_self=False)
        print(f"\nTower 1 (semantic):        {list(zip(decomp_labels1, [f'{v:.3f}' for v in decomp_values1]))}")
        print(f"Tower 2 (random):          {list(zip(decomp_labels2, [f'{v:.3f}' for v in decomp_values2]))}")
        print(f"Tower 3 (corr, no target): {list(zip(decomp_labels3, [f'{v:.3f}' for v in decomp_values3]))}")

        month_labels, month_values = month_interference_contributions(enc_w, w2i, target_idx)
        print(f"Month interference:        {list(zip(month_labels, [f'{v:.3f}' for v in month_values]))}")

        ctx1_desc = f"'{target}' +\ncorrelated\ncontext"
        ctx2_desc = f"'{target}' +\nuncorrelated\ncontext"
        ctx3_desc = f"correlated context\n(without '{target}')"
    else:
        top_ctx_idxs = np.argsort(-(enc_w.T @ W_target))
        top_ctx_idxs = [i for i in top_ctx_idxs if i != target_idx][: top_k_contrib * 4]
        top_ctx_set = set(int(i) for i in top_ctx_idxs)

        target_median = float(np.median(pre_natural[pre_natural > 0]) if (pre_natural > 0).any() else np.median(pre_natural))
        scores = [
            (sum(1 for w in window if w in top_ctx_set), -abs(pa - target_median), i)
            for i, (pa, window) in enumerate(zip(pre_natural, all_windows))
        ]
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_window = all_windows[scores[0][2]]

        print(f"\nRepresentative window ({len(best_window)} tokens):")
        print(f"  words: {[idx2word.get(i, str(i)) for i in best_window[:20]]}")
        print(f"  pre-activation: {pre_activation(enc_w, bias, target_idx, best_window):.4f}")

        decomp_labels, decomp_values = contribution_decomposition(
            enc_w,
            bias,
            target_idx,
            best_window,
            idx2word,
            top_k_context=top_k_contrib,
        )
        print(f"  components: {list(zip(decomp_labels, [f'{v:.3f}' for v in decomp_values]))}")

        context_words_top = sorted(
            [idx2word.get(i, str(i)) for i in best_window if i != target_idx],
            key=lambda w: abs(float(enc_w[:, target_idx] @ enc_w[:, w2i.get(w, target_idx)])),
            reverse=True,
        )[:4]
        context_desc = "{" + ", ".join([target] + context_words_top) + ", ...}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), dpi=150, gridspec_kw={"width_ratios": [0.45, 1.3, 1.65]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, wspace=0.25)

    if r2_words_data:
        labels_A = [x[0] for x in r2_words_data]
        cats_A = [x[1] for x in r2_words_data]
        r2_val_A = [x[2] for x in r2_words_data]
        r2_onehot_A = [x[3] for x in r2_words_data]
        panel_r2_bars(axes[0], labels_A, labels_A, r2_val_A, r2_onehot_A, cats_A, target_word=target, word_color_map=WORD_COLOR_OVERRIDES)
    else:
        axes[0].text(0.5, 0.5, "R² CSV not found", transform=axes[0].transAxes, ha="center", va="center")

    panel_preact_histogram(axes[1], pre_natural, pre_onehot_val, target)

    if use_three_towers:
        panel_three_towers(
            axes[2],
            decomp_labels1,
            decomp_values1,
            decomp_labels2,
            decomp_values2,
            decomp_labels3,
            decomp_values3,
            target,
            b_v,
            ctx1_desc,
            ctx2_desc,
            ctx3_desc,
        )
    else:
        panel_waterfall(axes[2], decomp_labels, decomp_values, target, context_desc)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    stem = f"mechanism_figure_ls{ls}_{target}"
    out_png = out_dir_p / f"{stem}.png"
    out_pdf = out_dir_p / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    if use_three_towers and target == "Christmas":
        fig_compact, ax_compact = plt.subplots(1, 1, figsize=(5.8, 6.3), dpi=150)
        panel_compact_two_towers(
            ax_compact,
            decomp_labels1,
            decomp_values1,
            month_labels,
            month_values,
            target,
            b_v,
            f"'{target}' +\ncorrelated context",
            f"all months\nwithout '{target}'",
        )
        fig_compact.subplots_adjust(left=0.16, right=0.96, top=0.84, bottom=0.14)
        compact_stem = f"mechanism_month_towers_ls{ls}_{target}"
        compact_png = out_dir_p / f"{compact_stem}.png"
        compact_pdf = out_dir_p / f"{compact_stem}.pdf"
        fig_compact.savefig(compact_png, bbox_inches="tight")
        fig_compact.savefig(compact_pdf, bbox_inches="tight")
        plt.close(fig_compact)
        print(f"Saved: {compact_png}")
        print(f"Saved: {compact_pdf}")


def run_months(args: argparse.Namespace) -> Dict[str, dict]:
    experiments = _single_experiment_from_args(args)
    if not args.skip_seasonal:
        generate_seasonal_square(
            ls=args.ls,
            vocab_path=args.vocab_path,
            model_path_template=args.model_path_template,
            plots_root=args.plots_root,
        )

    figures_dir = Path(args.plots_root)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return _compute_month_analysis(
        experiments=experiments,
        base_data_dir=args.base_data_dir,
        max_samples_for_pca=args.max_samples,
        seed=args.seed,
        figures_dir=figures_dir,
        run_corr=True,
        run_data_pca=True,
    )


def run_weights(args: argparse.Namespace) -> None:
    experiments = _single_experiment_from_args(args)
    figures_dir = Path(args.plots_root)
    figures_dir.mkdir(parents=True, exist_ok=True)

    month_analysis = _compute_month_analysis(
        experiments=experiments,
        base_data_dir=args.base_data_dir,
        max_samples_for_pca=args.max_samples,
        seed=args.seed,
        figures_dir=figures_dir,
        run_corr=False,
        run_data_pca=False,
    )

    for label, info in month_analysis.items():
        ckpt_path = info["config"].get("ckpt_path")
        if not ckpt_path or not Path(ckpt_path).is_file():
            print(f"Warning: checkpoint not found for {label} → {ckpt_path}")
            continue
        plot_month_pca_from_weights(ckpt_path, info["month_idxs"], info["month_names"], label, figures_dir)
        print(f"Finished weight PCA for {label}.")


def _single_experiment_from_args(args: argparse.Namespace) -> List[dict]:
    dataset = getattr(args, "dataset", DEFAULT_TEXT_DATASET)
    vocab_size = int(getattr(args, "vocab_size", DEFAULT_TEXT_VOCAB_SIZE))
    group_size = int(getattr(args, "group_size", DEFAULT_TEXT_GROUP_SIZE))
    stride = int(getattr(args, "stride", DEFAULT_TEXT_STRIDE))
    label = str(getattr(args, "label", "") or dataset)
    vocab_path = str(getattr(args, "vocab_path", DEFAULT_SEASONAL_VOCAB_PATH))
    ckpt_template = str(getattr(args, "model_path_template", DEFAULT_SEASONAL_MODEL_TEMPLATE))
    ls = int(getattr(args, "ls", DEFAULT_SEASONAL_LS))
    return [
        {
            "label": label,
            "dataset": dataset,
            "vocab_size": vocab_size,
            "group_size": group_size,
            "stride": stride,
            "vocab_path": vocab_path,
            "ckpt_path": ckpt_template.format(ls=ls),
        }
    ]


def run_umap(args: argparse.Namespace) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    out_dir = Path(args.out_dir)
    vocab_path = Path(args.vocab_path)
    if not vocab_path.is_file():
        print(f"Warning: vocab not found → {vocab_path}")
        return

    w2i = load_vocab_map(str(vocab_path))
    present: Set[str] = set()
    for model_path in args.model_paths:
        if not Path(model_path).is_file():
            print(f"Warning: model not found → {model_path}")
            continue
        try:
            _plot_umap_for_model(model_path, w2i, present, out_dir)
        except Exception as exc:
            print(f"Error on {model_path}: {exc}")

    if present:
        _save_umap_legend(sorted(present, key=lambda c: (c != "Other", c)), out_dir)
    print(f"Done UMAP for {args.label}. Output: {out_dir.resolve()}")


def run_mechanism(args: argparse.Namespace) -> None:
    generate_mechanism_figure(
        ls=args.ls,
        checkpoint_template=args.checkpoint_template,
        vocab_path=args.vocab_path,
        data_dir=args.data_dir,
        r2_csv=args.r2_csv,
        out_dir=args.out_dir,
        target=args.target,
        top_k_contrib=args.top_k_contrib,
        max_windows=args.max_windows,
    )


def run_all(args: argparse.Namespace) -> None:
    month_args = argparse.Namespace(
        skip_seasonal=False,
        ls=args.ls,
        vocab_path=args.vocab_path,
        model_path_template=args.model_path_template,
        plots_root=args.plots_root,
        base_data_dir=args.base_data_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    month_analysis = run_months(month_args)

    figures_dir = Path(args.plots_root)
    for label, info in month_analysis.items():
        ckpt_path = info["config"].get("ckpt_path")
        if not ckpt_path or not Path(ckpt_path).is_file():
            print(f"Warning: checkpoint not found for {label} → {ckpt_path}")
            continue
        plot_month_pca_from_weights(ckpt_path, info["month_idxs"], info["month_names"], label, figures_dir)
        print(f"Finished weight PCA for {label}.")

    umap_args = argparse.Namespace(
        label="wikitext",
        vocab_path=args.vocab_path,
        model_paths=args.umap_model_paths,
        out_dir=str(figures_dir / "umap_semcats" / "wikitext"),
    )
    run_umap(umap_args)

    structure_args = argparse.Namespace(
        vocab_path=args.vocab_path,
        plots_root=args.plots_root,
        latent_sizes=list(STRUCTURE_LATENT_SIZES),
        model_path_template=DEFAULT_STRUCTURE_MODEL_TEMPLATE,
        multi_seed_templates=list(DEFAULT_STRUCTURE_MULTI_SEED_TEMPLATES),
        metrics_path=str(figures_dir / "metrics_curves3.pdf"),
        metrics_fill_path=str(figures_dir / "metrics_curves_fill.pdf"),
    )
    run_structure(structure_args)

    for target in args.mechanism_targets:
        print(f"\nGenerating mechanism figure → target='{target}' ...")
        mech_args = argparse.Namespace(
            ls=args.mechanism_ls,
            checkpoint_template=args.mechanism_checkpoint_template,
            vocab_path=args.vocab_path,
            data_dir=args.base_data_dir,
            r2_csv=args.mechanism_r2_csv,
            out_dir=args.plots_root,
            target=target,
            top_k_contrib=args.top_k_contrib,
            max_windows=args.max_windows,
        )
        run_mechanism(mech_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plotting utilities for text_bows experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p_months = sub.add_parser("months", help="Generate seasonal/month figures from one checkpoint + train split")
    p_months.add_argument("--ls", type=int, default=DEFAULT_SEASONAL_LS)
    p_months.add_argument("--dataset", default=DEFAULT_TEXT_DATASET)
    p_months.add_argument("--vocab_size", type=int, default=DEFAULT_TEXT_VOCAB_SIZE)
    p_months.add_argument("--group_size", type=int, default=DEFAULT_TEXT_GROUP_SIZE)
    p_months.add_argument("--stride", type=int, default=DEFAULT_TEXT_STRIDE)
    p_months.add_argument("--label", default="")
    p_months.add_argument("--vocab_path", default=DEFAULT_SEASONAL_VOCAB_PATH)
    p_months.add_argument("--model_path_template", default=DEFAULT_SEASONAL_MODEL_TEMPLATE)
    p_months.add_argument("--plots_root", default=DEFAULT_PLOTS_ROOT)
    p_months.add_argument("--base_data_dir", default=DEFAULT_DATA_DIR)
    p_months.add_argument("--max_samples", type=int, default=MAX_SAMPLES_FOR_PCA)
    p_months.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_months.add_argument("--skip_seasonal", action="store_true")
    p_months.set_defaults(func=run_months)

    p_weights = sub.add_parser("weights", help="Generate month PCA from encoder weights")
    p_weights.add_argument("--ls", type=int, default=DEFAULT_SEASONAL_LS)
    p_weights.add_argument("--dataset", default=DEFAULT_TEXT_DATASET)
    p_weights.add_argument("--vocab_size", type=int, default=DEFAULT_TEXT_VOCAB_SIZE)
    p_weights.add_argument("--group_size", type=int, default=DEFAULT_TEXT_GROUP_SIZE)
    p_weights.add_argument("--stride", type=int, default=DEFAULT_TEXT_STRIDE)
    p_weights.add_argument("--label", default="")
    p_weights.add_argument("--vocab_path", default=DEFAULT_SEASONAL_VOCAB_PATH)
    p_weights.add_argument("--model_path_template", default=DEFAULT_SEASONAL_MODEL_TEMPLATE)
    p_weights.add_argument("--plots_root", default=DEFAULT_PLOTS_ROOT)
    p_weights.add_argument("--base_data_dir", default=DEFAULT_DATA_DIR)
    p_weights.add_argument("--max_samples", type=int, default=MAX_SAMPLES_FOR_PCA)
    p_weights.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_weights.set_defaults(func=run_weights)

    p_umap = sub.add_parser("umap", help="Generate UMAP semantic-category plots")
    p_umap.add_argument("--label", default="wikitext")
    p_umap.add_argument("--vocab_path", default=DEFAULT_SEASONAL_VOCAB_PATH)
    p_umap.add_argument("--model_paths", nargs="+", default=list(DEFAULT_UMAP_MODEL_PATHS))
    p_umap.add_argument("--out_dir", default=str(Path(DEFAULT_PLOTS_ROOT) / "umap_semcats" / "wikitext"))
    p_umap.set_defaults(func=run_umap)

    p_structure = sub.add_parser("structure", help="Generate month/Roman structure sweep plots")
    p_structure.add_argument("--vocab_path", default=DEFAULT_SEASONAL_VOCAB_PATH)
    p_structure.add_argument("--plots_root", default=DEFAULT_PLOTS_ROOT)
    p_structure.add_argument("--latent_sizes", nargs="+", type=int, default=list(STRUCTURE_LATENT_SIZES))
    p_structure.add_argument("--model_path_template", default=DEFAULT_STRUCTURE_MODEL_TEMPLATE)
    p_structure.add_argument("--multi_seed_templates", nargs="+", default=list(DEFAULT_STRUCTURE_MULTI_SEED_TEMPLATES))
    p_structure.add_argument("--metrics_path", default=DEFAULT_STRUCTURE_METRICS_PATH)
    p_structure.add_argument("--metrics_fill_path", default=DEFAULT_STRUCTURE_METRICS_FILL_PATH)
    p_structure.set_defaults(func=run_structure)

    p_mech = sub.add_parser("mechanism", help="Generate mechanism figure (panels A/B/C)")
    p_mech.add_argument("--ls", type=int, required=True, help="Latent size.")
    p_mech.add_argument("--checkpoint_template", default=str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L{ls}_mse_wd1.0_seed1.pt"))
    p_mech.add_argument("--vocab_path", default=DEFAULT_SEASONAL_VOCAB_PATH)
    p_mech.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p_mech.add_argument("--r2_csv", default=str(Path(DEFAULT_FIGURES_DIR) / "r2_gap_valtest_minus_onehot_ls{ls}_sorted.csv"))
    p_mech.add_argument("--out_dir", default=DEFAULT_FIGURES_DIR)
    p_mech.add_argument("--target", default="mbar", help="Target word for panels B and C.")
    p_mech.add_argument("--top_k_contrib", type=int, default=4, help="Top co-occurring words to show in panel C.")
    p_mech.add_argument("--max_windows", type=int, default=100_000, help="Max windows to scan for pre-activation histogram.")
    p_mech.set_defaults(func=run_mechanism)

    p_all = sub.add_parser("all", help="Run the legacy full plotting pipeline")
    p_all.add_argument("--ls", type=int, default=DEFAULT_SEASONAL_LS)
    p_all.add_argument("--vocab_path", default=DEFAULT_SEASONAL_VOCAB_PATH)
    p_all.add_argument("--model_path_template", default=DEFAULT_SEASONAL_MODEL_TEMPLATE)
    p_all.add_argument("--plots_root", default=DEFAULT_PLOTS_ROOT)
    p_all.add_argument("--base_data_dir", default=DEFAULT_DATA_DIR)
    p_all.add_argument("--max_samples", type=int, default=MAX_SAMPLES_FOR_PCA)
    p_all.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_all.add_argument("--umap_model_paths", nargs="+", default=list(DEFAULT_UMAP_MODEL_PATHS))
    p_all.add_argument("--mechanism_ls", type=int, default=1000)
    p_all.add_argument("--mechanism_targets", nargs="+", default=["Beatles"])
    p_all.add_argument("--mechanism_checkpoint_template", default=str(Path(DEFAULT_MODELS_DIR) / "ae_wikitext_v10000_w20_seed1_L{ls}_mse_wd1.0_seed1.pt"))
    p_all.add_argument("--mechanism_r2_csv", default=str(Path(DEFAULT_FIGURES_DIR) / "r2_gap_valtest_minus_onehot_ls{ls}_sorted.csv"))
    p_all.add_argument("--top_k_contrib", type=int, default=4)
    p_all.add_argument("--max_windows", type=int, default=100_000)
    p_all.set_defaults(func=run_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
