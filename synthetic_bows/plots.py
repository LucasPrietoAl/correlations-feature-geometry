"""Main figure for synthetic BOW experiments.

Produces a grid of W^T W Gram matrices for circular data across latent dims,
comparing linear vs ReLU autoencoders (with and without weight decay).

Usage:
    python -m synthetic_bows.plots
    python -m synthetic_bows.plots --results-dir synthetic_bows/results --output-dir synthetic_bows/figures
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from synthetic_bows.models import LinearAutoencoder

FS_TICK = 8
FS_LABEL = 9
FS_ROW_LABEL = 11
FS_COL_LABEL = 10

SEED = 42
DEFAULT_RESULTS_DIR = Path("synthetic_bows/results")
DEFAULT_OUTPUT_DIR = Path("synthetic_bows/figures")



def _wd_tag(value: float) -> str:
    cleaned = f"{value:g}".replace(".", "p").replace("-", "m")
    return f"wd_{cleaned}"


def find_experiment_dir(
    results_dir: Path,
    dtype: str,
    num_features: int,
    tie: bool,
    weight_decay: float,
) -> Path | None:
    """Return the experiment directory for the given configuration, or None."""
    base = results_dir / dtype / f"features_{num_features}" / ("tied" if tie else "untied")
    if not base.exists():
        return None
    if math.isclose(weight_decay, 0.0, abs_tol=1e-12):
        candidate = base / "config.json"
        return base if candidate.exists() else None
    candidate = base / _wd_tag(weight_decay)
    return candidate if (candidate / "config.json").exists() else None


def load_model(
    path: Path,
    latent_dim: int,
    relu: bool,
    tie: bool,
    num_features: int,
) -> LinearAutoencoder | None:
    model = LinearAutoencoder(num_features, latent_dim, use_relu=relu, tie_weights=tie)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as err:
        warnings.warn(f"Could not load {path}: {err}")
        return None


def gram_matrix(model: LinearAutoencoder) -> np.ndarray:
    """Return normalised W^T W: shape [num_features, num_features]."""
    W = model.encoder.weight.detach()         
    Wn = W / (W.norm() + 1e-9)        
    return (Wn.T @ Wn).numpy()


def plot_matrix(
    ax: plt.Axes,
    mat: np.ndarray | None,
    *,
    num_features: int,
    lim: tuple[float, float] | None = None,
    show_y: bool = False,
    colorbar: bool = False,
    col_label: str = "",
    row_label: str = "",
) -> None:
    if mat is None:
        ax.axis("off")
        return

    vmax = max(float(np.nanmax(np.abs(mat))), 1e-9) if lim is None else lim[1]
    vmin = -vmax if lim is None else lim[0]

    im = ax.imshow(
        mat,
        cmap="coolwarm",
        norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
        aspect="equal",
    )

    ticks = list(range(num_features))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=FS_TICK, rotation=90)
    if show_y:
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=FS_TICK)
    else:
        ax.set_yticks([])

    if colorbar:
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax).ax.tick_params(labelsize=FS_TICK)

    if col_label:
        ax.set_xlabel(col_label, fontsize=FS_COL_LABEL)

    if row_label:
        ax.annotate(
            row_label,
            xy=(-0.22, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=FS_ROW_LABEL,
            fontweight="bold",
        )


def make_main_figure(
    results_dir: Path,
    output_dir: Path,
    dtype: str = "circular",
    num_features: int = 12,
    wd_default: float = 1.0,
    wd_regularised: float = 6.0,
    exclude_dims: set[int] | None = None,
    seed: int = SEED,
) -> Path:
    if exclude_dims is None:
        exclude_dims = {5, 9}

    folder_default = find_experiment_dir(results_dir, dtype, num_features, tie=True, weight_decay=wd_default)
    folder_reg = find_experiment_dir(results_dir, dtype, num_features, tie=True, weight_decay=wd_regularised)
    if folder_default is None:
        raise FileNotFoundError(
            f"No results for {dtype}, features={num_features}, wd={wd_default} in {results_dir}"
        )
    if folder_reg is None:
        raise FileNotFoundError(
            f"No results for {dtype}, features={num_features}, wd={wd_regularised} in {results_dir}"
        )

    cfg = json.loads((folder_default / "config.json").read_text())
    all_dims: list[int] = sorted(cfg.get("latent_dims", []))
    target_dims = [d for d in all_dims if d not in exclude_dims]

    rows_config = [
        ("Linear", folder_default, "linear", False),
        ("ReLU", folder_default, "relu", True),
        (f"ReLU (wd={wd_regularised:g})", folder_reg, "relu", True),
    ]

    n_rows = len(rows_config)
    n_cols = len(target_dims) + 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
        constrained_layout=False,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.08, wspace=0.03, hspace=0.04)

    for ridx, (row_label, folder, act_tag, use_relu) in enumerate(rows_config):
        data_path = folder / f"base_{dtype}_data.pt"
        covariance = None
        if data_path.exists():
            data = torch.load(data_path, map_location="cpu", weights_only=True)
            covariance = np.cov(data[:, :num_features].numpy().T)

        matrices: dict[int, np.ndarray] = {}
        for latent_dim in target_dims:
            model_path = folder / act_tag / f"latent_{latent_dim}" / f"model_seed{seed}.pth"
            if not model_path.exists():
                continue
            model = load_model(model_path, latent_dim, use_relu, tie=True, num_features=num_features)
            if model is not None:
                matrices[latent_dim] = gram_matrix(model)

        lim = None
        if matrices:
            vmax = max(float(np.max(np.abs(m))) for m in matrices.values())
            vmax = max(vmax, 1e-9)
            lim = (-vmax, vmax)

        plot_matrix(
            axes[ridx, 0],
            covariance,
            num_features=num_features,
            lim=lim,
            show_y=True,
            col_label="Covariance" if ridx == n_rows - 1 else "",
            row_label=row_label,
        )
        if ridx == 0:
            axes[ridx, 0].set_title("Cov", fontsize=FS_COL_LABEL, pad=3)

        for cidx, latent_dim in enumerate(target_dims, 1):
            is_last_col = cidx == len(target_dims)
            plot_matrix(
                axes[ridx, cidx],
                matrices.get(latent_dim),
                num_features=num_features,
                lim=lim,
                colorbar=is_last_col,
            )
            if ridx == 0:
                axes[ridx, cidx].set_title(f"m={latent_dim}", fontsize=FS_COL_LABEL, pad=3)

    fig.canvas.draw()
    x_sep = (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0) / 2
    fig.lines.append(
        mlines.Line2D(
            [x_sep, x_sep], [0.06, 0.95],
            transform=fig.transFigure,
            color="black",
            lw=2,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dtype}_latent_sweep_features{num_features}"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved {out_path}")
    plt.close(fig)
    return output_dir / f"{stem}.pdf"



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot synthetic BOW main figure")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dtype", default="circular", help="Correlation structure type")
    p.add_argument("--num-features", type=int, default=12)
    p.add_argument("--wd-default", type=float, default=1.0, help="Weight decay for Linear/ReLU rows")
    p.add_argument("--wd-regularised", type=float, default=6.0, help="Weight decay for regularised ReLU row")
    p.add_argument(
        "--exclude-dims",
        type=int,
        nargs="*",
        default=[5, 9],
        help="Latent dims to exclude from the sweep",
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    make_main_figure(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        num_features=args.num_features,
        wd_default=args.wd_default,
        wd_regularised=args.wd_regularised,
        exclude_dims=set(args.exclude_dims),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
