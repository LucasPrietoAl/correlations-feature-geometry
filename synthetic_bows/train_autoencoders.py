from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import math

from synthetic_bows.models import LinearAutoencoder
from synthetic_bows.utils import _get_correlated_data, _iid_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _train(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int,
    optimiser: optim.Optimizer,
    device: torch.device,
    scheduler=None,
    *,
    amp: bool = False,
    grad_clip: float | None = None,
):
    """Generic training loop for the autoencoder."""
    crit = torch.nn.MSELoss()
    scaler = GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    losses: List[float] = []

    model.train()
    for ep in range(epochs):
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(amp and device.type == "cuda")):
                recon, _ = model(batch)
                loss = crit(recon, batch)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimiser)
            scaler.update()
            running += loss.item()
        losses.append(running / len(loader))
        if scheduler:
            scheduler.step()
        if ep in {0, epochs - 1} or (ep + 1) % 50 == 0:
            logging.info("Epoch %3d/%d  loss %.6f", ep + 1, epochs, losses[-1])
    return torch.tensor(losses)


def _split_dataset(
    data: torch.Tensor,
    train_size: int,
    val_size: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total = data.shape[0]
    if total == 0:
        return data, data

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(total, generator=generator)

    train_size = min(train_size, total)
    train_idx = perm[:train_size]
    remaining = perm[train_size:]

    if val_size is None:
        val_idx = remaining
    else:
        take = min(max(val_size, 0), remaining.numel())
        val_idx = remaining[:take]

    if val_idx.numel() == 0:
        val_idx = train_idx

    return data.index_select(0, train_idx), data.index_select(0, val_idx)

def _weight_decay_tag(value: float) -> str:
    value = float(value)
    cleaned = f"{value:g}".replace(".", "p").replace("-", "m")
    return f"wd_{cleaned}"


def _exp_dir(base: str, dtype: str, args) -> Path:
    """Return directory for experiment results.

    Layout ensures paths encode correlation structure, dimensionality, and
    whether weights are tied. Non-zero weight decays are placed in a dedicated
    subdirectory to avoid overwriting runs with different regularisation.
    """

    parts = [dtype, f"features_{args.num_features}"]
    parts.append("tied" if args.tie_weights else "untied")
    wd = getattr(args, "weight_decay", 0.0)
    if not math.isclose(wd, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        parts.append(_weight_decay_tag(wd))
    d = Path(base).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save(pt: torch.Tensor, path: Path):
    torch.save(pt, path)
    logging.info("saved %s", path.name)


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    criterion = torch.nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            total += loss.item()
            count += batch.size(0)
    if count == 0:
        return float("nan")
    return total / count

def run(args):
    device = torch.device("cpu" if not args.device.startswith("cuda") or not torch.cuda.is_available() else args.device)
    logging.info("device: %s", device)


    datasets: Dict[str, Tuple[torch.Tensor, float]] = {}
    first_sparsity = None


    for data_type in args.correlation_structures:
        if data_type == "iid":
            continue
        logging.info("\n--- Generating %s data ---", data_type)
        data, sparsity = _get_correlated_data(data_type, args)
        datasets[data_type] = (data, sparsity)
        if first_sparsity is None:
            first_sparsity = sparsity

    if 'iid' in args.correlation_structures:
        logging.info("\n--- Generating IID data ---")
        p = first_sparsity
        if p is None:
            p = 1.0 / (1.0 + math.exp(-args.feature_base_log_odds))
            logging.warning("No correlated data generated. Using default sparsity p=%.4f for IID data.", p)
        iid_data = _iid_data(args.num_samples, p, args.data_seed, args.num_features)
        datasets['iid'] = (iid_data, p)
    

    for dtype, (base_data, p) in datasets.items():
        logging.info("\n=== Training on %s data (p=%.4f) ===", dtype, p)

        exp_dir = _exp_dir(args.results_dir, dtype, args)
        cfg = vars(args).copy()
        cfg.update({"data_type": dtype, "avg_activation_prob": p, "correlation_structures": args.correlation_structures})
        (exp_dir / "config.json").write_text(json.dumps(cfg, indent=4, sort_keys=True))
        _save(base_data, exp_dir / f"base_{dtype}_data.pt")

        train_data, val_data = _split_dataset(
            base_data,
            args.num_samples_train,
            args.num_samples_val,
            args.data_seed,
        )
        if val_data.shape[0] == 0:
            logging.warning("Validation split empty; reusing training data for validation")
            val_data = train_data
        for seed in args.seeds:
            logging.info("-- seed %d --", seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            train_generator = torch.Generator()
            train_generator.manual_seed(seed)

            loader = DataLoader(
                TensorDataset(train_data),
                batch_size=args.batch_size,
                shuffle=True,
                generator=train_generator,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda" and args.num_workers > 0),
                persistent_workers=args.num_workers > 0,
            )

            val_loader = DataLoader(
                TensorDataset(val_data),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda" and args.num_workers > 0),
                persistent_workers=args.num_workers > 0,
            )

            for d in args.latent_dims:
                logging.info("-- latent dim %d --", d)
                for tag, relu in [("linear", False), ("relu", True)]:
                    name = f"{tag}_lat{d}"
                    logging.info("   training %s", name)
                    model = LinearAutoencoder(args.num_features, d, use_relu=relu, tie_weights=args.tie_weights, use_bias=True).to(device)
                    opt = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
                    losses = _train(model, loader, args.epochs, opt, device, scheduler=sched, amp=args.use_amp, grad_clip=args.grad_clip_norm)
                    model_dir = exp_dir / tag / f"latent_{d}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model_path = model_dir / f"model_seed{seed}.pth"
                    torch.save(model.state_dict(), model_path)
                    logging.info("saved %s", model_path.name)
                    _save(losses, model_dir / f"losses_seed{seed}.pt")
                    val_loss = _evaluate(model, val_loader, device)
                    logging.info("   validation loss %.6f", val_loss)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic bag-of-words autoencoders.")

    g_data = parser.add_argument_group("data generation")
    g_data.add_argument("--correlation-structures", type=str, nargs="+", 
                       default=["circular", "figure8", "sphere", "iid"],
                       help="Types of data correlation to run experiments on.")
    g_data.add_argument("--num-samples", type=int, default=10_000)
    g_data.add_argument("--num-features", type=int, default=12)

    g_data.add_argument("--num-samples-train", type=int, default=8000)
    g_data.add_argument("--num-samples-val", type=int, default=None)
    g_data.add_argument("--data-seed", type=int, default=42)
    g_data.add_argument("--feature-sharpness", type=float, default=10.0)
    g_data.add_argument("--feature-base-log-odds", type=float, default=-.0)
    g_data.add_argument("--feature-noise", type=float, default=0.)

    g_train = parser.add_argument_group("training")
    g_train.add_argument("--latent-dims", type=int, nargs="+", default=list(range(2, 13)))
    g_train.add_argument("--batch-size", type=int, default=256)
    g_train.add_argument("--epochs", type=int, default=200)
    g_train.add_argument("--learning-rate", type=float, default=1e-2)
    g_train.add_argument("--min-lr", type=float, default=1e-9)
    g_train.add_argument("--weight-decay", type=float, default=0)
    g_train.add_argument("--grad-clip-norm", type=float, default=1.0)
    g_train.add_argument("--tie-weights", action="store_true")
    g_train.add_argument("--seeds", type=int, nargs="+", default=[42])
    g_train.add_argument("--use-amp", action="store_true")
    g_train.add_argument("--device", type=str, default="cuda:0")
    g_train.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--results-dir", type=str, default="results")

    args = parser.parse_args()

    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0:
        args.grad_clip_norm = None
    if args.use_amp and not args.device.startswith("cuda"):
        logging.warning("AMP requested but CUDA not available – disabling AMP")
        args.use_amp = False
        
    return args


def main() -> None:
    args = _parse_args()
    logging.info("config\n%s", json.dumps(vars(args), indent=4, sort_keys=True))
    run(args)

if __name__ == "__main__":
    main()
