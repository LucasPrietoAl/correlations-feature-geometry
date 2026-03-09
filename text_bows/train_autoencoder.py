from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from contextlib import nullcontext
import numpy as np

from text_bows.utils_io import (
    SparseNPZDataset,
    find_existing_path,
    meta_for,
    path_candidates,
)


def build_dataset(path: Path) -> Tuple[Dataset, int]:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Expected .npz file, got: {path.suffix}")
    meta = meta_for(path)
    ds = SparseNPZDataset(str(path))
    vocab = int(meta["vocab_size"]) if (meta and "vocab_size" in meta) else ds.vocab_size()
    return ds, vocab

def split_train_val_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")
    if n < 2 or val_fraction == 0:
        return list(range(n)), []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_count = max(1, int(round(n * val_fraction)))
    val_count = min(val_count, n - 1)
    val_idx = perm[:val_count].tolist()
    train_idx = perm[val_count:].tolist()
    return train_idx, val_idx

def collate_indices(batch: List[List[int]]) -> Tuple[List[torch.LongTensor], torch.LongTensor]:
    batch = [b for b in batch if b]
    xs = [torch.as_tensor(b, dtype=torch.long) for b in batch]
    cnts = torch.tensor([len(b) for b in batch], dtype=torch.long)
    return xs, cnts


def dense_target(xs: List[torch.Tensor], cnts: torch.Tensor, inp_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if len(xs) == 0 or cnts.sum().item() == 0:
        return torch.zeros((len(xs), inp_dim), dtype=dtype, device=device)
    cols = torch.cat(xs).to(device)
    rows = torch.repeat_interleave(torch.arange(len(xs), device=device), cnts.to(device))
    idx = torch.stack([rows, cols])
    val = torch.ones(int(cnts.sum()), device=device, dtype=dtype)
    sp = torch.sparse_coo_tensor(idx, val, (len(xs), inp_dim), device=device).coalesce()
    return sp.to_dense()



class Autoencoder(nn.Module):
    def __init__(self, inp: int, hid: int, relu_latent: bool, linear_out: bool = False):
        super().__init__()
        self.inp, self.hid, self.relu_lat = inp, hid, relu_latent
        self.enc = nn.Linear(inp, hid, bias=False)
        nn.init.kaiming_uniform_(self.enc.weight, a=2**0.5)
        self.bias = nn.Parameter(torch.zeros(inp))
        self.out_act = None if linear_out else nn.ReLU()

    def encode(self, xs: List[torch.Tensor], cnts: torch.Tensor):
        dev, dt = self.enc.weight.device, self.enc.weight.dtype
        B, total = len(xs), int(cnts.sum())
        if total == 0:
            return torch.zeros(B, self.hid, device=dev, dtype=dt)

        rows = torch.repeat_interleave(torch.arange(B, device=dev), cnts.to(dev))
        cols = torch.cat(xs).to(dev)
        idx = torch.stack([rows, cols])
        val = torch.ones(total, device=dev, dtype=torch.float32)
        mask = torch.sparse_coo_tensor(idx, val, (B, self.inp), device=dev).coalesce()

        with torch.amp.autocast("cuda", enabled=False):
            lat = torch.sparse.mm(mask, self.enc.weight.t().float())
        return F.relu(lat) if self.relu_lat else lat

    def decode(self, lat):
        out = F.linear(lat, self.enc.weight.T, self.bias)
        return self.out_act(out) if self.out_act is not None else out

    def forward(self, xs, cnts):
        return self.decode(self.encode(xs, cnts))

@dataclass
class TrainCfg:
    loss: str
    mse_sigmoid: bool
    latent_size: int
    relu_latent: bool
    linear_out: bool
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: Optional[float]
    num_workers: int
    pin_memory: bool
    val_every: int
    seed: int

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, vocab: int, loss_name: str, mse_sigmoid: bool, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    crit = nn.MSELoss(reduction="mean") if loss_name == "mse" else nn.BCEWithLogitsLoss(reduction="mean")
    amp_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    total_sum, total_elems = 0.0, 0
    skipped_batches = 0
    with torch.inference_mode():
        with amp_ctx:
            for xs, cnts in loader:
                if not xs:
                    skipped_batches += 1
                    continue
                tgt = dense_target(xs, cnts, vocab, device, torch.float32)
                out = model(xs, cnts)
                if loss_name == "mse" and mse_sigmoid:
                    out = torch.sigmoid(out)
                loss = crit(out, tgt)
                elems = tgt.numel()
                total_sum += float(loss.item()) * elems
                total_elems += elems

    if skipped_batches:
        logging.info("Skipped %d empty validation batches.", skipped_batches)
    if total_elems == 0:
        logging.warning("Validation had no non-empty batches; returning loss=0.0.")

    if was_training:
        model.train()
    return total_sum / max(1, total_elems)

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    vocab: int,
    cfg: TrainCfg,
    device: torch.device,
    save_dir: Path,
    run_tag: str,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss(reduction="mean") if cfg.loss == "mse" else nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    amp_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    model.to(device)
    t0 = time.time()

    ckpt_stem = f"ae_{run_tag}_L{cfg.latent_size}_{cfg.loss}_wd{cfg.weight_decay}_seed{cfg.seed}{'_linear' if cfg.linear_out else ''}"
    ckpt_path = save_dir / f"{ckpt_stem}.pt"

    for ep in range(1, cfg.epochs + 1):
        ep_sum, ep_elems = 0.0, 0
        skipped_batches = 0

        for xs, cnts in train_loader:
            if not xs:
                skipped_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                out = model(xs, cnts)
                if cfg.loss == "mse" and cfg.mse_sigmoid:
                    out = torch.sigmoid(out)
                tgt = dense_target(xs, cnts, vocab, device, torch.float32)
                loss = criterion(out, tgt)

            scaler.scale(loss).backward()

            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            elems = tgt.numel()
            ep_sum += float(loss.item()) * elems
            ep_elems += elems

        train_loss = ep_sum / max(1, ep_elems)
        if skipped_batches:
            logging.info("Skipped %d empty training batches in epoch %d.", skipped_batches, ep)
        if ep_elems == 0:
            logging.warning("Epoch %d had no non-empty training batches; train_loss set to 0.0.", ep)
        msg = f"Epoch {ep}/{cfg.epochs} | train_loss={train_loss:.6f} | "

        val_loss = None
        if val_loader is not None and (ep % max(1, cfg.val_every) == 0 or ep == cfg.epochs):
            val_loss = evaluate(model, val_loader, vocab, cfg.loss, cfg.mse_sigmoid, device)
            msg += f"val_loss={val_loss:.6f} | "

        logging.info(msg + f"{time.time() - t0:.1f}s")

        torch.save(
            {
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": {
                    "run_tag": run_tag,
                    "epochs": cfg.epochs,
                    "loss": cfg.loss,
                    "latent_size": cfg.latent_size,
                    "relu_latent": cfg.relu_latent,
                    "linear_out": cfg.linear_out,
                    "lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "grad_clip": cfg.grad_clip,
                    "batch_size": cfg.batch_size,
                },
                "metrics": {"train_loss": train_loss, **({"val_loss": val_loss} if val_loss is not None else {})},
            },
            str(ckpt_path),
        )
        logging.info("Saved checkpoint → %s", ckpt_path)

    final_path = save_dir / f"{ckpt_stem}_last.pt"
    torch.save(
        {
            "epoch": cfg.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": {"run_tag": run_tag},
            "metrics": {},
        },
        str(final_path),
    )
    logging.info("Saved final checkpoint → %s", final_path)

def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train a small tied-weight linear AE on sparse NPZ windows")

    p.add_argument("--dataset", choices=["wikitext", "openwebtext"], default="wikitext")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--group_size", type=int, required=True)
    p.add_argument("--stride", type=int, help="Stride used when creating the dataset (if omitted, auto-detect)")

    p.add_argument("--loss", choices=["mse", "bce"], default="mse")
    p.add_argument("--mse_sigmoid", action="store_true", help="Apply sigmoid to outputs before MSE (keeps predictions in [0,1])")
    p.add_argument("--latent_size", type=int, default=4096)
    p.add_argument("--relu_latent", action="store_true")
    p.add_argument("--linear_out", action="store_true", help="Use linear decoder output (no ReLU). Default is ReLU output.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=4.0)
    p.add_argument("--grad_clip", type=float)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--device", help="cpu | cuda | cuda:0 | … (defaults to CUDA if available)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of train windows reserved for validation (set 0 to use validation split file if present).")
    p.add_argument("--val_seed", type=int, default=42, help="Seed for train/val split when using --val_fraction.")

    p.add_argument("--save_dir", default="./models")
    p.add_argument("--run_name", default="")

    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if not 0 <= args.val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    train_cands = path_candidates(args.data_dir, args.dataset, "train", args.vocab_size, args.group_size, args.stride)
    val_cands = path_candidates(args.data_dir, args.dataset, "validation", args.vocab_size, args.group_size, args.stride)

    train_path = find_existing_path(train_cands)
    if not train_path:
        raise FileNotFoundError("Could not find a training dataset matching provided parameters.")
    val_path = find_existing_path(val_cands)

    logging.info("Using train file: %s", train_path.name)
    if val_path:
        logging.info("Found val file:   %s", val_path.name)

    train_ds_full, vocab = build_dataset(train_path)
    val_ds = None
    if args.val_fraction > 0:
        if val_path:
            logging.info("Ignoring validation file %s due to --val_fraction=%.3f", val_path.name, args.val_fraction)
        train_idx, val_idx = split_train_val_indices(len(train_ds_full), args.val_fraction, args.val_seed)
        train_ds = Subset(train_ds_full, train_idx)
        if val_idx:
            val_ds = Subset(train_ds_full, val_idx)
        logging.info(
            "Using train-derived validation split: %d train / %d val windows",
            len(train_idx),
            len(val_idx),
        )
    else:
        train_ds = train_ds_full
        if val_path:
            val_ds, vocab_val = build_dataset(val_path)
            if vocab_val != vocab:
                logging.warning("Validation vocab (%d) differs from train vocab (%d). Using train vocab.", vocab_val, vocab)
        else:
            val_ds = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_indices
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_indices
        )

    model = Autoencoder(inp=vocab, hid=args.latent_size, relu_latent=args.relu_latent, linear_out=args.linear_out)

    cfg = TrainCfg(
        loss=args.loss,
        mse_sigmoid=bool(args.mse_sigmoid),
        latent_size=args.latent_size,
        relu_latent=bool(args.relu_latent),
        linear_out=bool(args.linear_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        val_every=args.val_every,
        seed=args.seed
    )

    run_tag = f"{args.dataset}_v{args.vocab_size}_w{args.group_size}_seed{args.seed}"
    if args.stride not in (None, 1):
        run_tag += f"_s{args.stride}"
    if args.run_name:
        run_tag += f"_{args.run_name}"
    save_dir = Path(args.save_dir)

    train(model, train_loader, val_loader, vocab, cfg, device, save_dir, run_tag)


if __name__ == "__main__":
    main()
