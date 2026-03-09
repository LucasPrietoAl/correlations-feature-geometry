import argparse
import math
from pathlib import Path
from typing import Callable, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import zipfile

COLUMNS = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "lat",
    "lon",
    "feature_class",
    "feature_code",
    "country",
    "cc2",
    "admin1",
    "admin2",
    "admin3",
    "admin4",
    "population",
    "elevation",
    "dem",
    "timezone",
    "mod_date",
]


class CityPairDataset(Dataset):
    def __init__(self, pairs: np.ndarray) -> None:
        self.i = torch.from_numpy(pairs[:, 0].astype(np.int64))
        self.j = torch.from_numpy(pairs[:, 1].astype(np.int64))
        self.y = torch.from_numpy(pairs[:, 2].astype(np.int64))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.i[idx], self.j[idx], self.y[idx]


class CityCompassModel(nn.Module):
    def __init__(self, num_cities: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_cities, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.emb(i), self.emb(j)], dim=-1)
        return self.fc(z)


def load_cities(resource: str | None, min_population: int) -> pd.DataFrame:
    if resource is None:
        base = Path(__file__).resolve().parents[1]
        candidates = [base / "map" / "cities1000.txt", base / "map" / "cities1000.zip"]
        resource = next((str(path) for path in candidates if path.exists()), None)
    if resource is None:
        resource = "https://download.geonames.org/export/dump/cities1000.zip"
    if resource.startswith("http"):
        df = pd.read_csv(
            resource,
            sep="\t",
            names=COLUMNS,
            usecols=["name", "country", "lat", "lon", "population"],
            compression="zip",
        )
    else:
        path = Path(resource)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                with zf.open("cities1000.txt") as fh:
                    df = pd.read_csv(
                        fh,
                        sep="\t",
                        names=COLUMNS,
                        usecols=["name", "country", "lat", "lon", "population"],
                    )
        else:
            df = pd.read_csv(
                path,
                sep="\t",
                names=COLUMNS,
                usecols=["name", "country", "lat", "lon", "population"],
            )
    df = df[(df.country == "US") & df.population.gt(min_population)].reset_index(drop=True)
    return df


def compass_quadrant(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    lambda1 = np.radians(lon1)
    lambda2 = np.radians(lon2)
    d_lambda = lambda2 - lambda1
    x = np.sin(d_lambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lambda)
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return (((bearing + 22.5) // 45.0).astype(np.int64)) % 8


def generate_pairs(df: pd.DataFrame, n_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(df)
    i = rng.integers(0, n, size=n_pairs, endpoint=False)
    j = rng.integers(0, n, size=n_pairs, endpoint=False)
    mask = i == j
    while np.any(mask):
        j[mask] = rng.integers(0, n, size=mask.sum(), endpoint=False)
        mask = i == j
    lat = df.lat.to_numpy()
    lon = df.lon.to_numpy()
    labels = compass_quadrant(lat[i], lon[i], lat[j], lon[j])
    return np.stack([i, j, labels], axis=1).astype(np.int64)


def train_epoch(model: CityCompassModel, dataloader: DataLoader, loss_fn: nn.Module, optim: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    count = 0
    for i, j, y in dataloader:
        i = i.to(device)
        j = j.to(device)
        y = y.to(device)
        optim.zero_grad()
        logits = model(i, j)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total += loss.item() * y.size(0)
        count += y.size(0)
    return total / count


@torch.no_grad()
def evaluate(model: CityCompassModel, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device, projector: Callable[[torch.Tensor], torch.Tensor] | None = None) -> Tuple[float, float]:
    model.eval()
    total = 0.0
    count = 0
    correct = 0
    for i, j, y in dataloader:
        i = i.to(device)
        j = j.to(device)
        y = y.to(device)
        zi = model.emb(i)
        zj = model.emb(j)
        if projector is not None:
            zi = projector(zi)
            zj = projector(zj)
        logits = model.fc(torch.cat([zi, zj], dim=-1))
        loss = loss_fn(logits, y)
        preds = logits.argmax(dim=-1)
        total += loss.item() * y.size(0)
        count += y.size(0)
        correct += (preds == y).sum().item()
    return total / count, correct / count


def split_probe_indices(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    pop = df.population.to_numpy()
    top20 = np.argsort(-pop)[:20]
    remaining = np.setdiff1d(idx, top20)
    extra = rng.choice(remaining, size=int(0.30 * len(remaining)), replace=False)
    val_idx = np.sort(np.concatenate([top20, extra]))
    train_idx = np.setdiff1d(idx, val_idx)
    return train_idx, val_idx


def build_projectors(model: CityCompassModel, probe: Ridge, device: torch.device) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    coef = torch.from_numpy(probe.coef_).to(device, dtype=torch.float32)
    u, _ = torch.linalg.qr(coef.t())
    embedding = model.emb.weight
    mean_orth = (embedding - (embedding @ u) @ u.t()).mean(dim=0, keepdim=True)

    def remove_probe_raw(z: torch.Tensor) -> torch.Tensor:
        return z - (z @ u) @ u.t()

    def remove_probe_renorm(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        clean = z - (z @ u) @ u.t()
        orig = z.norm(dim=-1, keepdim=True) + eps
        new = clean.norm(dim=-1, keepdim=True) + eps
        return clean * (orig / new)

    def keep_probe_plus_mean(z: torch.Tensor) -> torch.Tensor:
        return (z @ u) @ u.t() + mean_orth

    return remove_probe_raw, remove_probe_renorm, keep_probe_plus_mean


def render_map(df: pd.DataFrame, probe: Ridge, embedding: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, output_path: Path) -> None:
    recon = probe.predict(embedding)
    lat = recon[:, 0]
    lon = recon[:, 1]
    is_train = np.isin(np.arange(len(df)), train_idx)
    is_val = np.isin(np.arange(len(df)), val_idx)
    drop = df.name.isin(["Honolulu", "Anchorage"]).to_numpy()
    keep = ~drop
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.scatter(lon[is_train & keep], lat[is_train & keep], s=12, alpha=0.35, color="#ff9999", label="train")
    ax.scatter(lon[is_val & keep], lat[is_val & keep], s=18, alpha=0.85, color="#1f77b4", label="validation")
    for city in ["Los Angeles", "New York City", "Miami", "Seattle", "Austin", "Chicago", "San Diego", "Denver", "Washington"]:
        rows = df.index[df.name.eq(city)].tolist()
        if rows:
            idx = rows[0]
            if keep[idx]:
                ax.text(lon[idx], lat[idx], city, ha="center", va="center", fontsize=16, weight="bold")
    mid_lat = np.mean(lat[keep])
    ax.set_aspect(1.0 / math.cos(math.radians(mid_lat)))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("grey")
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=True, loc="lower left", fontsize=16)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geonames", type=str, default=None)
    parser.add_argument("--min-population", type=int, default=50000)
    parser.add_argument("--emb-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-pairs", type=int, default=700000)
    parser.add_argument("--val-pairs", type=int, default=175000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    df = load_cities(args.geonames, args.min_population)
    train_pairs = generate_pairs(df, args.train_pairs, args.seed)
    val_pairs = generate_pairs(df, args.val_pairs, args.seed + 1)
    train_ds = CityPairDataset(train_pairs)
    val_ds = CityPairDataset(val_pairs)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False)

    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CityCompassModel(len(df), args.emb_dim, args.hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dl, loss_fn, optim, device)
        val_loss, val_acc = evaluate(model, val_dl, loss_fn, device)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.2%}")

    with torch.no_grad():
        embedding = model.emb.weight.detach().cpu().numpy()
    coords = df[["lat", "lon"]].to_numpy()
    train_idx, val_idx = split_probe_indices(df, args.seed)
    probe = Ridge(alpha=1e-3)
    probe.fit(embedding[train_idx], coords[train_idx])

    remove_raw, remove_renorm, keep_plus_mean = build_projectors(model, probe, device)
    base_loss, base_acc = evaluate(model, val_dl, loss_fn, device)
    abl_raw_loss, abl_raw_acc = evaluate(model, val_dl, loss_fn, device, remove_raw)
    abl_rn_loss, abl_rn_acc = evaluate(model, val_dl, loss_fn, device, remove_renorm)
    keep_loss, keep_acc = evaluate(model, val_dl, loss_fn, device, keep_plus_mean)

    print("\n──────── Ablation variants on fixed validation set ────────")
    print(f"BASELINE                : loss {base_loss:.4f}   acc {base_acc:.2%}")
    print(f"DROP probe  (raw)       : loss {abl_raw_loss:.4f}   acc {abl_raw_acc:.2%}")
    print(f"DROP probe  + re-norm   : loss {abl_rn_loss:.4f}   acc {abl_rn_acc:.2%}")
    print(f"KEEP probe + mean orth  : loss {keep_loss:.4f}   acc {keep_acc:.2%}")
    print("────────────────────────────────────────────────────────────")

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(__file__).resolve().parents[1] / "map"
    output_path = output_dir / "city_map_probe.pdf"
    render_map(df, probe, embedding, train_idx, val_idx, output_path)


if __name__ == "__main__":
    main()
