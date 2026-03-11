# synthetic_bows

Synthetic bag-of-words experiments for testing feature geometry under controlled correlation structures.

## Install

From the repository root:

```bash
uv sync
```

## Train synthetic autoencoders

The main figure compares W^T W Gram matrices across latent dimensions for Linear
and ReLU autoencoders at two weight-decay levels (`--weight-decay 1.0` and `6.0`).

```bash
uv run python -m synthetic_bows.train_autoencoders \
  --correlation-structures circular figure8 sphere iid \
  --tie-weights \
  --num-features 12 \
  --epochs 200 \
  --weight-decay 1.0 \
  --device cpu \
  --results-dir ./synthetic_bows/results
```

## Generate plots

```bash
uv run python -m synthetic_bows.plots \
  --results-dir ./synthetic_bows/results \
  --output-dir ./synthetic_bows/figures \
  --dtype circular
```

## Full pipeline (train + plot)

The experiment script trains all correlation structures at both weight-decay
values and then generates the figures:

```bash
DEVICE=cuda:0 bash synthetic_bows/synthetic_experiments.sh
```
