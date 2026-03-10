# text_bows

Utilities for text bag-of-words dataset generation, autoencoder training, and figure production.

## Install

From the repository root:

```bash
uv sync
```

## 1) Build dataset windows

Tokenisation is **case-sensitive** (the default). Tokens like "January" and
"january" are distinct vocabulary entries, which is important for the month and
Roman-numeral structure analyses.

```bash
uv run python -m text_bows.make_dataset \
  --dataset wikitext \
  --vocab_size 10000 \
  --group_size 20 \
  --stride 1 \
  --save_dir ./text_bows/data
```

## 2) Train autoencoders

Two weight-decay settings are used. `wd=4.0` produces the months, weights, and
UMAP figures; `wd=1.0` produces the mechanism figure (and the structure sweep
when running extended experiments).

```bash
# wd=4.0 — months / weights / umap
uv run python -m text_bows.train_autoencoder \
  --dataset wikitext --vocab_size 10000 --group_size 20 --stride 1 \
  --data_dir ./text_bows/data --save_dir ./text_bows/models \
  --latent_size 1000 --loss mse --epochs 10 --lr 5e-4 \
  --weight_decay 4.0 --seed 1 --device cpu

# wd=1.0 — mechanism (and structure)
uv run python -m text_bows.train_autoencoder \
  --dataset wikitext --vocab_size 10000 --group_size 20 --stride 1 \
  --data_dir ./text_bows/data --save_dir ./text_bows/models \
  --latent_size 1000 --loss mse --epochs 10 --lr 5e-4 \
  --weight_decay 1.0 --seed 1 --device cpu
```

## 3) Generate plots

```bash
uv run python -m text_bows.plots --help
uv run python -m text_bows.plots months          # wd=4.0 checkpoint
uv run python -m text_bows.plots weights         # wd=4.0 checkpoint
uv run python -m text_bows.plots umap \          # wd=4.0 checkpoint
  --model_paths ./text_bows/models/ae_wikitext_v10000_w20_seed1_L1000_mse_wd4.0_seed1.pt
uv run python -m text_bows.plots mechanism --ls 1000   # wd=1.0 checkpoint (default template)
```

The per-word R2 table used by mechanism panel A can be generated separately:

```bash
uv run python -m text_bows.validation_vs_singleword_r2 \
  --ls 1000 \
  --split val_test
```

This writes:

- `text_bows/figures/r2_valtest_vs_singleword_ls1000.csv`
- `text_bows/figures/r2_gap_valtest_minus_onehot_ls1000_sorted.csv`

## Pipeline scripts

**Default run** — builds the dataset, trains two models (L=1000 at wd=4.0 and
wd=1.0), computes the R2 table, and generates months, weights, umap, and
mechanism figures:

```bash
DEVICE=cuda:0 bash text_bows/main_experiments.sh
```

**Extended run** — additionally trains models across 16 latent sizes and 3 seeds
(48 checkpoints) to produce the structure sweep figures from the paper:

```bash
DEVICE=cuda:0 bash text_bows/main_experiments.sh --extended
```
