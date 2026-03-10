# text_bows

Utilities for text bag-of-words dataset generation, autoencoder training, and figure production.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r text_bows/requirements.txt
```

## 1) Build dataset windows

```bash
python -m text_bows.make_dataset \
  --dataset wikitext \
  --vocab_size 10000 \
  --group_size 20 \
  --stride 1 \
  --save_dir ./text_bows/data
```

## 2) Train autoencoder

```bash
python -m text_bows.train_autoencoder \
  --dataset wikitext \
  --vocab_size 10000 \
  --group_size 20 \
  --stride 1 \
  --data_dir ./text_bows/data \
  --save_dir ./text_bows/models \
  --latent_size 1000 \
  --loss mse \
  --epochs 10 \
  --lr 5e-4 \
  --weight_decay 4.0 \
  --seed 1 \
  --device cpu
```

## 3) Generate plots

Quickstart plotting commands for the checkpoint above:

```bash
python -m text_bows.plots --help
python -m text_bows.plots months
python -m text_bows.plots weights
python -m text_bows.plots umap \
  --model_paths ./text_bows/models/ae_wikitext_v10000_w20_seed1_L1000_mse_wd4.0_seed1.pt
```

If you also want the per-word `val+test` vs one-word R2 tables used by the mechanism plot,
run:

```bash
python -m text_bows.validation_vs_singleword_r2 \
  --ls 1000 \
  --split val_test
```

This writes both:

- `text_bows/figures/r2_valtest_vs_singleword_ls1000.csv`
- `text_bows/figures/r2_gap_valtest_minus_onehot_ls1000_sorted.csv`

Legacy / paper-specific plotting commands such as `structure`, `mechanism`, and `all`
expect additional checkpoints or custom checkpoint templates. Use `python -m
text_bows.plots --help` for the full CLI.

## Example pipeline script

```bash
./text_bows/main_experiments.sh
```

`text_bows/main_experiments.sh` will also generate
`text_bows/figures/r2_gap_valtest_minus_onehot_ls${LATENT_SIZE}_sorted.csv`
automatically if it is missing.
