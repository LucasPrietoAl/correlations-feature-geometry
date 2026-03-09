#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET=${DATASET:-wikitext}
VOCAB_SIZE=${VOCAB_SIZE:-10000}
GROUP_SIZE=${GROUP_SIZE:-20}
STRIDE=${STRIDE:-1}
LATENT_SIZE=${LATENT_SIZE:-1000}
SEED=${SEED:-1}
DEVICE=${DEVICE:-cpu}
WEIGHT_DECAY=${WEIGHT_DECAY:-4.0}

STRIDE_TAG=""
if [[ -n "${STRIDE}" && "${STRIDE}" != "1" ]]; then
  STRIDE_TAG="_s${STRIDE}"
fi
VOCAB_PATH="./text_bows/data/${DATASET}_vocab_v${VOCAB_SIZE}.pt"
MODEL_TEMPLATE="./text_bows/models/ae_${DATASET}_v${VOCAB_SIZE}_w${GROUP_SIZE}_seed${SEED}${STRIDE_TAG}_L{ls}_mse_wd${WEIGHT_DECAY}_seed${SEED}.pt"
MODEL_PATH="./text_bows/models/ae_${DATASET}_v${VOCAB_SIZE}_w${GROUP_SIZE}_seed${SEED}${STRIDE_TAG}_L${LATENT_SIZE}_mse_wd${WEIGHT_DECAY}_seed${SEED}.pt"

# 1) Build sparse bag-of-words windows.
"$PYTHON_BIN" -m text_bows.make_dataset \
  --dataset "$DATASET" \
  --vocab_size "$VOCAB_SIZE" \
  --group_size "$GROUP_SIZE" \
  --stride "$STRIDE" \
  --save_dir ./text_bows/data

# 2) Train the tied-weight autoencoder.
"$PYTHON_BIN" -m text_bows.train_autoencoder \
  --dataset "$DATASET" \
  --vocab_size "$VOCAB_SIZE" \
  --group_size "$GROUP_SIZE" \
  --stride "$STRIDE" \
  --data_dir ./text_bows/data \
  --save_dir ./text_bows/models \
  --latent_size "$LATENT_SIZE" \
  --loss mse \
  --epochs 10 \
  --lr 5e-4 \
  --weight_decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --device "$DEVICE"

# 3) Generate the public quickstart figures against the trained checkpoint.
"$PYTHON_BIN" -m text_bows.plots months \
  --ls "$LATENT_SIZE" \
  --dataset "$DATASET" \
  --vocab_size "$VOCAB_SIZE" \
  --group_size "$GROUP_SIZE" \
  --stride "$STRIDE" \
  --vocab_path "$VOCAB_PATH" \
  --model_path_template "$MODEL_TEMPLATE" \
  --plots_root ./text_bows/figures \
  --base_data_dir ./text_bows/data

"$PYTHON_BIN" -m text_bows.plots weights \
  --ls "$LATENT_SIZE" \
  --dataset "$DATASET" \
  --vocab_size "$VOCAB_SIZE" \
  --group_size "$GROUP_SIZE" \
  --stride "$STRIDE" \
  --vocab_path "$VOCAB_PATH" \
  --model_path_template "$MODEL_TEMPLATE" \
  --plots_root ./text_bows/figures \
  --base_data_dir ./text_bows/data

"$PYTHON_BIN" -m text_bows.plots umap \
  --vocab_path "$VOCAB_PATH" \
  --model_paths "$MODEL_PATH" \
  --out_dir "./text_bows/figures/umap_semcats/${DATASET}"
