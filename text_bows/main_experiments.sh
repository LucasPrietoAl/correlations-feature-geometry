#!/usr/bin/env bash
set -euo pipefail
# --------------------------------------------------------------------------
# Text bag-of-words experiments.
#
# Default run: builds dataset, trains two autoencoders (L=1000, at wd=4.0
# and wd=1.0), and generates months, weights, umap, and mechanism plots.
#
# Extended run (--extended): additionally trains models at wd=1.0 across
# 16 latent sizes and 3 seeds (48 checkpoints) and generates the structure
# sweep figures.
# --------------------------------------------------------------------------

EXTENDED=false
for arg in "$@"; do
  case "$arg" in
    --extended) EXTENDED=true ;;
  esac
done

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET=${DATASET:-wikitext}
VOCAB_SIZE=${VOCAB_SIZE:-10000}
GROUP_SIZE=${GROUP_SIZE:-20}
STRIDE=${STRIDE:-1}
LATENT_SIZE=${LATENT_SIZE:-1000}
SEED=${SEED:-1}
DEVICE=${DEVICE:-cpu}
WEIGHT_DECAY=${WEIGHT_DECAY:-4.0}

# Weight decay used by the mechanism and structure plots (matches the
# checkpoint templates hard-coded in plots.py).
MECH_WD=1.0

STRIDE_TAG=""
if [[ -n "${STRIDE}" && "${STRIDE}" != "1" ]]; then
  STRIDE_TAG="_s${STRIDE}"
fi
VOCAB_PATH="./text_bows/data/${DATASET}_vocab_v${VOCAB_SIZE}.pt"
MODEL_TEMPLATE="./text_bows/models/ae_${DATASET}_v${VOCAB_SIZE}_w${GROUP_SIZE}_seed${SEED}${STRIDE_TAG}_L{ls}_mse_wd${WEIGHT_DECAY}_seed${SEED}.pt"
MODEL_PATH="./text_bows/models/ae_${DATASET}_v${VOCAB_SIZE}_w${GROUP_SIZE}_seed${SEED}${STRIDE_TAG}_L${LATENT_SIZE}_mse_wd${WEIGHT_DECAY}_seed${SEED}.pt"

# ===========================================================================
# 1) Build sparse bag-of-words windows (case-sensitive tokenisation).
# ===========================================================================
# NOTE: The paper uses case-sensitive vocabulary (--lowercase is NOT passed).
# Tokens like "January" and "january" are distinct entries.  Do not add
# --lowercase here unless you intend to change the experiment.
"$PYTHON_BIN" -m text_bows.make_dataset \
  --dataset "$DATASET" \
  --vocab_size "$VOCAB_SIZE" \
  --group_size "$GROUP_SIZE" \
  --stride "$STRIDE" \
  --save_dir ./text_bows/data

# ===========================================================================
# 2) Train autoencoders.
# ===========================================================================

# wd=4.0 checkpoint — used by months, weights, and umap plots.
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

# wd=1.0 checkpoint — used by the mechanism plot.
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
  --weight_decay "$MECH_WD" \
  --seed "$SEED" \
  --device "$DEVICE"

# ===========================================================================
# 3) Generate figures.
# ===========================================================================

# Months figure (wd=4.0 checkpoint)
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

# Weight PCA figure (wd=4.0 checkpoint)
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

# UMAP semantic categories figure (wd=4.0 checkpoint)
"$PYTHON_BIN" -m text_bows.plots umap \
  --vocab_path "$VOCAB_PATH" \
  --model_paths "$MODEL_PATH" \
  --out_dir "./text_bows/figures/umap_semcats/${DATASET}"

# Mechanism figure (wd=1.0 checkpoint — uses plots.py defaults)
"$PYTHON_BIN" -m text_bows.plots mechanism \
  --ls "$LATENT_SIZE" \
  --vocab_path "$VOCAB_PATH" \
  --data_dir ./text_bows/data \
  --out_dir ./text_bows/figures

# ===========================================================================
# 4) Extended: train across latent sizes × seeds for the structure figure.
# ===========================================================================
if $EXTENDED; then
  echo ""
  echo "=== Extended sweep: training across latent sizes and seeds (wd=${MECH_WD}) ==="
  echo ""

  STRUCTURE_LATENT_SIZES=(50 75 100 125 150 200 250 300 400 600 800 1000 2000 4000 6000 10000)
  STRUCTURE_SEEDS=(1 2 3)

  for S in "${STRUCTURE_SEEDS[@]}"; do
    for LS in "${STRUCTURE_LATENT_SIZES[@]}"; do
      CKPT="./text_bows/models/ae_${DATASET}_v${VOCAB_SIZE}_w${GROUP_SIZE}_seed${S}${STRIDE_TAG}_L${LS}_mse_wd${MECH_WD}_seed${S}.pt"
      if [[ -f "$CKPT" ]]; then
        echo "Checkpoint already exists, skipping: $CKPT"
        continue
      fi
      echo "Training seed=${S} L=${LS} wd=${MECH_WD} ..."
      "$PYTHON_BIN" -m text_bows.train_autoencoder \
        --dataset "$DATASET" \
        --vocab_size "$VOCAB_SIZE" \
        --group_size "$GROUP_SIZE" \
        --stride "$STRIDE" \
        --data_dir ./text_bows/data \
        --save_dir ./text_bows/models \
        --latent_size "$LS" \
        --loss mse \
        --epochs 10 \
        --lr 5e-4 \
        --weight_decay "$MECH_WD" \
        --seed "$S" \
        --device "$DEVICE"
    done
  done

  # Structure plot — uses plots.py defaults (wd=1.0 templates for all seeds).
  "$PYTHON_BIN" -m text_bows.plots structure \
    --vocab_path "$VOCAB_PATH" \
    --plots_root ./text_bows/figures \
    --latent_sizes "${STRUCTURE_LATENT_SIZES[@]}"
fi
