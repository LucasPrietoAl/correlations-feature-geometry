#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-200}
NUM_FEATURES=${NUM_FEATURES:-12}
RESULTS_DIR=${RESULTS_DIR:-./synthetic_bows/results}
LATENT_DIMS=${LATENT_DIMS:-"2 3 4 5 6 7 8 9 10 11 12"}
OUTPUT_DIR=${OUTPUT_DIR:-./synthetic_bows/figures}

# Weight decays required for the main figure (plots.py).
# The plot shows three rows: Linear (wd=1), ReLU (wd=1), ReLU (wd=6).
WD_DEFAULT=${WD_DEFAULT:-1.0}
WD_REGULARISED=${WD_REGULARISED:-6.0}

# --- Training -----------------------------------------------------------------

for STRUCTURE in circular figure8 sphere iid; do
  # Train with the default weight decay (used for Linear and ReLU rows).
  "$PYTHON_BIN" -m synthetic_bows.train_autoencoders \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --num-features "$NUM_FEATURES" \
    --tie-weights \
    --correlation-structures "$STRUCTURE" \
    --latent-dims $LATENT_DIMS \
    --weight-decay "$WD_DEFAULT" \
    --results-dir "$RESULTS_DIR"

  # Train with the higher weight decay (used for the regularised ReLU row).
  "$PYTHON_BIN" -m synthetic_bows.train_autoencoders \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --num-features "$NUM_FEATURES" \
    --tie-weights \
    --correlation-structures "$STRUCTURE" \
    --latent-dims $LATENT_DIMS \
    --weight-decay "$WD_REGULARISED" \
    --results-dir "$RESULTS_DIR"
done

# --- Plotting -----------------------------------------------------------------

for STRUCTURE in circular figure8 sphere iid; do
  "$PYTHON_BIN" -m synthetic_bows.plots \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --dtype "$STRUCTURE" \
    --num-features "$NUM_FEATURES" \
    --wd-default "$WD_DEFAULT" \
    --wd-regularised "$WD_REGULARISED"
done
