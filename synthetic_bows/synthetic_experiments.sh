set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-200}
NUM_FEATURES=${NUM_FEATURES:-12}
RESULTS_DIR=${RESULTS_DIR:-./synthetic_bows/results}
LATENT_DIMS=${LATENT_DIMS:-"2 3 4 5 6 7 8 9 10 11 12"}

for STRUCTURE in circular figure8 sphere iid; do
  "$PYTHON_BIN" -m synthetic_bows.train_autoencoders \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --num-features "$NUM_FEATURES" \
    --tie-weights \
    --correlation-structures "$STRUCTURE" \
    --latent-dims $LATENT_DIMS \
    --results-dir "$RESULTS_DIR"
done
