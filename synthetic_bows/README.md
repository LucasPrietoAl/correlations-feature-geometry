# synthetic_bows

Synthetic bag-of-words experiments for testing feature geometry under controlled correlation structures.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r synthetic_bows/requirements.txt
```

## Train synthetic autoencoders

```bash
python -m synthetic_bows.train_autoencoders \
  --correlation-structures circular figure8 sphere iid \
  --tie-weights \
  --num-features 12 \
  --epochs 200 \
  --device cpu \
  --results-dir ./synthetic_bows/results
```

## Run example sweep script

```bash
./synthetic_bows/synthetic_experiments.sh
```

Environment variable overrides:
- `PYTHON_BIN`
- `DEVICE`
- `EPOCHS`
- `NUM_FEATURES`
- `RESULTS_DIR`
- `LATENT_DIMS`

## Outputs

- Models: `synthetic_bows/results/**/model_seed*.pth`
- Training losses: `synthetic_bows/results/**/losses_seed*.pt`
- Config snapshots: `synthetic_bows/results/**/config.json`
