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
