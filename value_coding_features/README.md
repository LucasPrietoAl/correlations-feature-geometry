# value_coding_features

Experiments for compass-direction prediction from learned city embeddings and geographic probe ablations.

## Install

See the top-level `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run pipeline

```bash
python -m value_coding_features.city_map_pipeline \
  --epochs 40 \
  --batch-size 1024 \
  --eval-batch-size 4096 \
  --device cpu
```

Optional arguments:
- `--geonames` path/URL to geonames dump (defaults to local cache candidates or geonames URL)
- `--output-dir` output directory for rendered map figure

