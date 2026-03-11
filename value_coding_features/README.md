# value_coding_features

Experiments for compass-direction prediction from learned city embeddings and geographic probe ablations.

## Install

From the repository root:

```bash
uv sync
```

## Run pipeline

```bash
uv run python -m value_coding_features.city_map_pipeline \
  --epochs 40 \
  --batch-size 1024 \
  --eval-batch-size 4096 \
  --device cpu  # or cuda:0 / mps
```

Optional arguments:
- `--geonames` path/URL to geonames dump (defaults to local cache candidates or geonames URL)
- `--output-dir` output directory for rendered map figure
