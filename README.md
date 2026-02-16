# style_lock

A production-ready scaffold for deterministic style pipeline workflows.

## Features

- Python 3.10+
- CLI via Typer: `stylelock`
- Rich logging/progress output
- Pydantic config validation
- Deterministic seeds (`random`, `numpy`, optional `torch`)
- Optional GPU support (`--device cpu|cuda`)
- Image preprocessing with resize, RGB conversion, JPEG normalization, and perceptual-hash dedupe
- CLIP embedding (`stylelock embed --model clip`) via `open_clip_torch`
- Handcrafted image statistics extraction (`stylelock stats`) via OpenCV
- Optional parquet manifests/stats tables (`--use-parquet`)
- Fast iteration limit across stages (`--limit N`)
- Optional mixed precision embedding on CUDA (`--mixed-precision`)
- Optional per-image embedding cache (`embeddings/cache/*`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

## One command pipeline

```bash
stylelock run-all --config config.yaml --device cpu --force
```

This executes: preprocess -> embed dino -> embed clip -> stats -> cluster -> anchors -> export.
Use `--limit N` for quick iteration, `--use-parquet` for parquet tabular IO, and `--mixed-precision` on CUDA for faster embeddings.
Each stage skips existing outputs unless `--force` is provided.


1. Copy the example config:

```bash
cp config.example.yaml config.yaml
```

2. Run preprocess:

```bash
stylelock preprocess --config config.yaml
```

3. Run CLIP embedding:

```bash
stylelock embed --config config.yaml --model clip
```

4. Override CLIP model settings:

```bash
stylelock embed \
  --config config.yaml \
  --model clip \
  --clip-arch ViT-B-32 \
  --clip-pretrained laion2b_s34b_b79k \
  --force
```

5. Run stats extraction:

```bash
stylelock stats --config config.yaml
```

6. Run clustering:

```bash
stylelock cluster --config config.yaml
```

7. Build export pack:

```bash
stylelock export --config config.yaml
```

8. Run the full pipeline:

```bash
stylelock run-all --config config.yaml --model clip
```

### Preprocess outputs

- `manifests/images_clean.csv` (default) or `manifests/images_clean.parquet` (`--use-parquet`) with columns:
  - `image_id`
  - `src_path`
  - `clean_path`
  - `width`
  - `height`
  - `phash`
  - `bytes`

### Embed (CLIP) outputs

- `embeddings/clip.npy` as float32 shape `[N, D]`
- `embeddings/clip_meta.json` with `arch`, `pretrained`, `D`, `seed`, `device`, mixed precision/cache flags, and metadata
- Resume behavior: if existing embedding has matching `N`, command skips unless `--force`

### Stats outputs

- `embeddings/stats.csv` (default) or `embeddings/stats.parquet` (`--use-parquet`) with `image_id` and scalar feature columns
- `embeddings/stats.npy` numeric matrix `[N, F]`
- `embeddings/stats_meta.json` with thresholds and feature list

### Cluster outputs

- `outputs/style_vec.npy` final weighted concatenated style vectors
- `outputs/clusters.json` mapping `image_id -> cluster_id`
- `outputs/cluster_summary.csv` with count/probability and summary norms per cluster
- `outputs/umap_2d.npy` 2D UMAP coordinates
- `outputs/umap.png` UMAP scatter colored by cluster labels (`-1` is noise)

### Export outputs

- `${export_dir}/anchors/` selected cluster anchor folders
- `${export_dir}/crops/` all crop images flattened for quick upload
- `${export_dir}/cluster_summary.csv`, `${export_dir}/clusters.json`
- `${export_dir}/resolved_config.yaml` config snapshot
- `${export_dir}/STYLE_LOCK_SPEC.md` scaffold with required lock sections
- `${export_dir}/readme.md` upload and prompt usage instructions

## Commands

- `preprocess`
- `embed`
- `stats`
- `cluster`
- `anchors`
- `export`
- `run-all`

## License

MIT
