# Style Lock Pack Compiler

A production-ready Python CLI that distills an image style dataset into an uploadable **anchor pack** plus textual style-lock spec for downstream image generators.

## Install

### pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### uv
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Configure
Copy and edit the example:

```bash
cp config.example.yaml config.yaml
```

`data/images_raw/` should contain your source images (recursive discovery supported).

## One-liner run

```bash
stylelock run-all --config config.yaml
```

This runs: preprocess -> embed (dino + clip) -> stats -> cluster -> anchors -> export.

## Output
The command creates `style_lock_pack_v1/` containing:
- `anchors/cluster_{id}/centroids/*.jpg`
- `anchors/cluster_{id}/edges/*.jpg`
- `anchors/cluster_{id}/crops/*.jpg`
- `STYLE_LOCK_SPEC.md`
- `style_axes.yaml`
- `clusters.json`
- `cluster_summary.csv`
- `umap.png`
- `anchors_index.csv`
- `config_resolved.yaml`
- `readme.md`

## Using the pack with an image generator/assistant
1. Upload the `anchors/` folder (and optionally `umap.png` for team review).
2. Paste `STYLE_LOCK_SPEC.md` into the assistant/system style instructions.
3. Keep prompts on-subject while preserving the style lock directives.
4. Use `style_axes.yaml` for machine-readable guardrails in tooling.

## Example config
See `config.example.yaml` for all defaults (model names, weights, thresholds, clustering, and export settings).

## License
MIT
