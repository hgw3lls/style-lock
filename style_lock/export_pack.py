"""Export pack builder for style_lock."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from .config import PipelineConfig


SPEC_TEMPLATE = """# STYLE_LOCK_SPEC

## 1) Non-negotiables (lock)
- Core style invariants that must remain unchanged:
  - Palette / tonal regime:
  - Stroke/mark family:
  - Spatial rhythm:
  - Structural motif:

## 2) Allowed variance
- Controlled variation knobs:
  - Subject matter range:
  - Secondary texture variation:
  - Degree of asymmetry:
  - Acceptable color drift:

## 3) Forbidden moves
- Explicitly disallowed outputs:
  - Avoid these compositional tropes:
  - Avoid these rendering artifacts:
  - Avoid these semantic cues:

## 4) Plate / overprint logic rules
- Layer and interaction rules:
  - Plate order:
  - Overprint/knockout behavior:
  - Misregistration tolerance:

## 5) Composition constraints (void/mass, reading path)
- Layout constraints:
  - Void-to-mass ratio target:
  - Focal path / reading direction:
  - Margin pressure:

## 6) Mark physics directives (wobble, pressure, bleed)
- Physical behavior directives:
  - Wobble character:
  - Pressure dynamics:
  - Bleed/spread limits:

## 7) Ontology rules (wrongness-without-reveal)
- Keep the latent logic coherent but unresolved:
  - What must feel “wrong”:
  - What must remain hidden:
  - What should be implied, not stated:

## 8) Prompt skeleton (fill-in-the-blanks)
Use this structure when prompting a generator:

"Create [SUBJECT] in the locked style with [NON-NEGOTIABLES].
Maintain [VOID/MASS RULE], [MARK PHYSICS], and [OVERPRINT LOGIC].
Allow variation only in [ALLOWED VARIANCE].
Avoid [FORBIDDEN MOVES].
Preserve the sense of [ONTOLOGY RULE]."
"""


README_TEMPLATE = """# style_lock export pack

This folder is a portable style pack for downstream image generation.

## Contents
- `anchors/`: representative anchor images by cluster
- `crops/`: texture closeups (high-gradient local crops)
- `cluster_summary.csv`: cluster-level metrics and ranking data
- `clusters.json`: `image_id -> cluster_id` assignments
- `resolved_config.yaml`: resolved config snapshot used for export
- `STYLE_LOCK_SPEC.md`: editable style-lock specification template

## How to use with an image generator
1. Upload `anchors/` and `crops/` as visual references.
2. Open `STYLE_LOCK_SPEC.md` and fill every placeholder section.
3. Provide the completed spec text as system/style guidance to your generator.
4. Start with low variation and increase only within the “Allowed variance” section.
5. Reject outputs violating “Non-negotiables” or any “Forbidden moves”.

## Recommended iteration loop
- Generate a small batch (8–16 images)
- Compare against centroid anchors first, then edge anchors
- Tighten spec directives when drift appears
"""


def _load_cluster_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster summary: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _to_float(value: str | None) -> float:
    if value in {None, "", "nan", "NaN"}:
        return float("nan")
    return float(value)


def _to_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value or default)
    except Exception:
        return default


def _select_clusters(rows: list[dict[str, Any]], top_n: int, rank_by: str) -> list[int]:
    filtered = [r for r in rows if _to_int(r.get("cluster_id"), -1) != -1]

    if rank_by == "avg_prob":
        filtered.sort(key=lambda r: _to_float(r.get("avg_prob")), reverse=True)
    else:
        filtered.sort(key=lambda r: _to_int(r.get("count"), 0), reverse=True)

    return [_to_int(r.get("cluster_id"), -1) for r in filtered[:top_n]]


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_export_pack(config: PipelineConfig, console: Console | None = None) -> dict[str, str | int]:
    """Build a portable style pack from clustering and anchor outputs."""

    rich_console = console or Console()

    export_dir = config.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    cluster_summary_path = config.outputs_dir / "cluster_summary.csv"
    clusters_json_path = config.outputs_dir / "clusters.json"
    anchors_root = config.outputs_dir / "anchors"

    summary_rows = _load_cluster_summary(cluster_summary_path)
    selected_cluster_ids = _select_clusters(
        summary_rows,
        top_n=config.export_top_n_clusters,
        rank_by=config.export_rank_by,
    )

    export_anchors_root = export_dir / "anchors"
    export_anchors_root.mkdir(parents=True, exist_ok=True)

    for cid in selected_cluster_ids:
        src_cluster_dir = anchors_root / f"cluster_{cid}"
        dst_cluster_dir = export_anchors_root / f"cluster_{cid}"
        _copy_tree_if_exists(src_cluster_dir, dst_cluster_dir)

    # Also copy all crops across all clusters into a flat crops folder.
    export_crops_root = export_dir / "crops"
    export_crops_root.mkdir(parents=True, exist_ok=True)
    if anchors_root.exists():
        for crop in anchors_root.glob("cluster_*/crops/*.jpg"):
            out_name = f"{crop.parent.parent.name}__{crop.name}"
            shutil.copy2(crop, export_crops_root / out_name)

    shutil.copy2(cluster_summary_path, export_dir / "cluster_summary.csv")
    if clusters_json_path.exists():
        shutil.copy2(clusters_json_path, export_dir / "clusters.json")

    resolved_config_path = export_dir / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)

    spec_path = export_dir / "STYLE_LOCK_SPEC.md"
    spec_path.write_text(SPEC_TEMPLATE, encoding="utf-8")

    readme_path = export_dir / "readme.md"
    readme_path.write_text(README_TEMPLATE, encoding="utf-8")

    meta_path = export_dir / "export_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_cluster_ids": selected_cluster_ids,
                "top_n": config.export_top_n_clusters,
                "rank_by": config.export_rank_by,
            },
            handle,
            indent=2,
        )

    rich_console.print("[bold green]Export pack complete[/bold green]")
    rich_console.print(f"Export directory: {export_dir}")
    rich_console.print(f"Selected clusters: {selected_cluster_ids}")

    return {
        "export_dir": str(export_dir),
        "selected_clusters": len(selected_cluster_ids),
        "spec": str(spec_path),
        "readme": str(readme_path),
    }
