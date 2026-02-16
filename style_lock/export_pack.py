"""Export style_lock_pack_v1 artifacts, style axes, and lock spec."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from .config import PipelineConfig


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(v: str | None, default: float = float("nan")) -> float:
    if v is None or v == "" or v.lower() == "nan":
        return default
    return float(v)


def _to_int(v: str | None, default: int = 0) -> int:
    try:
        return int(v or default)
    except Exception:
        return default


def _axis_label_density(edge_density: float, ink: float, void: float) -> str:
    score = edge_density + ink - void
    if score > 0.35:
        return "dense / ink-heavy"
    if score < -0.05:
        return "airy / void-forward"
    return "balanced mass-void"


def _axis_label_contrast(std_luma: float, entropy: float) -> str:
    score = std_luma / 64.0 + entropy / 8.0
    if score > 1.7:
        return "high-contrast / information-rich"
    if score < 1.2:
        return "soft-contrast / restrained"
    return "moderate contrast"


def _axis_label_center(com_x: float, com_y: float, radial: float) -> str:
    if abs(com_x - 0.5) + abs(com_y - 0.5) < 0.2 and radial > 1.0:
        return "center-anchored"
    if radial < 0.8:
        return "border-biased"
    return "off-center drift"


def _axis_label_texture(lap_var: float, entropy: float) -> str:
    if lap_var > 600 and entropy > 5.0:
        return "rough / noisy substrate"
    if lap_var < 250:
        return "smooth / controlled substrate"
    return "mid-grain texture"


def _axis_label_overprint(sat_mean: float, sat_std: float, corr_mean: float) -> str:
    if sat_mean < 35 and corr_mean > 0.9:
        return "limited plates / near-monochrome overprint"
    if sat_std > 40 and corr_mean < 0.8:
        return "plate separation visible / overprint interaction"
    return "moderate plate coupling"


def _build_style_axes(summary_rows: list[dict[str, str]], anchor_rows: list[dict[str, str]]) -> dict[str, Any]:
    centroid_ids: dict[int, list[str]] = defaultdict(list)
    edge_ids: dict[int, list[str]] = defaultdict(list)
    for row in anchor_rows:
        cid = _to_int(row.get("cluster_id"), -1)
        if row.get("role") == "centroid":
            centroid_ids[cid].append(str(row.get("image_id", "")))
        elif row.get("role") == "edge":
            edge_ids[cid].append(str(row.get("image_id", "")))

    clusters: dict[str, Any] = {}
    for row in summary_rows:
        cid = _to_int(row.get("cluster_id"), -1)
        if cid == -1:
            continue

        edge_density = _to_float(row.get("avg_edge_density"), 0.0)
        ink_coverage = _to_float(row.get("avg_ink_coverage"), 0.0)
        void_ratio = _to_float(row.get("avg_void_ratio"), 0.0)
        std_luma = _to_float(row.get("avg_std_luma"), 0.0)
        entropy = _to_float(row.get("avg_entropy"), 0.0)
        com_x = _to_float(row.get("avg_center_of_mass_x"), 0.5)
        com_y = _to_float(row.get("avg_center_of_mass_y"), 0.5)
        radial = _to_float(row.get("avg_radial_energy_ratio"), 1.0)
        lap_var = _to_float(row.get("avg_texture_laplacian_var"), 0.0)
        sat_mean = _to_float(row.get("avg_saturation_mean"), 0.0)
        sat_std = _to_float(row.get("avg_saturation_std"), 0.0)
        corr_mean = (
            _to_float(row.get("avg_corr_rg"), 0.0)
            + _to_float(row.get("avg_corr_rb"), 0.0)
            + _to_float(row.get("avg_corr_gb"), 0.0)
        ) / 3.0

        clusters[f"cluster_{cid}"] = {
            "density_profile": {
                "label": _axis_label_density(edge_density, ink_coverage, void_ratio),
                "edge_density": edge_density,
                "ink_coverage": ink_coverage,
                "void_ratio": void_ratio,
            },
            "contrast_profile": {
                "label": _axis_label_contrast(std_luma, entropy),
                "std_luma": std_luma,
                "entropy": entropy,
            },
            "center_gravity": {
                "label": _axis_label_center(com_x, com_y, radial),
                "center_of_mass_x": com_x,
                "center_of_mass_y": com_y,
                "radial_energy_ratio": radial,
            },
            "material_texture": {
                "label": _axis_label_texture(lap_var, entropy),
                "texture_laplacian_var": lap_var,
                "entropy": entropy,
            },
            "overprint_proxy": {
                "label": _axis_label_overprint(sat_mean, sat_std, corr_mean),
                "saturation_mean": sat_mean,
                "saturation_std": sat_std,
                "corr_mean": corr_mean,
            },
            "representative_ids": centroid_ids[cid],
            "boundary_ids": edge_ids[cid],
            "notes": (
                f"Cluster {cid} presents {_axis_label_density(edge_density, ink_coverage, void_ratio)} dynamics. "
                f"Contrast reads as {_axis_label_contrast(std_luma, entropy)} with {_axis_label_texture(lap_var, entropy)} behavior.\n"
                "Preserve mark pressure and overprint coupling from centroid anchors; use edge anchors as boundary guards."
            ),
        }

    return {"clusters": clusters}


def _build_spec(style_axes: dict[str, Any]) -> str:
    cluster_lines = []
    for cid in style_axes.get("clusters", {}).keys():
        cluster_lines.append(f"- Match mark behavior and plate logic to anchors in `{cid}`.")
    cluster_text = "\n".join(cluster_lines) if cluster_lines else "- No non-noise clusters found; inspect data quality before generation."

    return f"""# STYLE_LOCK_SPEC.md

## 1) Non-negotiables (LOCK)
- Preserve mark physics from anchor clusters exactly: wobble amplitude, pressure transitions, hatch density, and correction marks.
- Preserve material behavior: paper tooth visibility, ink bleed envelope, xerox/noise floor, and edge rag.
- Preserve palette discipline and value curve inferred from anchors; do not introduce unrelated hue families.
{cluster_text}

## 2) Allowed variance (SAFE MUTATIONS)
- Subject identity and iconography may change while retaining style physics.
- Composition may rotate/reposition primary mass if void-to-mass ratio remains within cluster tendencies.
- Minor plate drift and overlap changes are allowed within cluster overprint proxies.

## 3) Forbidden moves (STYLE BREAKERS)
- No glossy gradients, airbrush smoothing, or cinematic depth effects unless explicitly present in anchors.
- No clean vector-perfect contouring if anchors display hand-pressure variation or wobble.
- No palette expansion beyond observed cluster logic.

## 4) Plate / overprint logic rules
- Model output as plate-like passes; preserve overlap behavior and edge interactions seen in anchors.
- Allow misregistration only at subtle amplitudes consistent with edge anchors.
- Do not flatten overprint interactions into single blended gradients.

## 5) Composition constraints (void/mass, reading paths, density windows)
- Respect cluster density profile targets (edge density, ink coverage, void ratio).
- Maintain reading-path coherence (center gravity and radial energy tendencies).
- Keep negative space intentional; avoid fully saturated all-over fills unless anchors demand it.

## 6) Mark physics directives (wobble, pressure, visible corrections, ink behavior)
- Keep stroke start/end pressure, nib drag, and micro-jitter congruent with centroid anchors.
- Preserve visible corrections/overdraw where present.
- Simulate material transfer (bleed, dry-brush skip, toner noise) according to cluster texture profile.

## 7) Ontology / tone rules
- Keep output diagrammatic/score-like/painterly in the same mode as nearest anchor cluster.
- Maintain "wrongness-without-reveal": suggest latent system logic without explicit explanation.
- Tone must read as intentional artifact, not polished illustration.

## 8) Prompt skeleton template
- **SUBJECT (new content):** <what to depict>
- **COMPOSITION:** <mass/void target, reading path, center gravity>
- **STYLE LOCK DIRECTIVES:**
  - Match anchor cluster: <cluster_id>
  - Preserve mark wobble/pressure/hatch behavior
  - Preserve density and contrast profile
- **PALETTE / PLATE LOGIC:** <plate order, overprint behavior, misregistration tolerance>
- **HUMANIZER / MATERIAL BEHAVIOR:** <paper tooth, bleed, xerox noise, correction marks>
"""


def run_export_pack(config: PipelineConfig, console: Console | None = None) -> dict[str, str | int]:
    """Build style_lock_pack_v1 for upload to downstream image generators."""

    rich_console = console or Console()

    export_dir = config.export_dir
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    anchors_root = config.outputs_dir / "anchors"
    if anchors_root.exists():
        shutil.copytree(anchors_root, export_dir / "anchors")

    for fname in ["clusters.json", "cluster_summary.csv", "umap.png", "anchors_index.csv"]:
        src = config.outputs_dir / fname
        if src.exists():
            shutil.copy2(src, export_dir / fname)

    (export_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    summary_rows = _read_csv(config.outputs_dir / "cluster_summary.csv")
    anchor_rows = _read_csv(config.outputs_dir / "anchors_index.csv")
    style_axes = _build_style_axes(summary_rows, anchor_rows)
    (export_dir / "style_axes.yaml").write_text(yaml.safe_dump(style_axes, sort_keys=False), encoding="utf-8")

    (export_dir / "STYLE_LOCK_SPEC.md").write_text(_build_spec(style_axes), encoding="utf-8")

    (export_dir / "readme.md").write_text(
        "# style_lock_pack_v1\n\n"
        "Upload `anchors/` plus `STYLE_LOCK_SPEC.md` to your image generator/assistant to lock style behavior.\n\n"
        "Also inspect `style_axes.yaml`, `cluster_summary.csv`, and `umap.png` for QA before prompting.\n",
        encoding="utf-8",
    )

    rich_console.print("[bold green]Export pack complete[/bold green]")
    return {"export_dir": str(export_dir), "clusters": len(style_axes.get("clusters", {}))}
