"""Image statistics extraction from cleaned images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import PipelineConfig
from .tabular import manifest_path, read_rows, stats_path, write_rows


@dataclass
class ManifestRow:
    image_id: str
    clean_path: str


FEATURE_COLUMNS = [
    "mean_luma",
    "std_luma",
    "edge_density",
    "ink_coverage",
    "entropy",
    "void_ratio",
    "center_of_mass_x",
    "center_of_mass_y",
    "radial_energy",
    "saturation_mean",
    "saturation_std",
    "corr_rg",
    "corr_rb",
    "corr_gb",
    "texture_laplacian_var",
]


def _load_manifest(config: PipelineConfig) -> tuple[list[ManifestRow], Path]:
    path = manifest_path(config.manifests_dir, config.use_parquet)
    rows_raw = read_rows(path)
    rows = [ManifestRow(image_id=str(row["image_id"]), clean_path=str(row["clean_path"])) for row in rows_raw]
    if config.limit is not None:
        rows = rows[: config.limit]
    return rows, path


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-12 or b_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _compute_features(
    bgr: np.ndarray,
    low_luma_threshold: int,
    high_luma_threshold: int,
    canny_low_threshold: int,
    canny_high_threshold: int,
) -> dict[str, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    mean_luma = float(gray_f.mean())
    std_luma = float(gray_f.std())

    edges = cv2.Canny(gray, canny_low_threshold, canny_high_threshold)
    edge_density = float((edges > 0).mean())

    ink_mask = gray < low_luma_threshold
    ink_coverage = float(ink_mask.mean())

    void_mask = gray > high_luma_threshold
    void_ratio = float(void_mask.mean())

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())

    ys, xs = np.where(ink_mask)
    if xs.size > 0:
        center_of_mass_x = float(xs.mean() / max(gray.shape[1] - 1, 1))
        center_of_mass_y = float(ys.mean() / max(gray.shape[0] - 1, 1))
    else:
        center_of_mass_x = float("nan")
        center_of_mass_y = float("nan")

    h, w = gray.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    max_radius = np.sqrt(cx**2 + cy**2)
    center_mask = dist <= (0.5 * max_radius)
    border_mask = dist >= (0.75 * max_radius)

    edge_binary = edges > 0
    center_energy = float(edge_binary[center_mask].mean()) if np.any(center_mask) else 0.0
    border_energy = float(edge_binary[border_mask].mean()) if np.any(border_mask) else 0.0
    radial_energy = float(center_energy / max(border_energy, 1e-12))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    saturation_mean = float(sat.mean())
    saturation_std = float(sat.std())

    b = bgr[..., 0].astype(np.float32).reshape(-1)
    g = bgr[..., 1].astype(np.float32).reshape(-1)
    r = bgr[..., 2].astype(np.float32).reshape(-1)
    corr_rg = _safe_corr(r, g)
    corr_rb = _safe_corr(r, b)
    corr_gb = _safe_corr(g, b)

    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    texture_laplacian_var = float(lap.var())

    return {
        "mean_luma": mean_luma,
        "std_luma": std_luma,
        "edge_density": edge_density,
        "ink_coverage": ink_coverage,
        "entropy": entropy,
        "void_ratio": void_ratio,
        "center_of_mass_x": center_of_mass_x,
        "center_of_mass_y": center_of_mass_y,
        "radial_energy": radial_energy,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "corr_rg": corr_rg,
        "corr_rb": corr_rb,
        "corr_gb": corr_gb,
        "texture_laplacian_var": texture_laplacian_var,
    }


def run_stats(config: PipelineConfig, console: Console | None = None) -> dict[str, int | str]:
    """Compute per-image handcrafted stats features and save csv/npy/meta outputs."""

    rich_console = console or Console()
    rows, resolved_manifest = _load_manifest(config)

    config.embeddings_dir.mkdir(parents=True, exist_ok=True)
    table_path = stats_path(config.embeddings_dir, config.use_parquet)
    npy_path = config.embeddings_dir / "stats.npy"
    meta_path = config.embeddings_dir / "stats_meta.json"

    matrix = np.full((len(rows), len(FEATURE_COLUMNS)), np.nan, dtype=np.float32)
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=rich_console,
    ) as progress:
        task = progress.add_task("Computing image stats", total=len(rows))
        for i, row in enumerate(rows):
            progress.update(task, advance=1)
            path = config.images_clean_dir / row.clean_path
            try:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("cv2.imread returned None")
                features = _compute_features(
                    bgr,
                    low_luma_threshold=config.ink_luma_threshold,
                    high_luma_threshold=config.void_luma_threshold,
                    canny_low_threshold=config.canny_low_threshold,
                    canny_high_threshold=config.canny_high_threshold,
                )
                matrix[i] = np.array([features[col] for col in FEATURE_COLUMNS], dtype=np.float32)
            except Exception as exc:
                failed += 1
                rich_console.log(f"[yellow]stats: skipping corrupt/unreadable image[/yellow] {path}: {exc}")

    table_rows: list[dict[str, float | str | None]] = []
    for i, row in enumerate(rows):
        payload: dict[str, float | str | None] = {"image_id": row.image_id}
        for j, col in enumerate(FEATURE_COLUMNS):
            value = matrix[i, j]
            payload[col] = None if np.isnan(value) else float(value)
        table_rows.append(payload)
    write_rows(table_path, table_rows, fieldnames=["image_id", *FEATURE_COLUMNS], use_parquet=config.use_parquet)

    np.save(npy_path, matrix)

    meta = {
        "feature_list": FEATURE_COLUMNS,
        "ink_luma_threshold": config.ink_luma_threshold,
        "void_luma_threshold": config.void_luma_threshold,
        "canny_low_threshold": config.canny_low_threshold,
        "canny_high_threshold": config.canny_high_threshold,
        "rows": len(rows),
        "failed": failed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(resolved_manifest),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    rich_console.print("[bold green]Stats complete[/bold green]")
    rich_console.print(f"Rows={len(rows)} Failed={failed} Features={len(FEATURE_COLUMNS)}")
    rich_console.print(f"Saved: {table_path}")
    rich_console.print(f"Saved: {npy_path}")
    rich_console.print(f"Saved: {meta_path}")

    return {
        "rows": len(rows),
        "failed": failed,
        "table": str(table_path),
        "npy": str(npy_path),
        "meta": str(meta_path),
    }
