"""Style vector construction, clustering, and inspection outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from rich.console import Console
from sklearn.decomposition import PCA

from .config import PipelineConfig
from .stats import FEATURE_COLUMNS
from .tabular import manifest_path, read_rows


@dataclass
class ManifestRow:
    image_id: str


def _load_manifest_rows(config: PipelineConfig) -> list[ManifestRow]:
    path = manifest_path(config.manifests_dir, config.use_parquet)
    rows_raw = read_rows(path)
    rows = [ManifestRow(image_id=str(row["image_id"])) for row in rows_raw]
    if config.limit is not None:
        rows = rows[: config.limit]
    return rows


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return x / norms


def _zscore_cols(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (x - mean) / std


def _load_required_arrays(config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ManifestRow]]:
    dino = np.load(config.embeddings_dir / "dino.npy").astype(np.float32)
    clip = np.load(config.embeddings_dir / "clip.npy").astype(np.float32)
    stats = np.load(config.embeddings_dir / "stats.npy").astype(np.float32)
    manifest_rows = _load_manifest_rows(config)

    if dino.ndim != 2 or clip.ndim != 2 or stats.ndim != 2:
        raise ValueError("Expected dino.npy, clip.npy, stats.npy to all be rank-2 arrays")

    n = dino.shape[0]
    if clip.shape[0] != n or stats.shape[0] != n or len(manifest_rows) != n:
        raise ValueError(
            f"Input row mismatch: dino={dino.shape[0]}, clip={clip.shape[0]}, stats={stats.shape[0]}, manifest={len(manifest_rows)}"
        )

    return dino, clip, stats, manifest_rows


def run_cluster(config: PipelineConfig, console: Console | None = None) -> dict[str, int | str]:
    """Combine blocks into STYLE_VEC, run HDBSCAN, and save diagnostics."""

    rich_console = console or Console()
    dino, clip, stats, manifest_rows = _load_required_arrays(config)
    n = dino.shape[0]

    dino_block = _l2_normalize_rows(dino)
    clip_block = _l2_normalize_rows(clip)
    stats_block = np.nan_to_num(_zscore_cols(stats), nan=0.0, posinf=0.0, neginf=0.0)

    style_vec = np.concatenate(
        [dino_block * config.w_dino, stats_block * config.w_stats, clip_block * config.w_clip],
        axis=1,
    ).astype(np.float32)

    style_for_cluster = style_vec
    if config.cluster_use_pca and style_vec.shape[1] > config.cluster_pca_dim:
        pca = PCA(n_components=config.cluster_pca_dim, random_state=config.seed)
        style_for_cluster = pca.fit_transform(style_vec).astype(np.float32)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(style_for_cluster)
    probs = clusterer.probabilities_.astype(np.float32)

    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.outputs_dir / "style_vec.npy", style_vec)
    np.save(config.outputs_dir / "umap_2d.npy", umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=config.seed).fit_transform(style_for_cluster).astype(np.float32))

    image_to_cluster = {row.image_id: int(label) for row, label in zip(manifest_rows, labels)}
    (config.outputs_dir / "clusters.json").write_text(json.dumps(image_to_cluster, indent=2), encoding="utf-8")

    summary_cols = ["cluster_id", "count", "avg_prob"] + [f"avg_{c}" for c in FEATURE_COLUMNS]
    with (config.outputs_dir / "cluster_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_cols)
        writer.writeheader()
        for label in sorted(set(labels.tolist())):
            idx = labels == label
            row: dict[str, float | int] = {
                "cluster_id": int(label),
                "count": int(idx.sum()),
                "avg_prob": float(probs[idx].mean()) if np.any(idx) else float("nan"),
            }
            for f_idx, name in enumerate(FEATURE_COLUMNS):
                row[f"avg_{name}"] = float(np.nanmean(stats[idx, f_idx])) if np.any(idx) else float("nan")
            writer.writerow(row)

    umap_2d = np.load(config.outputs_dir / "umap_2d.npy")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels, s=9)
    ax.set_title("Style UMAP")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.colorbar(sc, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(config.outputs_dir / "umap.png", dpi=150)
    plt.close(fig)

    (config.outputs_dir / "cluster_meta.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "seed": config.seed,
                "weights": {"w_dino": config.w_dino, "w_stats": config.w_stats, "w_clip": config.w_clip},
                "cluster_use_pca": config.cluster_use_pca,
                "cluster_pca_dim": config.cluster_pca_dim,
                "hdbscan_min_cluster_size": config.hdbscan_min_cluster_size,
                "hdbscan_min_samples": config.hdbscan_min_samples,
                "n": n,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rich_console.print("[bold green]Cluster complete[/bold green]")
    return {
        "n": n,
        "style_vec": str(config.outputs_dir / "style_vec.npy"),
        "clusters_json": str(config.outputs_dir / "clusters.json"),
        "cluster_summary": str(config.outputs_dir / "cluster_summary.csv"),
        "umap": str(config.outputs_dir / "umap.png"),
    }
