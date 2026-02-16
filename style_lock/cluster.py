"""Style vector construction and HDBSCAN clustering."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from rich.console import Console
from sklearn.decomposition import PCA

from .config import PipelineConfig


@dataclass
class ManifestRow:
    image_id: str


def _load_manifest_rows(manifest_path: Path) -> list[ManifestRow]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(ManifestRow(image_id=row["image_id"]))
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
    dino_path = config.embeddings_dir / "dino.npy"
    clip_path = config.embeddings_dir / "clip.npy"
    stats_path = config.embeddings_dir / "stats.npy"
    manifest_path = config.manifests_dir / "images_clean.csv"

    dino = np.load(dino_path)
    clip = np.load(clip_path)
    stats = np.load(stats_path)
    manifest_rows = _load_manifest_rows(manifest_path)

    if dino.ndim != 2 or clip.ndim != 2 or stats.ndim != 2:
        raise ValueError("Expected dino.npy, clip.npy, stats.npy to all be rank-2 arrays")

    n = dino.shape[0]
    if clip.shape[0] != n or stats.shape[0] != n or len(manifest_rows) != n:
        raise ValueError(
            f"Input row mismatch: dino={dino.shape[0]}, clip={clip.shape[0]}, stats={stats.shape[0]}, manifest={len(manifest_rows)}"
        )

    return dino.astype(np.float32), clip.astype(np.float32), stats.astype(np.float32), manifest_rows


def run_cluster(config: PipelineConfig, console: Console | None = None) -> dict[str, int | str]:
    """Combine embedding blocks into style vectors and cluster with HDBSCAN."""

    rich_console = console or Console()

    dino, clip, stats, manifest_rows = _load_required_arrays(config)
    n = dino.shape[0]

    dino_block = _l2_normalize_rows(dino)
    clip_block = _l2_normalize_rows(clip)
    stats_block = _zscore_cols(stats)
    stats_block = np.nan_to_num(stats_block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    style_vec = np.concatenate(
        [
            dino_block * config.w_dino,
            stats_block * config.w_stats,
            clip_block * config.w_clip,
        ],
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
    probabilities = clusterer.probabilities_.astype(np.float32)

    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    clusters_json = config.outputs_dir / "clusters.json"
    summary_csv = config.outputs_dir / "cluster_summary.csv"
    style_vec_npy = config.outputs_dir / "style_vec.npy"
    umap_2d_npy = config.outputs_dir / "umap_2d.npy"
    umap_png = config.outputs_dir / "umap.png"

    np.save(style_vec_npy, style_vec)

    image_to_cluster = {row.image_id: int(label) for row, label in zip(manifest_rows, labels)}
    with clusters_json.open("w", encoding="utf-8") as handle:
        json.dump(image_to_cluster, handle, indent=2)

    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id",
                "count",
                "avg_prob",
                "avg_style_norm",
                "avg_dino_norm",
                "avg_stats_norm",
                "avg_clip_norm",
            ],
        )
        writer.writeheader()
        for label in unique_labels:
            idx = labels == label
            writer.writerow(
                {
                    "cluster_id": int(label),
                    "count": int(idx.sum()),
                    "avg_prob": float(probabilities[idx].mean()) if np.any(idx) else float("nan"),
                    "avg_style_norm": float(np.linalg.norm(style_vec[idx], axis=1).mean()) if np.any(idx) else float("nan"),
                    "avg_dino_norm": float(np.linalg.norm(dino_block[idx], axis=1).mean()) if np.any(idx) else float("nan"),
                    "avg_stats_norm": float(np.linalg.norm(stats_block[idx], axis=1).mean()) if np.any(idx) else float("nan"),
                    "avg_clip_norm": float(np.linalg.norm(clip_block[idx], axis=1).mean()) if np.any(idx) else float("nan"),
                }
            )

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=config.seed)
    umap_2d = reducer.fit_transform(style_for_cluster).astype(np.float32)
    np.save(umap_2d_npy, umap_2d)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels, s=8)
    ax.set_title("Style UMAP (cluster labels)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.colorbar(scatter, ax=ax, label="Cluster label")
    fig.tight_layout()
    fig.savefig(umap_png, dpi=150)
    plt.close(fig)

    rich_console.print("[bold green]Cluster complete[/bold green]")
    rich_console.print(f"N={n} clusters={len([x for x in unique_labels if x != -1])} noise={(labels == -1).sum()}")
    rich_console.print(f"Saved: {clusters_json}")
    rich_console.print(f"Saved: {summary_csv}")
    rich_console.print(f"Saved: {style_vec_npy}")
    rich_console.print(f"Saved: {umap_2d_npy}")
    rich_console.print(f"Saved: {umap_png}")

    meta_path = config.outputs_dir / "cluster_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n": n,
                "weights": {"w_dino": config.w_dino, "w_stats": config.w_stats, "w_clip": config.w_clip},
                "cluster_use_pca": config.cluster_use_pca,
                "cluster_pca_dim": config.cluster_pca_dim,
                "hdbscan_min_cluster_size": config.hdbscan_min_cluster_size,
                "hdbscan_min_samples": config.hdbscan_min_samples,
                "seed": config.seed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
        )

    return {
        "n": n,
        "clusters_json": str(clusters_json),
        "summary_csv": str(summary_csv),
        "style_vec": str(style_vec_npy),
        "umap_2d": str(umap_2d_npy),
        "umap_png": str(umap_png),
    }
