"""Anchor-set generation: centroids, edges, and texture crops."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from rich.console import Console

from .config import PipelineConfig
from .tabular import manifest_path, read_rows


@dataclass
class ManifestRow:
    image_id: str
    clean_path: str


def _load_manifest(config: PipelineConfig) -> list[ManifestRow]:
    rows_raw = read_rows(manifest_path(config.manifests_dir, config.use_parquet))
    rows = [ManifestRow(image_id=str(r["image_id"]), clean_path=str(r["clean_path"])) for r in rows_raw]
    if config.limit is not None:
        rows = rows[: config.limit]
    return rows


def _best_crop(gray: np.ndarray, crop_size: int) -> tuple[int, int, int]:
    h, w = gray.shape
    c = min(crop_size, h, w)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    best_score = -1.0
    best_xy = (0, 0)
    step = max(8, c // 8)
    for y in range(0, max(1, h - c + 1), step):
        for x in range(0, max(1, w - c + 1), step):
            score = float(mag[y : y + c, x : x + c].mean())
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return best_xy[0], best_xy[1], c


def run_anchors(config: PipelineConfig, force: bool = False, console: Console | None = None) -> dict[str, int | str]:
    rich_console = console or Console()

    style_vec = np.load(config.outputs_dir / "style_vec.npy")
    cluster_map = json.loads((config.outputs_dir / "clusters.json").read_text(encoding="utf-8"))
    manifest = _load_manifest(config)
    if style_vec.ndim != 2 or style_vec.shape[0] != len(manifest):
        raise ValueError("style_vec row count must match manifest")

    out_root = config.outputs_dir / "anchors"
    index_path = config.outputs_dir / "anchors_index.csv"
    if out_root.exists() and force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    image_ids = [m.image_id for m in manifest]
    cluster_ids = np.array([int(cluster_map.get(iid, -1)) for iid in image_ids], dtype=int)

    clusters = sorted(set(cluster_ids.tolist()))
    if not config.anchors_include_noise:
        clusters = [c for c in clusters if c != -1]

    rows_out: list[dict[str, str | int | float]] = []
    for cid in clusters:
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue

        vecs = style_vec[idx]
        mean = vecs.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(vecs - mean, axis=1)

        cent_local = np.argsort(dist)[: min(config.anchors_k_centroids, idx.size)]
        edge_local = np.argsort(dist)[::-1][: min(config.anchors_m_edges, idx.size)]

        cluster_dir = out_root / f"cluster_{cid}"
        cent_dir = cluster_dir / "centroids"
        edge_dir = cluster_dir / "edges"
        crop_dir = cluster_dir / "crops"
        for d in (cent_dir, edge_dir, crop_dir):
            d.mkdir(parents=True, exist_ok=True)

        selected = [(idx[i], "centroid", float(dist[i])) for i in cent_local] + [
            (idx[i], "edge", float(dist[i])) for i in edge_local
        ]

        for global_idx, role, distance in selected:
            m = manifest[global_idx]
            src = config.images_clean_dir / m.clean_path
            role_dir = cent_dir if role == "centroid" else edge_dir
            anchor_path = role_dir / f"{m.image_id}.jpg"
            shutil.copy2(src, anchor_path)

            crop_path = crop_dir / f"{m.image_id}.jpg"
            with Image.open(src) as im:
                rgb = im.convert("RGB")
                gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
                x, y, c = _best_crop(gray, config.anchor_crop_size)
                rgb.crop((x, y, x + c, y + c)).save(crop_path, format="JPEG", quality=95)

            rows_out.append(
                {
                    "image_id": m.image_id,
                    "cluster_id": cid,
                    "role": role,
                    "distance_to_mean": distance,
                    "anchor_path": str(anchor_path),
                    "crop_path": str(crop_path),
                }
            )

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "cluster_id", "role", "distance_to_mean", "anchor_path", "crop_path"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    rich_console.print(f"[bold green]Anchors complete[/bold green] rows={len(rows_out)}")
    return {"anchors_index": str(index_path), "rows": len(rows_out)}
