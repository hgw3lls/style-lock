"""Anchor-set generation from clustered style vectors."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rich.console import Console

from .config import PipelineConfig


@dataclass
class ManifestRow:
    image_id: str
    clean_path: str


def _load_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as h:
        reader = csv.DictReader(h)
        for r in reader:
            rows.append(ManifestRow(image_id=r["image_id"], clean_path=r["clean_path"]))
    return rows


def _best_crop(gray: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
    h, w = gray.shape
    c = min(crop_size, h, w)
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    best_score = -1.0
    best = (0, 0, c, c)
    step = max(8, c // 8)
    for y in range(0, max(1, h - c + 1), step):
        for x in range(0, max(1, w - c + 1), step):
            score = float(mag[y : y + c, x : x + c].mean())
            if score > best_score:
                best_score = score
                best = (x, y, c, c)
    return best


def run_anchors(config: PipelineConfig, force: bool = False, console: Console | None = None) -> dict[str, int | str]:
    rich_console = console or Console()

    style_vec = np.load(config.outputs_dir / "style_vec.npy")
    cluster_map = json.loads((config.outputs_dir / "clusters.json").read_text(encoding="utf-8"))
    manifest = _load_manifest(config.manifests_dir / "images_clean.csv")

    if style_vec.ndim != 2 or style_vec.shape[0] != len(manifest):
        raise ValueError("style_vec row count must match manifest")

    out_root = config.outputs_dir / "anchors"
    index_path = config.outputs_dir / "anchors_index.csv"
    if out_root.exists() and force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    image_ids = [m.image_id for m in manifest]
    cluster_ids = np.array([int(cluster_map.get(iid, -1)) for iid in image_ids], dtype=int)

    unique_clusters = sorted(set(cluster_ids.tolist()))
    if not config.anchors_include_noise:
        unique_clusters = [c for c in unique_clusters if c != -1]

    rows_out: list[dict[str, str | int | float]] = []

    for cid in unique_clusters:
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue

        vecs = style_vec[idx]
        center = vecs.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(vecs - center, axis=1)

        k = min(config.anchors_k_centroids, idx.size)
        m = min(config.anchors_m_edges, idx.size)
        centroid_local = np.argsort(dist)[:k]
        edge_local = np.argsort(dist)[::-1][:m]

        cluster_dir = out_root / f"cluster_{cid}"
        cent_dir = cluster_dir / "centroids"
        edge_dir = cluster_dir / "edges"
        crop_dir = cluster_dir / "crops"
        cent_dir.mkdir(parents=True, exist_ok=True)
        edge_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)

        selected_global = [(idx[i], "centroid", float(dist[i])) for i in centroid_local] + [
            (idx[i], "edge", float(dist[i])) for i in edge_local
        ]

        for global_idx, role, distance in selected_global:
            item = manifest[global_idx]
            src = config.images_clean_dir / item.clean_path
            role_dir = cent_dir if role == "centroid" else edge_dir
            anchor_path = role_dir / f"{item.image_id}.jpg"
            shutil.copy2(src, anchor_path)

            crop_path = crop_dir / f"{item.image_id}.jpg"
            with Image.open(src) as img:
                rgb = img.convert("RGB")
                arr = np.array(rgb)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                x, y, c, _ = _best_crop(gray, config.anchor_crop_size)
                crop = rgb.crop((x, y, x + c, y + c))
                crop.save(crop_path, format="JPEG", quality=95)

            rows_out.append(
                {
                    "image_id": item.image_id,
                    "cluster_id": int(cid),
                    "role": role,
                    "distance": distance,
                    "crop_path": str(crop_path),
                }
            )

    with index_path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["image_id", "cluster_id", "role", "distance", "crop_path"])
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    rich_console.print(f"[bold green]Anchors complete[/bold green] rows={len(rows_out)}")
    return {"anchors_index": str(index_path), "rows": len(rows_out)}
