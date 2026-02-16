"""DINOv2 embedding pipeline using timm."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from torch.utils.data import DataLoader, Dataset

from .config import PipelineConfig


@dataclass
class ManifestRow:
    image_id: str
    clean_path: str


class CleanImageDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, images_clean_dir: Path, rows: list[ManifestRow], transform) -> None:
        self.images_clean_dir = images_clean_dir
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.rows[idx]
        path = self.images_clean_dir / row.clean_path
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row.image_id


def _load_manifest(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(ManifestRow(image_id=row["image_id"], clean_path=row["clean_path"]))
    return rows


def run_embed_dino(config: PipelineConfig, force: bool = False, console: Console | None = None) -> dict[str, int | str]:
    rich_console = console or Console()

    import timm
    from timm.data import create_transform, resolve_data_config

    manifest_path = config.manifests_dir / "images_clean.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows = _load_manifest(manifest_path)
    n = len(rows)

    config.embeddings_dir.mkdir(parents=True, exist_ok=True)
    emb_path = config.embeddings_dir / "dino.npy"
    meta_path = config.embeddings_dir / "dino_meta.json"

    if emb_path.exists() and not force:
        arr = np.load(emb_path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[0] == n:
            rich_console.print(f"[yellow]DINO embedding exists for N={n}; skipping.[/yellow]")
            return {"status": "skipped", "count": n, "embeddings": str(emb_path)}

    device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = timm.create_model(config.dino_model_name, pretrained=True, num_classes=0)
    model.eval().to(device)

    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg, is_training=False)

    ds = CleanImageDataset(config.images_clean_dir, rows, transform)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    chunks: list[np.ndarray] = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=rich_console) as p:
        task = p.add_task("Encoding images with DINO", total=n)
        with torch.inference_mode():
            for batch, _ in loader:
                feats = model(batch.to(device))
                chunks.append(feats.detach().cpu().numpy().astype(np.float32, copy=False))
                p.update(task, advance=len(batch))

    embeddings = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)
    np.save(emb_path, embeddings)

    meta = {
        "model_name": config.dino_model_name,
        "D": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "N": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
        "preprocess": data_cfg,
        "date": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "device": str(device),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"status": "ok", "count": int(embeddings.shape[0]), "embeddings": str(emb_path), "meta": str(meta_path)}
