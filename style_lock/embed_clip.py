"""CLIP embedding pipeline using open_clip_torch."""

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
        image_path = self.images_clean_dir / row.clean_path
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, row.image_id


def _load_manifest(manifest_path: Path) -> list[ManifestRow]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(ManifestRow(image_id=row["image_id"], clean_path=row["clean_path"]))
    return rows


def _resolve_manifest_path(config: PipelineConfig) -> Path:
    return config.manifests_dir / "images_clean.csv"


def run_embed_clip(config: PipelineConfig, force: bool = False, console: Console | None = None) -> dict[str, int | str]:
    """Create CLIP embeddings from cleaned images listed in the manifest."""

    rich_console = console or Console()

    try:
        import open_clip
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("open_clip_torch is required for clip embedding") from exc

    manifest_path = _resolve_manifest_path(config)
    rows = _load_manifest(manifest_path)
    num_images = len(rows)

    config.embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = config.embeddings_dir / "clip.npy"
    meta_path = config.embeddings_dir / "clip_meta.json"

    if embedding_path.exists() and not force:
        existing = np.load(embedding_path, mmap_mode="r")
        if existing.ndim == 2 and existing.shape[0] == num_images:
            rich_console.print(f"[yellow]Embedding exists and matches N={num_images}, skipping.[/yellow]")
            return {"status": "skipped", "count": num_images, "embeddings": str(embedding_path)}

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_arch,
        pretrained=config.clip_pretrained,
    )
    model.eval()

    device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = CleanImageDataset(config.images_clean_dir, rows, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    chunks: list[np.ndarray] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=rich_console,
    ) as progress:
        task = progress.add_task("Encoding images with CLIP", total=num_images)
        with torch.inference_mode():
            for batch, _image_ids in loader:
                batch = batch.to(device)
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                chunks.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
                progress.update(task, advance=len(batch))

    if chunks:
        embeddings = np.concatenate(chunks, axis=0)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)

    np.save(embedding_path, embeddings.astype(np.float32, copy=False))

    metadata = {
        "arch": config.clip_arch,
        "pretrained": config.clip_pretrained,
        "D": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "N": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
        "seed": config.seed,
        "device": str(device),
        "manifest": str(manifest_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    rich_console.print("[bold green]CLIP embedding complete[/bold green]")
    rich_console.print(f"Saved embeddings: {embedding_path} shape={tuple(embeddings.shape)}")
    rich_console.print(f"Saved metadata: {meta_path}")

    return {
        "status": "ok",
        "count": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "embeddings": str(embedding_path),
        "meta": str(meta_path),
    }
