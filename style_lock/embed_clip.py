"""CLIP embedding pipeline using open_clip_torch."""

from __future__ import annotations

import json
from contextlib import nullcontext
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
from .tabular import manifest_path, read_rows


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


def _load_manifest(config: PipelineConfig) -> tuple[list[ManifestRow], Path]:
    path = manifest_path(config.manifests_dir, config.use_parquet)
    rows_raw = read_rows(path)
    rows = [ManifestRow(image_id=str(r["image_id"]), clean_path=str(r["clean_path"])) for r in rows_raw]
    if config.limit is not None:
        rows = rows[: config.limit]
    return rows, path


def _cache_dir(config: PipelineConfig) -> Path:
    return config.embeddings_dir / "cache" / "clip"


def run_embed_clip(config: PipelineConfig, force: bool = False, console: Console | None = None) -> dict[str, int | str]:
    """Create CLIP embeddings from cleaned images listed in the manifest."""

    rich_console = console or Console()

    try:
        import open_clip
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("open_clip_torch is required for clip embedding") from exc

    rows, resolved_manifest_path = _load_manifest(config)
    num_images = len(rows)

    config.embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = config.embeddings_dir / "clip.npy"
    meta_path = config.embeddings_dir / "clip_meta.json"
    cache_root = _cache_dir(config)

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

    dim_cache: int | None = None
    row_to_cache: dict[str, Path] = {}
    in_memory_cache: dict[str, np.ndarray] = {}
    missing_rows: list[ManifestRow] = []
    if config.cache_embeddings:
        cache_root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            cache_path = cache_root / f"{row.image_id}.npy"
            row_to_cache[row.image_id] = cache_path
            if cache_path.exists() and not force:
                arr = np.load(cache_path)
                if arr.ndim == 1:
                    dim_cache = arr.shape[0]
                    continue
            missing_rows.append(row)
    else:
        missing_rows = rows

    if missing_rows:
        dataset = CleanImageDataset(config.images_clean_dir, missing_rows, preprocess)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=rich_console,
        ) as progress:
            task = progress.add_task("Encoding images with CLIP", total=len(missing_rows))
            with torch.inference_mode():
                for batch, image_ids in loader:
                    batch = batch.to(device)
                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if config.mixed_precision and device.type == "cuda"
                        else nullcontext()
                    )
                    with amp_ctx:
                        features = model.encode_image(batch)
                    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    features_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
                    for iid, vec in zip(image_ids, features_np):
                        if config.cache_embeddings:
                            np.save(row_to_cache[iid], vec)
                        else:
                            in_memory_cache[iid] = vec
                    progress.update(task, advance=len(batch))
                    if dim_cache is None and features_np.size:
                        dim_cache = int(features_np.shape[1])

    embeddings: np.ndarray
    if rows:
        vectors: list[np.ndarray] = []
        for row in rows:
            if config.cache_embeddings:
                cache_path = row_to_cache[row.image_id]
                if not cache_path.exists():
                    raise RuntimeError(f"Missing CLIP cache for image_id={row.image_id}")
                vectors.append(np.load(cache_path).astype(np.float32, copy=False))
            else:
                vec = in_memory_cache.get(row.image_id)
                if vec is None:
                    raise RuntimeError(f"Missing CLIP embedding for image_id={row.image_id}")
                vectors.append(vec.astype(np.float32, copy=False))
        embeddings = np.stack(vectors, axis=0)
    else:
        embeddings = np.empty((0, dim_cache or 0), dtype=np.float32)

    np.save(embedding_path, embeddings.astype(np.float32, copy=False))

    metadata = {
        "arch": config.clip_arch,
        "pretrained": config.clip_pretrained,
        "D": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
        "N": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
        "seed": config.seed,
        "device": str(device),
        "manifest": str(resolved_manifest_path),
        "mixed_precision": bool(config.mixed_precision and device.type == "cuda"),
        "cache_embeddings": config.cache_embeddings,
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
