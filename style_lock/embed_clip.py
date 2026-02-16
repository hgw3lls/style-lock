"""OpenCLIP embedding pipeline."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import PipelineConfig
from .tabular import manifest_path, read_rows


@dataclass
class ManifestRow:
    image_id: str
    clean_path: str


def _load_manifest(config: PipelineConfig) -> tuple[list[ManifestRow], Path]:
    mpath = manifest_path(config.manifests_dir, config.use_parquet)
    rows_raw = read_rows(mpath)
    rows = [ManifestRow(image_id=str(r["image_id"]), clean_path=str(r["clean_path"])) for r in rows_raw]
    if config.limit is not None:
        rows = rows[: config.limit]
    return rows, mpath


def run_embed_clip(config: PipelineConfig, force: bool = False, console: Console | None = None) -> dict[str, int | str]:
    rich_console = console or Console()

    try:
        import open_clip
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("open_clip_torch + torch are required for CLIP embeddings") from exc

    rows, manifest_used = _load_manifest(config)
    n = len(rows)

    config.embeddings_dir.mkdir(parents=True, exist_ok=True)
    emb_path = config.embeddings_dir / "clip.npy"
    meta_path = config.embeddings_dir / "clip_meta.json"
    cache_root = config.embeddings_dir / "cache" / "clip"

    if emb_path.exists() and not force:
        arr = np.load(emb_path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[0] == n:
            return {"status": "skipped", "count": n, "embeddings": str(emb_path)}

    class CleanImageDataset(Dataset):
        def __init__(self, images_clean_dir: Path, selected: list[ManifestRow], transform) -> None:
            self.images_clean_dir = images_clean_dir
            self.selected = selected
            self.transform = transform

        def __len__(self) -> int:
            return len(self.selected)

        def __getitem__(self, idx: int):
            row = self.selected[idx]
            with Image.open(self.images_clean_dir / row.clean_path) as image:
                return self.transform(image.convert("RGB")), row.image_id

    model, _, preprocess = open_clip.create_model_and_transforms(config.clip_arch, pretrained=config.clip_pretrained)
    device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    row_to_cache: dict[str, Path] = {}
    in_memory_cache: dict[str, np.ndarray] = {}
    missing_rows: list[ManifestRow] = []
    if config.cache_embeddings:
        cache_root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            cp = cache_root / f"{row.image_id}.npy"
            row_to_cache[row.image_id] = cp
            if cp.exists() and not force:
                continue
            missing_rows.append(row)
    else:
        missing_rows = rows

    if missing_rows:
        loader = DataLoader(CleanImageDataset(config.images_clean_dir, missing_rows, preprocess), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=rich_console) as p:
            task = p.add_task("Encoding images with CLIP", total=len(missing_rows))
            with torch.inference_mode():
                for batch, image_ids in loader:
                    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if config.mixed_precision and device.type == "cuda" else nullcontext()
                    with amp_ctx:
                        feats = model.encode_image(batch.to(device))
                    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    feats_np = feats.detach().cpu().numpy().astype(np.float32, copy=False)
                    for iid, vec in zip(image_ids, feats_np):
                        if config.cache_embeddings:
                            np.save(row_to_cache[iid], vec)
                        else:
                            in_memory_cache[iid] = vec
                    p.update(task, advance=len(batch))

    vectors: list[np.ndarray] = []
    for row in rows:
        if config.cache_embeddings:
            cp = row_to_cache[row.image_id]
            if not cp.exists():
                raise RuntimeError(f"Missing CLIP cache for image_id={row.image_id}")
            vectors.append(np.load(cp).astype(np.float32, copy=False))
        else:
            vectors.append(in_memory_cache[row.image_id].astype(np.float32, copy=False))

    embeddings = np.stack(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)
    np.save(emb_path, embeddings)
    meta_path.write_text(json.dumps({"arch": config.clip_arch, "pretrained": config.clip_pretrained, "N": int(embeddings.shape[0]), "D": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0, "seed": config.seed, "device": str(device), "manifest": str(manifest_used), "created_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    return {"status": "ok", "count": int(embeddings.shape[0]), "embeddings": str(emb_path), "meta": str(meta_path)}
