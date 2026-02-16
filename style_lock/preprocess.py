"""Image preprocessing pipeline for style_lock."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import imagehash
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import PipelineConfig
from .tabular import manifest_path, write_rows

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}


@dataclass
class CandidateImage:
    src_abs: Path
    src_rel: Path
    clean_rel: Path
    width: int
    height: int
    phash: imagehash.ImageHash
    jpeg_bytes: bytes

    @property
    def num_bytes(self) -> int:
        return len(self.jpeg_bytes)

    @property
    def resolution(self) -> int:
        return self.width * self.height


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest == max_side:
        return image

    ratio = max_side / float(longest)
    new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _stable_image_id(relative_path: Path) -> str:
    return hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:16]


def _discover_images(images_raw_dir: Path, limit: int | None = None) -> list[Path]:
    paths = [
        path
        for path in images_raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths = sorted(paths)
    if limit is not None:
        return paths[:limit]
    return paths


def _load_candidates(config: PipelineConfig, console: Console) -> tuple[list[CandidateImage], int]:
    image_paths = _discover_images(config.images_raw_dir, limit=config.limit)
    candidates: list[CandidateImage] = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading and normalizing images", total=len(image_paths))
        for path in image_paths:
            progress.update(task, advance=1)
            try:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    normalized = _resize_to_max_side(rgb, config.max_side)
                    width, height = normalized.size

                    buf = BytesIO()
                    normalized.save(
                        buf,
                        format="JPEG",
                        quality=config.jpeg_quality,
                        optimize=True,
                    )
                    jpeg_bytes = buf.getvalue()
                    phash = imagehash.phash(normalized)
            except Exception as exc:
                console.log(f"[yellow]Skipping unreadable image[/yellow] {path}: {exc}")
                failed += 1
                continue

            clean_rel = path.relative_to(config.images_raw_dir).with_suffix(".jpg")
            candidates.append(
                CandidateImage(
                    src_abs=path,
                    src_rel=path.relative_to(config.images_raw_dir),
                    clean_rel=clean_rel,
                    width=width,
                    height=height,
                    phash=phash,
                    jpeg_bytes=jpeg_bytes,
                )
            )

    return candidates, failed


def _select_unique(candidates: list[CandidateImage], threshold: int) -> tuple[list[CandidateImage], int]:
    if not candidates:
        return [], 0

    uf = UnionFind(len(candidates))
    for idx_a in range(len(candidates)):
        for idx_b in range(idx_a + 1, len(candidates)):
            if candidates[idx_a].phash - candidates[idx_b].phash <= threshold:
                uf.union(idx_a, idx_b)

    groups: dict[int, list[int]] = {}
    for idx in range(len(candidates)):
        groups.setdefault(uf.find(idx), []).append(idx)

    unique: list[CandidateImage] = []
    duplicates = 0

    for members in groups.values():
        best_idx = sorted(
            members,
            key=lambda i: (
                -candidates[i].resolution,
                -candidates[i].num_bytes,
                candidates[i].src_rel.as_posix(),
            ),
        )[0]
        unique.append(candidates[best_idx])
        duplicates += max(0, len(members) - 1)

    unique.sort(key=lambda c: c.src_rel.as_posix())
    return unique, duplicates


def _write_outputs(config: PipelineConfig, unique_candidates: list[CandidateImage]) -> Path:
    config.images_clean_dir.mkdir(parents=True, exist_ok=True)
    config.manifests_dir.mkdir(parents=True, exist_ok=True)

    for candidate in unique_candidates:
        out_path = config.images_clean_dir / candidate.clean_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(candidate.jpeg_bytes)

    out_path = manifest_path(config.manifests_dir, config.use_parquet)
    rows: list[dict[str, object]] = []
    for candidate in unique_candidates:
        rows.append(
            {
                "image_id": _stable_image_id(candidate.src_rel),
                "src_path": candidate.src_rel.as_posix(),
                "clean_path": candidate.clean_rel.as_posix(),
                "width": candidate.width,
                "height": candidate.height,
                "phash": str(candidate.phash),
                "bytes": candidate.num_bytes,
            }
        )
    write_rows(
        out_path,
        rows,
        fieldnames=["image_id", "src_path", "clean_path", "width", "height", "phash", "bytes"],
        use_parquet=config.use_parquet,
    )

    return out_path


def run_preprocess(config: PipelineConfig, console: Console | None = None) -> dict[str, int | str]:
    """Run preprocessing + near-duplicate removal and emit a manifest table."""

    rich_console = console or Console()

    candidates, failed = _load_candidates(config, rich_console)
    unique_candidates, duplicates = _select_unique(candidates, config.dedupe_threshold)
    manifest_path = _write_outputs(config, unique_candidates)

    stats = {
        "found": len(candidates) + failed,
        "processed": len(candidates),
        "failed": failed,
        "duplicates_removed": duplicates,
        "kept": len(unique_candidates),
        "manifest": str(manifest_path),
    }

    rich_console.print("[bold green]Preprocess complete[/bold green]")
    rich_console.print(
        f"Found={stats['found']} Processed={stats['processed']} Kept={stats['kept']} "
        f"DuplicatesRemoved={stats['duplicates_removed']} Failed={stats['failed']}"
    )
    rich_console.print(f"Manifest: {manifest_path}")

    return stats
