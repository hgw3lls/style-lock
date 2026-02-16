"""Configuration models and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


DeviceType = Literal["cpu", "cuda"]


class PipelineConfig(BaseModel):
    """Runtime configuration for style_lock commands."""

    model_config = ConfigDict(extra="allow")

    # Global/runtime
    seed: int = Field(default=42)
    device: DeviceType = Field(default="cpu")
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)

    # Preprocess
    images_raw_dir: Path = Field(default=Path("data/images_raw"))
    images_clean_dir: Path = Field(default=Path("data/images_clean"))
    manifests_dir: Path = Field(default=Path("manifests"))
    max_side: int = Field(default=1024, ge=64)
    jpeg_quality: int = Field(default=92, ge=1, le=100)
    dedupe_threshold: int = Field(default=6, ge=0, le=64)
    limit: int | None = Field(default=None, ge=1)
    use_parquet: bool = Field(default=False)

    # Embedding
    embeddings_dir: Path = Field(default=Path("embeddings"))
    clip_arch: str = Field(default="ViT-B-32")
    clip_pretrained: str = Field(default="laion2b_s34b_b79k")
    dino_model_name: str = Field(default="timm/vit_base_patch14_dinov2.lvd142m")
    mixed_precision: bool = Field(default=False)
    cache_embeddings: bool = Field(default=True)

    # Stats
    ink_luma_threshold: int = Field(default=64, ge=0, le=255)
    void_luma_threshold: int = Field(default=224, ge=0, le=255)
    canny_low_threshold: int = Field(default=100, ge=0, le=255)
    canny_high_threshold: int = Field(default=200, ge=0, le=255)

    # Cluster
    outputs_dir: Path = Field(default=Path("outputs"))
    w_dino: float = Field(default=0.65)
    w_stats: float = Field(default=0.30)
    w_clip: float = Field(default=0.05)
    cluster_use_pca: bool = Field(default=True)
    cluster_pca_dim: int = Field(default=128, ge=2)
    hdbscan_min_cluster_size: int = Field(default=15, ge=2)
    hdbscan_min_samples: int | None = Field(default=None, ge=1)

    # Export
    export_dir: Path = Field(default=Path("style_lock_pack_v1"))
    export_top_n_clusters: int = Field(default=7, ge=1)
    export_rank_by: Literal["size", "avg_prob"] = Field(default="size")

    # Anchors
    anchors_include_noise: bool = Field(default=False)
    anchors_k_centroids: int = Field(default=3, ge=1)
    anchors_m_edges: int = Field(default=2, ge=1)
    anchor_crop_size: int = Field(default=384, ge=64)


def _set_nested_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = target
    for key in keys[:-1]:
        current = cursor.get(key)
        if not isinstance(current, dict):
            current = {}
            cursor[key] = current
        cursor = current
    cursor[keys[-1]] = value


def load_config(config_path: Path, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """Load YAML config and apply dotted-key overrides before validation."""

    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config root must be a mapping, got: {type(loaded)!r}")
            raw = loaded

    for key, value in (overrides or {}).items():
        if value is not None:
            _set_nested_value(raw, key, value)

    try:
        return PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
