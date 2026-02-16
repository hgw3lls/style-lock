"""Shared runtime utilities."""

from __future__ import annotations

import random
from typing import Literal

import numpy as np

DeviceType = Literal["cpu", "cuda"]


def detect_default_device() -> DeviceType:
    """Auto-detect default device based on torch availability and CUDA."""

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch (if installed) for determinism."""

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch is optional; no-op when unavailable
        pass


def coerce_device(requested: str | None) -> DeviceType:
    """Resolve user-requested device or auto-detect when not provided."""

    if requested is None:
        return detect_default_device()

    normalized = requested.lower().strip()
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("Device must be one of: cpu, cuda")

    if normalized == "cuda" and detect_default_device() == "cpu":
        return "cpu"

    return normalized  # type: ignore[return-value]
