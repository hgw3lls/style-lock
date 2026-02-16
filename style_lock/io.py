"""Input/output helpers for style_lock."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def ensure_output_dir(path: Path) -> Path:
    """Create output directory if missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_stage_report(output_dir: Path, stage: str, payload: dict[str, Any]) -> Path:
    """Write a stage report YAML file for pipeline bookkeeping."""

    ensure_output_dir(output_dir)
    report_path = output_dir / f"{stage}.report.yaml"
    report = {
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    return report_path
