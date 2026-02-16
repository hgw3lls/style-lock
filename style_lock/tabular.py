"""Tabular IO helpers for manifests/stats with optional parquet support."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def manifest_path(manifests_dir: Path, use_parquet: bool) -> Path:
    return manifests_dir / ("images_clean.parquet" if use_parquet else "images_clean.csv")


def stats_path(embeddings_dir: Path, use_parquet: bool) -> Path:
    return embeddings_dir / ("stats.parquet" if use_parquet else "stats.csv")


def _ensure_parquet_backend() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Parquet output requested but pyarrow is unavailable. Install 'pyarrow' to use --use-parquet."
        ) from exc


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], use_parquet: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_parquet:
        _ensure_parquet_backend()
        import pyarrow as pa
        import pyarrow.parquet as pq

        arrays = [pa.array([row.get(col) for row in rows]) for col in fieldnames]
        table = pa.Table.from_arrays(arrays, names=fieldnames)
        pq.write_table(table, path)
        return path

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Table does not exist: {path}")

    if path.suffix == ".parquet":
        _ensure_parquet_backend()
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        as_lists = table.to_pydict()
        columns = list(as_lists.keys())
        if not columns:
            return []
        n = len(as_lists[columns[0]])
        return [{col: as_lists[col][i] for col in columns} for i in range(n)]

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_existing_path(parquet_path: Path, csv_path: Path, prefer_parquet: bool) -> Path:
    candidates = [parquet_path, csv_path] if prefer_parquet else [csv_path, parquet_path]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing table file. Tried: {parquet_path}, {csv_path}")
