from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner


@pytest.mark.smoke
def test_pipeline_smoke(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("hdbscan")
    pytest.importorskip("umap")
    pytest.importorskip("matplotlib")

    from style_lock.cli import app

    raw = tmp_path / "raw"
    raw.mkdir(parents=True)

    # 10 synthetic images
    for i in range(10):
        img = Image.new("RGB", (300, 220), color=(255 - i * 10, 200 - i * 5, 120 + i * 3))
        d = ImageDraw.Draw(img)
        d.rectangle((20 + i * 5, 30, 150, 170), outline=(10, 10, 10), width=3)
        d.ellipse((80, 40 + i * 3, 240, 200), outline=(30, 30, 30), width=4)
        img.save(raw / f"img_{i}.png")

    cfg = {
        "seed": 7,
        "device": "cpu",
        "batch_size": 4,
        "num_workers": 0,
        "images_raw_dir": str(raw),
        "images_clean_dir": str(tmp_path / "clean"),
        "manifests_dir": str(tmp_path / "manifests"),
        "embeddings_dir": str(tmp_path / "embeddings"),
        "outputs_dir": str(tmp_path / "outputs"),
        "export_dir": str(tmp_path / "pack"),
        "max_side": 256,
        "hdbscan_min_cluster_size": 3,
        "cluster_use_pca": False,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("\n".join(f"{k}: {json.dumps(v) if isinstance(v,str) else v}" for k, v in cfg.items()))

    def fake_dino(config, force=False, console=None):
        manifest = config.manifests_dir / "images_clean.csv"
        with manifest.open("r", encoding="utf-8", newline="") as h:
            n = sum(1 for _ in csv.DictReader(h))
        config.embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.save(config.embeddings_dir / "dino.npy", np.random.RandomState(0).randn(n, 32).astype(np.float32))
        (config.embeddings_dir / "dino_meta.json").write_text("{}")

    def fake_clip(config, force=False, console=None):
        manifest = config.manifests_dir / "images_clean.csv"
        with manifest.open("r", encoding="utf-8", newline="") as h:
            n = sum(1 for _ in csv.DictReader(h))
        config.embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.save(config.embeddings_dir / "clip.npy", np.random.RandomState(1).randn(n, 24).astype(np.float32))
        (config.embeddings_dir / "clip_meta.json").write_text("{}")

    monkeypatch.setattr("style_lock.cli.run_embed_dino", fake_dino)
    monkeypatch.setattr("style_lock.cli.run_embed_clip", fake_clip)

    runner = CliRunner()
    result = runner.invoke(app, ["run-all", "--config", str(cfg_path), "--device", "cpu", "--force"])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "manifests" / "images_clean.csv").exists()
    assert (tmp_path / "embeddings" / "stats.npy").exists()
    assert (tmp_path / "outputs" / "style_vec.npy").exists()
    assert (tmp_path / "outputs" / "clusters.json").exists()
    assert (tmp_path / "outputs" / "anchors_index.csv").exists()
    assert (tmp_path / "pack" / "STYLE_LOCK_SPEC.md").exists()

    stats = np.load(tmp_path / "embeddings" / "stats.npy")
    style_vec = np.load(tmp_path / "outputs" / "style_vec.npy")
    with (tmp_path / "manifests" / "images_clean.csv").open("r", encoding="utf-8", newline="") as h:
        n = sum(1 for _ in csv.DictReader(h))

    assert stats.shape[0] == n
    assert style_vec.shape[0] == n
