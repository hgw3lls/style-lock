"""Typer-based CLI entrypoint for style_lock."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from .anchors import run_anchors
from .cluster import run_cluster
from .config import PipelineConfig, load_config
from .embed_clip import run_embed_clip
from .embed_dino import run_embed_dino
from .export_pack import run_export_pack
from .preprocess import run_preprocess
from .stats import run_stats
from .utils import coerce_device, seed_everything

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def resolve_config(
    config: Path,
    seed: int | None,
    device: str | None,
    batch_size: int | None,
    num_workers: int | None,
    limit: int | None = None,
    use_parquet: bool | None = None,
    mixed_precision: bool | None = None,
    cache_embeddings: bool | None = None,
    overrides: dict[str, object | None] | None = None,
) -> PipelineConfig:
    payload: dict[str, object | None] = {
        "seed": seed,
        "device": coerce_device(device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "limit": limit,
        "use_parquet": use_parquet,
        "mixed_precision": mixed_precision,
        "cache_embeddings": cache_embeddings,
    }
    payload.update(overrides or {})
    cfg = load_config(config, payload)
    seed_everything(cfg.seed)
    return cfg


def run_step(name: str, fn, hint: str) -> None:
    try:
        fn()
    except Exception as exc:
        console.print(Panel(f"[bold red]{name} failed[/bold red]\n\n{exc}\n\n[dim]{hint}[/dim]", title="Pipeline Error"))
        raise typer.Exit(code=1)


def output_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= 0


@app.command("preprocess")
def preprocess_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    num_workers: int | None = typer.Option(None, "--num-workers"),
    images_raw_dir: Path | None = typer.Option(None, "--images-raw-dir"),
    images_clean_dir: Path | None = typer.Option(None, "--images-clean-dir"),
    manifests_dir: Path | None = typer.Option(None, "--manifests-dir"),
    max_side: int | None = typer.Option(None, "--max-side"),
    jpeg_quality: int | None = typer.Option(None, "--jpeg-quality"),
    dedupe_threshold: int | None = typer.Option(None, "--dedupe-threshold"),
    limit: int | None = typer.Option(None, "--limit"),
    use_parquet: bool | None = typer.Option(None, "--use-parquet/--no-use-parquet"),
) -> None:
    cfg = resolve_config(config, seed, device, batch_size, num_workers, limit=limit, use_parquet=use_parquet, overrides={
        "images_raw_dir": images_raw_dir,
        "images_clean_dir": images_clean_dir,
        "manifests_dir": manifests_dir,
        "max_side": max_side,
        "jpeg_quality": jpeg_quality,
        "dedupe_threshold": dedupe_threshold,
    })
    run_preprocess(cfg, console=console)


@app.command("embed")
def embed_cmd(
    model: str = typer.Option("clip", "--model"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    num_workers: int | None = typer.Option(None, "--num-workers"),
    images_clean_dir: Path | None = typer.Option(None, "--images-clean-dir"),
    manifests_dir: Path | None = typer.Option(None, "--manifests-dir"),
    embeddings_dir: Path | None = typer.Option(None, "--embeddings-dir"),
    clip_arch: str | None = typer.Option(None, "--clip-arch"),
    clip_pretrained: str | None = typer.Option(None, "--clip-pretrained"),
    dino_model_name: str | None = typer.Option(None, "--dino-model-name"),
    force: bool = typer.Option(False, "--force"),
    limit: int | None = typer.Option(None, "--limit"),
    mixed_precision: bool | None = typer.Option(None, "--mixed-precision/--no-mixed-precision"),
    cache_embeddings: bool | None = typer.Option(None, "--cache-embeddings/--no-cache-embeddings"),
    use_parquet: bool | None = typer.Option(None, "--use-parquet/--no-use-parquet"),
) -> None:
    cfg = resolve_config(config, seed, device, batch_size, num_workers, limit=limit, use_parquet=use_parquet, mixed_precision=mixed_precision, cache_embeddings=cache_embeddings, overrides={
        "images_clean_dir": images_clean_dir,
        "manifests_dir": manifests_dir,
        "embeddings_dir": embeddings_dir,
        "clip_arch": clip_arch,
        "clip_pretrained": clip_pretrained,
        "dino_model_name": dino_model_name,
    })
    model = model.lower().strip()
    if model == "clip":
        run_embed_clip(cfg, force=force, console=console)
    elif model == "dino":
        run_embed_dino(cfg, force=force, console=console)
    else:
        raise typer.BadParameter("--model must be one of: clip, dino")


@app.command("stats")
def stats_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    num_workers: int | None = typer.Option(None, "--num-workers"),
    images_clean_dir: Path | None = typer.Option(None, "--images-clean-dir"),
    manifests_dir: Path | None = typer.Option(None, "--manifests-dir"),
    embeddings_dir: Path | None = typer.Option(None, "--embeddings-dir"),
    ink_luma_threshold: int | None = typer.Option(None, "--ink-luma-threshold"),
    void_luma_threshold: int | None = typer.Option(None, "--void-luma-threshold"),
    canny_low_threshold: int | None = typer.Option(None, "--canny-low-threshold"),
    canny_high_threshold: int | None = typer.Option(None, "--canny-high-threshold"),
    limit: int | None = typer.Option(None, "--limit"),
    use_parquet: bool | None = typer.Option(None, "--use-parquet/--no-use-parquet"),
) -> None:
    cfg = resolve_config(config, seed, device, batch_size, num_workers, limit=limit, use_parquet=use_parquet, overrides={
        "images_clean_dir": images_clean_dir,
        "manifests_dir": manifests_dir,
        "embeddings_dir": embeddings_dir,
        "ink_luma_threshold": ink_luma_threshold,
        "void_luma_threshold": void_luma_threshold,
        "canny_low_threshold": canny_low_threshold,
        "canny_high_threshold": canny_high_threshold,
    })
    run_stats(cfg, console=console)


@app.command("cluster")
def cluster_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    num_workers: int | None = typer.Option(None, "--num-workers"),
    embeddings_dir: Path | None = typer.Option(None, "--embeddings-dir"),
    manifests_dir: Path | None = typer.Option(None, "--manifests-dir"),
    outputs_dir: Path | None = typer.Option(None, "--outputs-dir"),
    w_dino: float | None = typer.Option(None, "--w-dino"),
    w_stats: float | None = typer.Option(None, "--w-stats"),
    w_clip: float | None = typer.Option(None, "--w-clip"),
    cluster_use_pca: bool | None = typer.Option(None, "--cluster-use-pca/--no-cluster-use-pca"),
    cluster_pca_dim: int | None = typer.Option(None, "--cluster-pca-dim"),
    hdbscan_min_cluster_size: int | None = typer.Option(None, "--hdbscan-min-cluster-size"),
    hdbscan_min_samples: int | None = typer.Option(None, "--hdbscan-min-samples"),
) -> None:
    cfg = resolve_config(config, seed, device, batch_size, num_workers, overrides={
        "embeddings_dir": embeddings_dir,
        "manifests_dir": manifests_dir,
        "outputs_dir": outputs_dir,
        "w_dino": w_dino,
        "w_stats": w_stats,
        "w_clip": w_clip,
        "cluster_use_pca": cluster_use_pca,
        "cluster_pca_dim": cluster_pca_dim,
        "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
        "hdbscan_min_samples": hdbscan_min_samples,
    })
    run_cluster(cfg, console=console)


@app.command("anchors")
def anchors_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    num_workers: int | None = typer.Option(None, "--num-workers"),
    manifests_dir: Path | None = typer.Option(None, "--manifests-dir"),
    images_clean_dir: Path | None = typer.Option(None, "--images-clean-dir"),
    outputs_dir: Path | None = typer.Option(None, "--outputs-dir"),
    anchors_include_noise: bool | None = typer.Option(None, "--include-noise/--exclude-noise"),
    anchors_k_centroids: int | None = typer.Option(None, "--anchors-k-centroids"),
    anchors_m_edges: int | None = typer.Option(None, "--anchors-m-edges"),
    anchor_crop_size: int | None = typer.Option(None, "--anchor-crop-size"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    cfg = resolve_config(config, seed, device, batch_size, num_workers, overrides={
        "manifests_dir": manifests_dir,
        "images_clean_dir": images_clean_dir,
        "outputs_dir": outputs_dir,
        "anchors_include_noise": anchors_include_noise,
        "anchors_k_centroids": anchors_k_centroids,
        "anchors_m_edges": anchors_m_edges,
        "anchor_crop_size": anchor_crop_size,
    })
    run_anchors(cfg, force=force, console=console)


@app.command("export")
def export_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    num_workers: int | None = typer.Option(None, "--num-workers"),
    outputs_dir: Path | None = typer.Option(None, "--outputs-dir"),
    export_dir: Path | None = typer.Option(None, "--export-dir"),
    export_top_n_clusters: int | None = typer.Option(None, "--export-top-n-clusters"),
    export_rank_by: str | None = typer.Option(None, "--export-rank-by"),
) -> None:
    cfg = resolve_config(config, seed, device, batch_size, num_workers, overrides={
        "outputs_dir": outputs_dir,
        "export_dir": export_dir,
        "export_top_n_clusters": export_top_n_clusters,
        "export_rank_by": export_rank_by,
    })
    run_export_pack(cfg, console=console)


@app.command("run-all")
def run_all_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    num_workers: int | None = typer.Option(None, "--num-workers"),
    force: bool = typer.Option(False, "--force"),
    limit: int | None = typer.Option(None, "--limit"),
    use_parquet: bool | None = typer.Option(None, "--use-parquet/--no-use-parquet"),
    mixed_precision: bool | None = typer.Option(None, "--mixed-precision/--no-mixed-precision"),
    cache_embeddings: bool | None = typer.Option(None, "--cache-embeddings/--no-cache-embeddings"),
) -> None:
    cfg = resolve_config(
        config,
        seed,
        device,
        batch_size,
        num_workers,
        limit=limit,
        use_parquet=use_parquet,
        mixed_precision=mixed_precision,
        cache_embeddings=cache_embeddings,
        overrides=None,
    )

    manifest_path = cfg.manifests_dir / ("images_clean.parquet" if cfg.use_parquet else "images_clean.csv")
    dino_path = cfg.embeddings_dir / "dino.npy"
    clip_path = cfg.embeddings_dir / "clip.npy"
    stats_path = cfg.embeddings_dir / "stats.npy"
    cluster_json = cfg.outputs_dir / "clusters.json"
    anchor_index = cfg.outputs_dir / "anchors_index.csv"
    export_flag = cfg.export_dir / "STYLE_LOCK_SPEC.md"

    if force or not output_exists(manifest_path):
        run_step("preprocess", lambda: run_preprocess(cfg, console=console), "Check images_raw_dir and image formats.")
    else:
        console.print("[yellow]Skipping preprocess (manifest exists).[/yellow]")

    if force or not output_exists(dino_path):
        run_step("embed dino", lambda: run_embed_dino(cfg, force=force, console=console), "Install timm/torch and verify manifest/images_clean_dir.")
    else:
        console.print("[yellow]Skipping embed dino (dino.npy exists).[/yellow]")

    if force or not output_exists(clip_path):
        run_step("embed clip", lambda: run_embed_clip(cfg, force=force, console=console), "Install open_clip_torch and verify manifest/images_clean_dir.")
    else:
        console.print("[yellow]Skipping embed clip (clip.npy exists).[/yellow]")

    if force or not output_exists(stats_path):
        run_step("stats", lambda: run_stats(cfg, console=console), "Check OpenCV installation and images_clean_dir paths.")
    else:
        console.print("[yellow]Skipping stats (stats.npy exists).[/yellow]")

    if force or not output_exists(cluster_json):
        run_step("cluster", lambda: run_cluster(cfg, console=console), "Ensure dino/clip/stats arrays exist and row counts match manifest.")
    else:
        console.print("[yellow]Skipping cluster (clusters.json exists).[/yellow]")

    if force or not output_exists(anchor_index):
        run_step("anchors", lambda: run_anchors(cfg, force=force, console=console), "Run cluster first and ensure clean images are available.")
    else:
        console.print("[yellow]Skipping anchors (anchors_index.csv exists).[/yellow]")

    if force or not output_exists(export_flag):
        run_step("export", lambda: run_export_pack(cfg, console=console), "Run anchors/cluster first to produce exportable assets.")
    else:
        console.print("[yellow]Skipping export (STYLE_LOCK_SPEC.md exists).[/yellow]")


if __name__ == "__main__":
    app()
