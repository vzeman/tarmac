from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"

app = typer.Typer(no_args_is_help=True)
download_app = typer.Typer(no_args_is_help=True)
app.add_typer(download_app, name="download")
console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@download_app.command("streetsurfacevis")
def download_streetsurfacevis_cmd(
    output_dir: Path = typer.Option(
        Path("data/raw/streetsurfacevis"),
        "--output-dir",
        "-o",
        help="Directory for the StreetSurfaceVis raw files.",
    ),
) -> None:
    """Download StreetSurfaceVis v1.0 from Zenodo."""
    from tarmac.datasets.streetsurfacevis import download_streetsurfacevis

    result = download_streetsurfacevis(output_dir)
    console.print(
        f"StreetSurfaceVis ready: {result.image_count} images, CSV at {result.csv_path}"
    )


@app.command()
def prepare(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Raw dataset root."),
    output: Path = typer.Option(
        Path("data/processed/manifest.parquet"), help="Manifest output path."
    ),
) -> None:
    """Build the unified parquet manifest."""
    from tarmac.datasets.unify import build_manifest

    manifest = build_manifest(raw_dir=raw_dir, output_path=output)
    console.print(f"Manifest written to {manifest.path} ({manifest.row_count} rows)")
    console.print(manifest.stats.to_string(index=False))


def _stub(command: str) -> None:
    console.print(f"{command}: stub for a later phase.")


@app.command()
def embed(
    manifest: Path = typer.Option(
        Path("data/processed/manifest.parquet"), help="Input manifest parquet."
    ),
    output: Path = typer.Option(
        Path("data/processed/embeddings.parquet"), help="Embedding parquet output."
    ),
    faiss_index: Path = typer.Option(
        Path("models/faiss_full.index"), help="FAISS full-image index output."
    ),
    metadata: Path = typer.Option(
        Path("models/embedding_metadata.json"), help="Embedding run metadata output."
    ),
    model_name: str = typer.Option(DINOV3_MODEL, help="Primary HF backbone model."),
    checkpoint: Path | None = typer.Option(None, help="Fine-tuned backbone checkpoint."),
    suffix: str | None = typer.Option(None, help="Artifact suffix, e.g. finetuned."),
    batch_size: int = typer.Option(16, help="Image batch size."),
) -> None:
    """Embed manifest images with a frozen ViT backbone."""
    from tarmac.embedding.embedder import embed_manifest

    output = _suffix_path(output, suffix)
    faiss_index = _suffix_path(faiss_index, suffix)
    metadata = _suffix_path(metadata, suffix)
    info = embed_manifest(
        manifest_path=manifest,
        output_path=output,
        faiss_index_path=faiss_index,
        metadata_path=metadata,
        model_name=model_name,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
    )
    console.print(
        f"Embeddings written to {output}; FAISS index at {faiss_index}; "
        f"backbone={info.backbone} ({info.model_name}) on {info.device}"
    )


@app.command()
def cluster(
    embeddings: Path = typer.Option(
        Path("data/processed/embeddings.parquet"), help="Embedding parquet input."
    ),
    centroids: Path = typer.Option(
        Path("models/kmeans_centroids.npy"), help="K-means centroid output."
    ),
    assignments: Path = typer.Option(
        Path("data/processed/cluster_assignments.parquet"), help="Cluster assignment output."
    ),
    profile: Path = typer.Option(
        Path("reports/cluster_profile.csv"), help="Cluster profile CSV output."
    ),
    metadata: Path = typer.Option(
        Path("models/cluster_metadata.json"), help="Cluster metadata output."
    ),
    suffix: str | None = typer.Option(None, help="Artifact suffix, e.g. finetuned."),
) -> None:
    """Cluster frozen full-image embeddings."""
    from tarmac.cluster.cluster import run_clustering

    embeddings = _suffix_path(embeddings, suffix)
    centroids = _suffix_path(centroids, suffix)
    assignments = _suffix_path(assignments, suffix)
    profile = _suffix_path(profile, suffix)
    metadata = _suffix_path(metadata, suffix)
    result = run_clustering(
        embeddings_path=embeddings,
        centroids_path=centroids,
        assignments_path=assignments,
        profile_path=profile,
        metadata_path=metadata,
    )
    console.print(
        f"Cluster assignments written to {assignments}; chosen k={result['chosen_k']} "
        f"(train silhouette={result['best_train_silhouette']:.4f})"
    )


@app.command()
def train(
    manifest: Path = typer.Option(
        Path("data/processed/manifest.parquet"), help="Input manifest parquet."
    ),
    checkpoint: Path = typer.Option(
        Path("models/finetuned_backbone.pt"), help="Best backbone checkpoint output."
    ),
    metadata: Path = typer.Option(
        Path("models/finetuned_backbone.json"), help="Training config/history JSON output."
    ),
    model_name: str = typer.Option(DINOV3_MODEL, help="Primary HF backbone model."),
    epochs: int = typer.Option(10, help="Maximum fine-tuning epochs."),
    batch_size: int = typer.Option(32, help="Physical image batch size."),
    effective_batch_size: int = typer.Option(128, help="Effective batch size via accumulation."),
    backbone_lr: float = typer.Option(5e-5, help="Backbone AdamW learning rate."),
    head_lr: float = typer.Option(5e-4, help="Projection-head AdamW learning rate."),
    device: str = typer.Option("auto", help="Training device: auto, mps, or cpu."),
    run_name: str = typer.Option("supcon", help="Run name for per-epoch checkpoints."),
    resume: bool = typer.Option(False, help="Resume from the latest epoch checkpoint for run-name."),
    patience: int = typer.Option(3, help="Early-stopping patience in epochs."),
    attn_implementation: str = typer.Option("eager", help="HF attention implementation, e.g. eager or sdpa."),
) -> None:
    """Fine-tune the last four ViT blocks with supervised contrastive loss."""
    from tarmac.train.supcon import train_supcon

    result = train_supcon(
        manifest_path=manifest,
        output_checkpoint=checkpoint,
        output_metadata=metadata,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        device_name=device,
        run_name=run_name,
        resume=resume,
        patience=patience,
        attn_implementation=attn_implementation,
    )
    console.print(
        f"Training complete: backbone={result['backbone']} epochs={result['epochs_trained']} "
        f"best_epoch={result['best_epoch']} best_val_quality_macro_f1={result['best_val_quality_macro_f1']:.4f}"
    )


@app.command()
def evaluate(
    embeddings: Path = typer.Option(
        Path("data/processed/embeddings.parquet"), help="Embedding parquet input."
    ),
    assignments: Path = typer.Option(
        Path("data/processed/cluster_assignments.parquet"), help="Cluster assignment input."
    ),
    embed_metadata: Path = typer.Option(
        Path("models/embedding_metadata.json"), help="Embedding metadata JSON."
    ),
    cluster_metadata: Path = typer.Option(
        Path("models/cluster_metadata.json"), help="Cluster metadata JSON."
    ),
    metrics: Path = typer.Option(
        Path("reports/phase2_metrics.json"), help="Metrics JSON output."
    ),
    report_path: Path = typer.Option(
        Path("reports/PHASE2_BASELINE.md"), help="Markdown report output."
    ),
    umap_html: Path = typer.Option(
        Path("reports/umap_scatter.html"), help="Interactive UMAP HTML output."
    ),
    umap_png: Path = typer.Option(
        Path("reports/umap_quality.png"), help="Static UMAP PNG output."
    ),
    suffix: str | None = typer.Option(None, help="Artifact suffix, e.g. finetuned."),
    frozen_metrics: Path = typer.Option(
        Path("reports/phase2_metrics.json"), help="Frozen baseline metrics for Phase 3 comparison."
    ),
) -> None:
    """Evaluate frozen embeddings and write Phase 2 reports."""
    from tarmac.eval.evaluate import run_evaluation, write_phase3_comparison_report

    embeddings = _suffix_path(embeddings, suffix)
    assignments = _suffix_path(assignments, suffix)
    embed_metadata = _suffix_path(embed_metadata, suffix)
    cluster_metadata = _suffix_path(cluster_metadata, suffix)
    if suffix == "finetuned":
        metrics = Path("reports/phase3_metrics.json")
        report_path = Path("reports/PHASE3_FINETUNE.md")
        umap_html = Path("reports/umap_scatter_finetuned.html")
        umap_png = Path("reports/umap_quality_finetuned.png")
    else:
        metrics = _suffix_path(metrics, suffix)
        report_path = _suffix_path(report_path, suffix)
        umap_html = _suffix_path(umap_html, suffix)
        umap_png = _suffix_path(umap_png, suffix)
    result = run_evaluation(
        embeddings_path=embeddings,
        assignments_path=assignments,
        embed_metadata_path=embed_metadata,
        cluster_metadata_path=cluster_metadata,
        metrics_path=metrics,
        report_path=report_path,
        umap_html_path=umap_html,
        umap_png_path=umap_png,
    )
    if suffix == "finetuned":
        write_phase3_comparison_report(
            frozen_metrics_path=frozen_metrics,
            finetuned_metrics_path=metrics,
            report_path=report_path,
        )
    console.print(
        f"Metrics written to {metrics}; backbone={result['backbone']} "
        f"chosen_k={result['chosen_k']}"
    )


@app.command()
def analyze(
    path: Path = typer.Argument(..., help="Photo, directory of photos, or video to analyze."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output run directory."),
    fps: float = typer.Option(2.0, "--fps", help="Video frame extraction rate."),
    k: int = typer.Option(10, "--k", help="Nearest neighbors per tile."),
    non_road_threshold: float | None = typer.Option(
        None,
        "--non-road-threshold",
        help="Mean cosine threshold below which a tile is marked non-road. Default calibrates from val tiles, capped at 0.45.",
    ),
    batch_size: int = typer.Option(16, "--batch-size", help="Embedding batch size."),
    device: str = typer.Option("cpu", "--device", help="Inference device: cpu, mps, or auto."),
) -> None:
    """Analyze a photo, image directory, or video."""
    from tarmac.inference.analyze import analyze_path, print_summary

    summary = analyze_path(
        input_path=path,
        out_dir=out,
        fps=fps,
        k=k,
        non_road_threshold=non_road_threshold,
        batch_size=batch_size,
        device=device,
    )
    print_summary(summary, console)


@app.command()
def visualize(
    directory: Path = typer.Argument(..., help="Directory of jpg/png/webp images to visualize recursively."),
    out: Path | None = typer.Option(None, "--out", "-o", help="HTML output path."),
    k: int = typer.Option(10, "--k", help="Nearest neighbors per image."),
    batch_size: int = typer.Option(16, "--batch-size", help="Embedding batch size."),
    device: str = typer.Option("cpu", "--device", help="Inference device: cpu, mps, or auto."),
) -> None:
    """Project a folder of images into the persisted reference UMAP space."""
    from tarmac.inference.visualize import visualize_directory

    report_path = visualize_directory(
        directory=directory,
        out=out,
        k=k,
        batch_size=batch_size,
        device=device,
    )
    console.print(f"Visualization written to {report_path}")


@app.command()
def report(
    run_dir: Path = typer.Argument(..., help="Run directory produced by `tarmac analyze`."),
    output: Path | None = typer.Option(None, "--output", "-o", help="HTML output path."),
) -> None:
    """Build a self-contained HTML report for an analysis run."""
    from tarmac.report.html import build_html_report

    report_path = build_html_report(run_dir=run_dir, output=output)
    console.print(f"Report written to {report_path}")


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", help="Streamlit server port."),
) -> None:
    """Start the Streamlit UI."""
    import subprocess

    cmd = [
        "streamlit",
        "run",
        "src/tarmac/ui/app.py",
        "--server.port",
        str(port),
    ]
    raise typer.Exit(subprocess.call(cmd))


def _suffix_path(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    if suffix == "finetuned" and path.name == "phase2_metrics.json":
        return path.with_name("phase3_metrics.json")
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


if __name__ == "__main__":
    app()
