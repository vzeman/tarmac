from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from tarmac.cluster.cluster import run_clustering
from tarmac.datasets.streetsurfacevis import download_streetsurfacevis
from tarmac.datasets.unify import build_manifest
from tarmac.embedding.embedder import DINOV3_MODEL, embed_manifest
from tarmac.eval.evaluate import run_evaluation

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
    batch_size: int = typer.Option(16, help="Image batch size."),
) -> None:
    """Embed manifest images with a frozen ViT backbone."""
    info = embed_manifest(
        manifest_path=manifest,
        output_path=output,
        faiss_index_path=faiss_index,
        metadata_path=metadata,
        model_name=model_name,
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
) -> None:
    """Cluster frozen full-image embeddings."""
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
def train() -> None:
    _stub("train")


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
) -> None:
    """Evaluate frozen embeddings and write Phase 2 reports."""
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
    console.print(
        f"Metrics written to {metrics}; backbone={result['backbone']} "
        f"chosen_k={result['chosen_k']}"
    )


@app.command()
def analyze() -> None:
    _stub("analyze")


@app.command()
def report() -> None:
    _stub("report")


@app.command()
def ui() -> None:
    _stub("ui")


if __name__ == "__main__":
    app()
