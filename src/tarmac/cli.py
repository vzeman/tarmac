from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

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


@download_app.command("cracks-concrete-pavement")
def download_cracks_concrete_pavement_cmd(
    output_dir: Path = typer.Option(
        Path("data/raw/cracks_concrete_pavement"),
        "--output-dir",
        "-o",
        help="Directory for the Mendeley concrete/pavement crack dataset.",
    ),
) -> None:
    """Download Mendeley 429vzbgmbx crack/non-crack images."""
    from tarmac.datasets.cracks_concrete_pavement import download_cracks_concrete_pavement

    result = download_cracks_concrete_pavement(output_dir)
    console.print(
        f"Concrete/pavement cracks ready: positive={result.positive_count}, "
        f"negative={result.negative_count}, dir={result.output_dir}"
    )


@download_app.command("crack500")
def download_crack500_cmd(
    output_dir: Path = typer.Option(Path("data/raw/crack500"), "--output-dir", "-o"),
) -> None:
    """Download CRACK500 from known GitHub mirrors when available."""
    from tarmac.datasets.crack_masks import download_crack500

    result = download_crack500(output_dir)
    console.print(result.message)


@download_app.command("deepcrack")
def download_deepcrack_cmd(
    output_dir: Path = typer.Option(Path("data/raw/deepcrack"), "--output-dir", "-o"),
) -> None:
    """Download DeepCrack from GitHub when available."""
    from tarmac.datasets.crack_masks import download_deepcrack

    result = download_deepcrack(output_dir)
    console.print(result.message)


@download_app.command("runway-roboflow")
def download_runway_roboflow_cmd(
    output_dir: Path = typer.Option(Path("data/raw/runway_roboflow"), "--output-dir", "-o"),
    version: int | None = typer.Option(None, "--version", help="Roboflow project version."),
) -> None:
    """Download runway-specific crack boxes from Roboflow Universe."""
    from tarmac.datasets.runway_roboflow import download_runway_roboflow

    result = download_runway_roboflow(output_dir=output_dir, version=version)
    console.print(
        f"Runway Roboflow ready: images={result.image_count}, tile_labels={result.tile_label_count}, "
        f"positive_tiles={result.positive_tile_count}, negative_tiles={result.negative_tile_count}"
    )


@download_app.command("crackairport")
def download_crackairport_cmd(
    output_dir: Path = typer.Option(
        Path("data/raw/crackairport"),
        "--output-dir",
        "-o",
        help="Directory for the Mendeley CrackAirport dataset.",
    ),
) -> None:
    """Download CrackAirport 3v5r2fxf89 v1 from Mendeley."""
    from tarmac.datasets.crackairport import download_crackairport

    result = download_crackairport(output_dir)
    console.print(
        f"CrackAirport ready: pairs={result.pair_count}, images={result.image_count}, "
        f"masks={result.mask_count}, pairs_index={result.pairs_path}"
    )


@download_app.command("crackforest")
def download_crackforest_cmd(
    output_dir: Path = typer.Option(
        Path("data/raw/crackforest"),
        "--output-dir",
        "-o",
        help="Directory for normalized CrackForest files.",
    ),
) -> None:
    """Download CrackForest/CFD from GitHub and convert masks to binary PNG."""
    from tarmac.datasets.crackforest import download_crackforest

    result = download_crackforest(output_dir)
    console.print(
        f"CrackForest ready: pairs={result.pair_count}, images={result.image_count}, "
        f"masks={result.mask_count}, pairs_index={result.pairs_path}"
    )


@download_app.command("rdd2022")
def download_rdd2022_cmd(
    output_dir: Path = typer.Option(
        Path("data/raw/rdd2022"),
        "--output-dir",
        "-o",
        help="Directory for normalized RDD2022 files.",
    ),
    country: str = typer.Option("Czech", "--country", help="Country subset to download."),
    max_download_mb: float = typer.Option(
        1024.0,
        "--max-download-mb",
        help="Skip automatic download above this upstream archive size.",
    ),
) -> None:
    """Download one RDD2022 country subset and normalize annotated train data."""
    from tarmac.datasets.rdd2022 import download_rdd2022

    result = download_rdd2022(output_dir=output_dir, country=country, max_download_mb=max_download_mb)
    if result.downloaded:
        console.print(
            f"RDD2022 ready: country={result.country}, images={result.image_count}, "
            f"annotations={result.annotation_count}, classes={result.class_counts}"
        )
    else:
        console.print(f"RDD2022 skipped: country={result.country}; instructions={result.output_dir / 'MANUAL_DOWNLOAD.md'}")


@download_app.command("codebrim")
def download_codebrim_cmd(
    output_dir: Path = typer.Option(
        Path("data/raw/codebrim"),
        "--output-dir",
        "-o",
        help="Directory for the Zenodo CODEBRIM dataset.",
    ),
) -> None:
    """Download CODEBRIM from Zenodo record 2620293."""
    from tarmac.datasets.codebrim import download_codebrim

    result = download_codebrim(output_dir)
    console.print(f"CODEBRIM ready: images={result.image_count}, annotations={result.annotations_path}")
    console.print(result.class_counts)


@download_app.command("sdnet2018")
def download_sdnet2018_cmd(
    output_dir: Path = typer.Option(
        Path("data/raw/sdnet2018"),
        "--output-dir",
        "-o",
        help="Directory for the SDNET2018 dataset.",
    ),
) -> None:
    """Download SDNET2018 from a keyless mirror when available."""
    from tarmac.datasets.sdnet2018 import download_sdnet2018

    result = download_sdnet2018(output_dir)
    if result.downloaded:
        console.print(
            f"SDNET2018 ready: images={result.image_count}, source={result.source}, dir={result.output_dir}"
        )
        console.print(result.counts)
    else:
        console.print(f"SDNET2018 skipped: {result.message}")


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


@app.command("prepare-cracks")
def prepare_cracks(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Raw dataset root."),
    output: Path = typer.Option(
        Path("data/processed/crack_manifest.parquet"), help="Crack manifest output path."
    ),
) -> None:
    """Build the separate crack-detection parquet manifest."""
    from tarmac.datasets.crack_manifest import build_crack_manifest

    manifest = build_crack_manifest(raw_dir=raw_dir, output_path=output)
    console.print(f"Crack manifest written to {manifest.path} ({manifest.row_count} rows)")
    console.print(manifest.stats.to_string(index=False))


@app.command("prepare-defects")
def prepare_defects(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Raw dataset root."),
    output: Path = typer.Option(
        Path("data/processed/defect_manifest.parquet"), help="Defect manifest output path."
    ),
) -> None:
    """Build the unified multi-domain, multi-label defect manifest."""
    from tarmac.datasets.defect_manifest import build_defect_manifest

    manifest = build_defect_manifest(raw_dir=raw_dir, output_path=output)
    console.print(f"Defect manifest written to {manifest.path} ({manifest.row_count} rows)")
    console.print(manifest.source_domain_stats.to_string(index=False))
    console.print(manifest.label_totals.to_string(index=False))


@app.command("embed-defects")
def embed_defects(
    manifest: Path = typer.Option(
        Path("data/processed/defect_manifest.parquet"), help="Input defect manifest parquet."
    ),
    output: Path = typer.Option(
        Path("data/processed/defect_embeddings.parquet"), help="Defect embedding parquet output."
    ),
    metadata: Path = typer.Option(
        Path("data/processed/defect_embeddings.json"), help="Defect embedding metadata JSON."
    ),
    none_cap: int = typer.Option(20_000, help="Maximum pure-none rows to embed."),
    batch_size: int = typer.Option(64, help="MPS backbone embedding batch size."),
    num_workers: int = typer.Option(0, help="DataLoader workers."),
    force: bool = typer.Option(False, help="Rebuild the embedding cache even if it exists."),
) -> None:
    """Embed the balanced defect-manifest subset with the frozen active DINOv3 backbone."""
    from tarmac.defect.embeddings import build_defect_embeddings

    result = build_defect_embeddings(
        manifest_path=manifest,
        output_path=output,
        metadata_path=metadata,
        none_cap=none_cap,
        batch_size=batch_size,
        num_workers=num_workers,
        force=force,
    )
    console.print(
        f"Defect embeddings ready: rows={result.row_count} positives={result.positive_rows} "
        f"pure_none={result.pure_none_rows} dim={result.embedding_dim}; output={result.embeddings_path}"
    )


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


@app.command("train-crack")
def train_crack(
    manifest: Path = typer.Option(
        Path("data/processed/crack_manifest.parquet"), help="Input crack manifest parquet."
    ),
    checkpoint: Path = typer.Option(Path("models/crack_head.pt"), help="Best crack head output."),
    metadata: Path = typer.Option(Path("models/crack_head.json"), help="Training metadata JSON."),
    epochs: int = typer.Option(8, help="Maximum crack-head epochs."),
    batch_size: int = typer.Option(128, help="Embedding/head batch size."),
    lr: float = typer.Option(1e-3, help="Crack-head AdamW learning rate."),
    patience: int = typer.Option(3, help="Early-stopping patience."),
    resume: bool = typer.Option(False, help="Resume from latest crack checkpoint."),
) -> None:
    """Train a binary crack classifier head on frozen active-backbone embeddings."""
    from tarmac.crack.train import train_crack_head

    result = train_crack_head(
        manifest_path=manifest,
        output_checkpoint=checkpoint,
        output_metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        resume=resume,
    )
    console.print(
        f"Crack training complete: best_epoch={result['best_epoch']} "
        f"best_val_f1={result['best_val_f1']:.4f}; checkpoint={result['checkpoint']}"
    )


@app.command("train-defect")
def train_defect(
    manifest: Path = typer.Option(
        Path("data/processed/defect_manifest.parquet"), help="Input defect manifest parquet."
    ),
    embeddings: Path = typer.Option(
        Path("data/processed/defect_embeddings.parquet"), help="Cached defect embeddings parquet."
    ),
    embedding_metadata: Path = typer.Option(
        Path("data/processed/defect_embeddings.json"), help="Cached defect embeddings metadata JSON."
    ),
    checkpoint: Path = typer.Option(Path("models/defect_head.pt"), help="Best defect head output."),
    metadata: Path = typer.Option(Path("models/defect_head.json"), help="Training metadata JSON."),
    epochs: int = typer.Option(40, help="Maximum defect-head epochs."),
    batch_size: int = typer.Option(512, help="Head training batch size."),
    embed_batch_size: int = typer.Option(64, help="MPS backbone batch size when cache is missing."),
    lr: float = typer.Option(1e-3, help="Defect-head AdamW learning rate."),
    patience: int = typer.Option(5, help="Early-stopping patience."),
    resume: bool = typer.Option(False, help="Resume from latest defect checkpoint."),
    none_cap: int = typer.Option(20_000, help="Maximum pure-none rows in the embedding cache."),
) -> None:
    """Train a multi-label structural defect classifier head on cached embeddings."""
    from tarmac.defect.train import train_defect_head

    result = train_defect_head(
        manifest_path=manifest,
        embeddings_path=embeddings,
        embedding_metadata_path=embedding_metadata,
        output_checkpoint=checkpoint,
        output_metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        embed_batch_size=embed_batch_size,
        lr=lr,
        patience=patience,
        resume=resume,
        none_cap=none_cap,
    )
    console.print(
        f"Defect training complete: best_epoch={result['best_epoch']} "
        f"best_val_macro_ap={result['best_val_macro_ap']:.4f}; checkpoint={result['checkpoint']}"
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


@app.command("evaluate-crack")
def evaluate_crack(
    manifest: Path = typer.Option(
        Path("data/processed/crack_manifest.parquet"), help="Input crack manifest parquet."
    ),
    checkpoint: Path = typer.Option(Path("models/crack_head.pt"), help="Crack head checkpoint."),
    metrics: Path = typer.Option(Path("reports/crack_metrics.json"), help="Metrics JSON output."),
    report_path: Path = typer.Option(Path("reports/CRACK_DETECTION.md"), help="Markdown report output."),
    batch_size: int = typer.Option(128, help="Embedding batch size."),
    device: str = typer.Option("auto", help="Embedding device: auto, mps, or cpu."),
) -> None:
    """Evaluate the crack classifier on val and test splits."""
    from tarmac.crack.evaluate import evaluate_crack_head

    result = evaluate_crack_head(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        report_path=report_path,
        batch_size=batch_size,
        device_name=device,
    )
    test = result["test"]["overall"]
    console.print(
        f"Crack metrics written to {metrics}; threshold={result['threshold']:.3f} "
        f"test_f1={test['f1']:.4f} precision={test['precision']:.4f} recall={test['recall']:.4f}"
    )


@app.command("train-seg-head")
def train_seg_head_cmd(
    manifest: Path = typer.Option(
        Path("data/processed/crack_seg_expanded/manifest.jsonl"),
        help="Expanded segmentation manifest with source masks.",
    ),
    checkpoint: Path = typer.Option(Path("models/crack_seg_head.pt"), help="Best dense segmentation head output."),
    metadata: Path = typer.Option(Path("models/crack_seg_head.json"), help="Training metadata JSON."),
    checkpoint_dir: Path = typer.Option(
        Path("models/checkpoints/seg_head"),
        help="Per-epoch checkpoint directory.",
    ),
    epochs: int = typer.Option(60, help="Maximum dense-head epochs."),
    batch_size: int = typer.Option(4, help="MPS image batch size."),
    lr: float = typer.Option(2e-4, help="Dense-head AdamW learning rate."),
    weight_decay: float = typer.Option(1e-4, help="Dense-head AdamW weight decay."),
    patience: int = typer.Option(8, help="Early-stopping patience."),
    seed: int = typer.Option(42, help="Training seed."),
    resume: bool = typer.Option(False, help="Resume from latest dense-head checkpoint."),
    num_workers: int = typer.Option(0, help="DataLoader workers."),
    device: str = typer.Option("mps", help="Training device; MPS only."),
) -> None:
    """Train the frozen-DINOv3 dense patch-token crack segmentation head."""
    from tarmac.crack.seg_head import train_seg_head

    result = train_seg_head(
        manifest_path=manifest,
        output_checkpoint=checkpoint,
        output_metadata=metadata,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        seed=seed,
        resume=resume,
        num_workers=num_workers,
        device=device,
    )
    console.print(
        f"Seg head trained: best_epoch={result['best_epoch']} "
        f"best_val_dice={result['best_val_dice']:.4f}; checkpoint={result['checkpoint']}"
    )


@app.command("evaluate-seg-head")
def evaluate_seg_head_cmd(
    manifest: Path = typer.Option(
        Path("data/processed/crack_seg_expanded/manifest.jsonl"),
        help="Expanded segmentation manifest with source masks.",
    ),
    checkpoint: Path = typer.Option(Path("models/crack_seg_head.pt"), help="Dense segmentation head checkpoint."),
    metadata: Path = typer.Option(Path("models/crack_seg_head.json"), help="Dense segmentation metadata JSON."),
    metrics: Path = typer.Option(Path("reports/crack_seg_head_metrics.json"), help="Metrics JSON output."),
    report_path: Path = typer.Option(Path("reports/CRACK_SEGMENTATION.md"), help="Markdown report output."),
    batch_size: int = typer.Option(4, help="MPS image batch size."),
    num_workers: int = typer.Option(0, help="DataLoader workers."),
    device: str = typer.Option("mps", help="Evaluation device; MPS only."),
    render_examples: bool = typer.Option(True, help="Render CrackAirport example panels."),
    examples_dir: Path = typer.Option(Path("reports/examples"), help="Example overlay output directory."),
    compare_classical: bool = typer.Option(True, help="Compare the classical fallback on the common test split."),
) -> None:
    """Evaluate the dense segmentation head on val/test masks and update the crack segmentation report."""
    from tarmac.crack.seg_head import evaluate_seg_head

    result = evaluate_seg_head(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        metadata_path=metadata,
        metrics_path=metrics,
        report_path=report_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        render_examples=render_examples,
        examples_dir=examples_dir,
        compare_classical=compare_classical,
    )
    test = result["test"]["overall"]
    console.print(
        f"Seg head metrics written to {metrics}; threshold={result['threshold']:.3f} "
        f"test_iou={test['iou']:.4f} test_dice={test['dice']:.4f}"
    )


@app.command("evaluate-defect")
def evaluate_defect(
    embeddings: Path = typer.Option(
        Path("data/processed/defect_embeddings.parquet"), help="Cached defect embeddings parquet."
    ),
    checkpoint: Path = typer.Option(Path("models/defect_head.pt"), help="Defect head checkpoint."),
    metadata: Path = typer.Option(Path("models/defect_head.json"), help="Defect head metadata JSON."),
    metrics: Path = typer.Option(Path("reports/defect_metrics.json"), help="Metrics JSON output."),
    report_path: Path = typer.Option(Path("reports/DEFECT_DETECTION.md"), help="Markdown report output."),
) -> None:
    """Evaluate the multi-label structural defect classifier on val and test splits."""
    from tarmac.defect.evaluate import evaluate_defect_head

    result = evaluate_defect_head(
        embeddings_path=embeddings,
        checkpoint_path=checkpoint,
        metadata_path=metadata,
        metrics_path=metrics,
        report_path=report_path,
    )
    _print_defect_metrics(result)
    console.print(f"Defect metrics written to {metrics}; report={report_path}")


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
    region: str = typer.Option("auto", "--region", help="Tile region: auto, lower_half, or full."),
    crack_segmentation: bool = typer.Option(
        False,
        "--crack-segmentation",
        help="Render pixel-precise crack masks and add crack geometry columns.",
    ),
    mm_per_pixel: float | None = typer.Option(
        None,
        "--mm-per-pixel",
        help="Optional calibration for metric crack measurements.",
    ),
    defect_gating: bool = typer.Option(
        True,
        "--defect-gating/--no-defect-gating",
        help="Gate CODEBRIM-backed concrete defect labels to concrete/structural surfaces.",
    ),
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
        region=region,
        crack_segmentation=crack_segmentation,
        mm_per_pixel=mm_per_pixel,
        defect_gating=defect_gating,
    )
    print_summary(summary, console)


@app.command()
def assess(
    path: Path = typer.Argument(..., help="Photo, directory of photos, or video to assess."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output assessment run directory."),
    mm_per_pixel: float | None = typer.Option(
        None,
        "--mm-per-pixel",
        help="Optional calibration for metric crack measurements and AASHTO width banding.",
    ),
    fps: float = typer.Option(2.0, "--fps", help="Video frame extraction rate."),
    k: int = typer.Option(10, "--k", help="Nearest neighbors per tile."),
    non_road_threshold: float | None = typer.Option(
        None,
        "--non-road-threshold",
        help="Mean cosine threshold below which a tile is marked non-road. Default calibrates from val tiles.",
    ),
    batch_size: int = typer.Option(16, "--batch-size", help="Embedding batch size."),
    device: str = typer.Option("cpu", "--device", help="Inference device. Use cpu for reproducible smoke runs."),
    region: str = typer.Option("auto", "--region", help="Tile region: auto, lower_half, or full."),
    defect_gating: bool = typer.Option(
        True,
        "--defect-gating/--no-defect-gating",
        help="Gate CODEBRIM-backed concrete defect labels to concrete/structural surfaces.",
    ),
) -> None:
    """Run analysis and aggregate a PCI-proxy condition/repair-priority assessment."""
    from tarmac.inference.assess import assess_path, print_assessment_summary

    payload = assess_path(
        input_path=path,
        out_dir=out,
        fps=fps,
        k=k,
        non_road_threshold=non_road_threshold,
        batch_size=batch_size,
        device=device,
        region=region,
        mm_per_pixel=mm_per_pixel,
        defect_gating=defect_gating,
    )
    print_assessment_summary(payload, console)


@app.command("survey")
def survey_cmd(
    video: Path = typer.Argument(..., help="GPS/IMU dashcam-style video to survey."),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output survey run directory. Defaults to runs/<video-basename>.",
    ),
    gps_sidecar: Path | None = typer.Option(
        None,
        "--gps-sidecar",
        help="Explicit GPS sidecar (.track.json, .gpx, or DJI .srt).",
    ),
    gps_source: str = typer.Option(
        "auto",
        "--gps-source",
        help="GPS source mode: auto, embedded, sidecar, imu, or none.",
    ),
    fps: float = typer.Option(1.0, "--fps", help="Seek-sampled frame rate in frames per second."),
    clip_seconds: float | None = typer.Option(None, "--clip-seconds", help="Limit processing to the first N seconds."),
    quality_threshold: int = typer.Option(
        4,
        "--quality-threshold",
        help="Save a frame as a problem when quality grade is this value or worse.",
    ),
    crack_prob: float = typer.Option(
        0.6,
        "--crack-prob",
        help="Minimum tile crack classifier probability before dense-mask confirmation is allowed.",
    ),
    min_crack_area: float = typer.Option(
        0.3,
        "--min-crack-area",
        help="Minimum dense crack mask area percentage required to confirm a crack.",
    ),
    min_crack_length_px: int = typer.Option(
        64,
        "--min-crack-length-px",
        help="Minimum longest connected crack component length in pixels.",
    ),
    device: str = typer.Option("cpu", "--device", help="DINOv3 inference device: cpu, mps, or auto."),
) -> None:
    """Survey a GPS/IMU video and map DINOv3-recognized road problems."""
    from tarmac.survey.survey import print_survey_summary, run_survey

    summary = run_survey(
        video_path=video,
        out_dir=out,
        fps=fps,
        clip_seconds=clip_seconds,
        quality_threshold=quality_threshold,
        crack_prob=crack_prob,
        min_crack_area=min_crack_area,
        min_crack_length_px=min_crack_length_px,
        device=device,
        gps_sidecar=gps_sidecar,
        gps_source=gps_source,
    )
    print_survey_summary(summary, console)


@app.command("survey-confirm")
def survey_confirm_cmd(
    run_dir: Path = typer.Argument(..., help="Existing survey run directory with saved problem_images/."),
    crack_prob: float = typer.Option(
        0.6,
        "--crack-prob",
        help="Minimum tile crack classifier probability before dense-mask confirmation is allowed.",
    ),
    min_crack_area: float = typer.Option(
        0.3,
        "--min-crack-area",
        help="Minimum dense crack mask area percentage required to confirm a crack.",
    ),
    min_crack_length_px: int = typer.Option(
        64,
        "--min-crack-length-px",
        help="Minimum longest connected crack component length in pixels.",
    ),
    quality_threshold: int | None = typer.Option(
        None,
        "--quality-threshold",
        help="Quality threshold to preserve non-crack problem frames. Defaults to the run summary value.",
    ),
    batch_size: int = typer.Option(8, "--batch-size", help="Embedding batch size for saved image recheck."),
    device: str = typer.Option("cpu", "--device", help="DINOv3 inference device: cpu, mps, or auto."),
) -> None:
    """Re-check saved survey problem images with dense crack confirmation, without reading the video."""
    from tarmac.survey.survey import confirm_survey_problems

    summary = confirm_survey_problems(
        run_dir,
        crack_prob=crack_prob,
        min_crack_area=min_crack_area,
        min_crack_length_px=min_crack_length_px,
        quality_threshold=quality_threshold,
        device=device,
        batch_size=batch_size,
        rebuild_reports=True,
    )
    before = int(summary.get("problems_before_confirmation", summary.get("original_problems_found", 0)))
    after = int(summary.get("problems_after_confirmation", summary.get("problems_found", 0)))
    before_cracks = int(summary.get("crack_count_before_confirmation", 0))
    after_cracks = int(summary.get("crack_count_after_confirmation", 0))
    console.print(
        f"Survey confirmation complete: problems {before}->{after}; cracks {before_cracks}->{after_cracks}"
    )
    console.print(f"Confirmed issue counts: {summary.get('confirmed_problem_issue_counts', {})}")


@app.command("strip-view")
def strip_view_cmd(
    run_dir: Path = typer.Argument(..., help="Existing survey run directory with frames/ and samples.parquet."),
    band_frac: float = typer.Option(
        0.5,
        "--band-frac",
        help="Fraction of each frame height to use for the lower road band.",
    ),
    ribbon_width: int = typer.Option(
        512,
        "--ribbon-width",
        help="Output ribbon width in pixels before LOD downsampling.",
    ),
) -> None:
    """Build a continuous tiled canvas strip viewer for a survey run."""
    from tarmac.survey.strip import build_strip_view

    result = build_strip_view(run_dir, band_frac=band_frac, ribbon_width=ribbon_width)
    table = Table(title="Continuous Strip Viewer")
    table.add_column("LOD")
    table.add_column("Dimensions")
    table.add_column("Tiles", justify="right")
    for lod in result.lods:
        table.add_row(
            f"z{lod['level']}",
            f"{lod['width']}x{lod['height']}",
            str(lod["tile_count"]),
        )
    console.print(table)
    console.print(f"Ribbon: {result.ribbon_width}x{result.ribbon_height}")
    console.print(f"Viewer: {result.html_path}")
    console.print(f"Manifest: {result.manifest_path}")


@app.command("crack-measure")
def crack_measure(
    path: Path = typer.Argument(..., help="Image file or directory of images to measure."),
    mm_per_pixel: float | None = typer.Option(
        None,
        "--mm-per-pixel",
        help="Optional calibration for metric crack measurements.",
    ),
    out: Path = typer.Option(Path("runs/crack_measure"), "--out", "-o", help="Output directory."),
    batch_size: int = typer.Option(32, "--batch-size", help="Sliding-window embedding batch size."),
    device: str = typer.Option("cpu", "--device", help="Inference device: cpu, mps, or auto."),
) -> None:
    """Measure crack area, length, and width on full images."""
    import pandas as pd
    from PIL import Image

    from tarmac.crack.segment import segment_cracks
    from tarmac.embedding.embedder import HFBackboneEmbedder
    from tarmac.inference.analyze import IMAGE_EXTENSIONS, load_active_artifacts, load_crack_detector

    path = path.expanduser().resolve()
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        image_paths = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        image_paths = [path]
    else:
        raise typer.BadParameter(f"Expected an image file or image directory: {path}")
    if not image_paths:
        raise typer.BadParameter(f"No images found in {path}")

    learned_checkpoint = Path("models/crack_seg_head.pt")
    crack_detector = None if learned_checkpoint.exists() else load_crack_detector()
    embedder = None
    if crack_detector is not None:
        active = load_active_artifacts()
        embedder = HFBackboneEmbedder(
            model_name=active.model_name,
            checkpoint_path=active.checkpoint_path,
            allow_fallback=False,
            device_name=device,
            attn_implementation="eager",
        )

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            overlay = out / f"{image_path.stem}_crackseg.png"
            result = segment_cracks(
                image,
                crack_head=crack_detector,
                embedder=embedder,
                mm_per_pixel=mm_per_pixel,
                output_path=overlay,
                batch_size=batch_size,
                device_name=device,
            )
        row = {
            "image_path": str(image_path),
            "filename": image_path.name,
            "overlay_path": str(overlay),
            "segmenter": result.segmenter,
            **result.measurements,
        }
        rows.append(row)
        console.print(
            f"{index + 1:02d}. {image_path.name}: "
            f"segmenter={result.segmenter} "
            f"area={float(result.measurements['crack_area_pct']):.4f}% "
            f"length={int(result.measurements['total_length_px'])} px"
        )

    frame = pd.DataFrame(rows)
    csv_path = out / "crack_measurements.csv"
    parquet_path = out / "crack_measurements.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    console.print(f"Crack measurements written to {csv_path} and {parquet_path}")


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


def _print_defect_metrics(result: dict) -> None:
    for split in ("val", "test"):
        label_table = Table(title=f"Defect {split} per-label metrics")
        label_table.add_column("Label")
        label_table.add_column("Precision", justify="right")
        label_table.add_column("Recall", justify="right")
        label_table.add_column("F1", justify="right")
        label_table.add_column("AP", justify="right")
        label_table.add_column("Support", justify="right")
        for label, metrics in result[split]["per_label"].items():
            label_table.add_row(
                label,
                f"{float(metrics['precision']):.4f}",
                f"{float(metrics['recall']):.4f}",
                f"{float(metrics['f1']):.4f}",
                f"{float(metrics['ap']):.4f}",
                str(metrics["support"]),
            )
        console.print(label_table)

        domain_table = Table(title=f"Defect {split} per-domain metrics")
        domain_table.add_column("Domain")
        domain_table.add_column("Rows", justify="right")
        domain_table.add_column("Labels", justify="right")
        domain_table.add_column("Macro F1", justify="right")
        domain_table.add_column("Micro F1", justify="right")
        domain_table.add_column("Macro AP", justify="right")
        for domain, metrics in result[split]["per_domain"].items():
            domain_table.add_row(
                domain,
                str(metrics["rows"]),
                str(metrics.get("macro_label_count", "")),
                f"{float(metrics['macro_f1']):.4f}",
                f"{float(metrics['micro_f1']):.4f}",
                f"{float(metrics['macro_ap']):.4f}",
            )
        console.print(domain_table)


if __name__ == "__main__":
    app()
