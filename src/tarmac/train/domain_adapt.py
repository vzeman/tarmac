from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from tarmac.cluster.cluster import run_clustering
from tarmac.embedding.embedder import DINOV3_MODEL, HFBackboneEmbedder, embed_manifest
from tarmac.embedding.tiling import make_embedding_inputs
from tarmac.eval.evaluate import run_evaluation
from tarmac.inference.analyze import load_active_artifacts, load_reference_embeddings
from tarmac.train.supcon import _freeze_except_last_blocks, _select_device, _seed_everything, train_supcon

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 42
DEFAULT_VIDEO_PATHS = [
    Path("/Users/viktorzeman/Downloads/RoadSurveyRecorder/rs_20260614_142351975Z/rs_20260614_142351975Z_seg001.mp4"),
    Path("/Users/viktorzeman/Downloads/RoadSurveyRecorder/rs_20260614_143223000Z/rs_20260614_143223000Z_seg001.mp4"),
]


@dataclass(frozen=True)
class SimSiamConfig:
    tile_manifest_path: str
    initial_checkpoint: str
    output_checkpoint: str
    output_metadata: str
    model_name: str
    epochs: int
    batch_size: int
    effective_batch_size: int
    accumulation_steps: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    unfrozen_blocks: int
    max_tiles: int
    seed: int
    device: str
    attn_implementation: str
    checkpoint_dir: str


class SimSiamTileDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        max_attempts = min(10, len(self.frame))
        for offset in range(max_attempts):
            candidate_index = (index + offset) % len(self.frame)
            row = self.frame.iloc[candidate_index]
            try:
                with Image.open(row["image_path"]) as image:
                    rgb = image.convert("RGB")
                    view_a = self.transform(rgb)
                    view_b = self.transform(rgb)
                return {"view_a": view_a, "view_b": view_b}
            except Exception:
                continue

        gray = Image.new("RGB", (224, 224), color=(128, 128, 128))
        view_a = self.transform(gray)
        view_b = self.transform(gray)
        return {"view_a": view_a, "view_b": view_b}


class TileInferenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, processor: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ][0]
        return {"index": int(index), "pixel_values": pixel_values}


class SimSiamProjector(nn.Module):
    def __init__(self, hidden_size: int, projection_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


class SimSiamPredictor(nn.Module):
    def __init__(self, projection_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        return self.net(projections)


def extract_video_frames(
    video_paths: list[Path],
    *,
    output_dir: Path = Path("data/raw/video_frames"),
    manifest_path: Path = Path("data/processed/video_frames_manifest.parquet"),
    fps: float = 2.0,
    force: bool = False,
) -> pd.DataFrame:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for video frame extraction.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for video_path in video_paths:
        source = video_path.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Video does not exist: {source}")
        session = _session_slug(source)
        frame_dir = output_dir / session / "frames"
        if force and frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(frame_dir.glob("frame_*.jpg"))
        if not existing:
            pattern = frame_dir / "frame_%06d.jpg"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                f"fps={float(fps)}",
                "-q:v",
                "2",
                "-start_number",
                "0",
                "-y",
                str(pattern),
            ]
            subprocess.run(cmd, check=True)
            existing = sorted(frame_dir.glob("frame_*.jpg"))
        for index, frame_path in enumerate(existing):
            rows.append(
                {
                    "session": session,
                    "video_path": str(source),
                    "frame_index": int(index),
                    "timestamp_s": float(index / fps),
                    "image_path": str(frame_path),
                }
            )
    frame = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(manifest_path, index=False)
    return frame


def build_road_tile_manifest(
    frames: pd.DataFrame,
    *,
    output_dir: Path = Path("data/raw/video_frames"),
    manifest_path: Path = Path("data/processed/video_road_tiles_manifest.parquet"),
    input_size: int = 224,
    region: str = "lower_half",
    force: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in tqdm(frames.itertuples(index=False), total=len(frames), desc="Tiling frames", unit="frame"):
        frame_path = Path(str(row.image_path))
        session = str(row.session)
        tile_dir = output_dir / session / "tiles"
        if force and tile_dir.exists() and int(row.frame_index) == 0:
            shutil.rmtree(tile_dir)
        tile_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(frame_path) as image:
            inputs = make_embedding_inputs(image, input_size=input_size, region=region)
        for item in inputs:
            if item.kind == "full":
                continue
            tile_index = int(item.kind.split("_")[-1])
            tile_path = tile_dir / f"frame_{int(row.frame_index):06d}_{item.kind}.jpg"
            if force or not tile_path.exists():
                item.image.save(tile_path, quality=95)
            rows.append(
                {
                    "image_path": str(tile_path),
                    "frame_path": str(frame_path),
                    "session": session,
                    "video_path": str(row.video_path),
                    "frame_index": int(row.frame_index),
                    "timestamp_s": float(row.timestamp_s),
                    "tile": item.kind,
                    "tile_index": tile_index,
                    "region": region,
                }
            )
    tile_frame = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tile_frame.to_parquet(manifest_path, index=False)
    return tile_frame


def train_simsiam_domain_adapt(
    tile_manifest_path: Path,
    *,
    initial_checkpoint: Path | None = None,
    output_checkpoint: Path = Path("models/checkpoints/domain_adapt/domain_adapted.pt"),
    output_metadata: Path = Path("models/checkpoints/domain_adapt/domain_adapted.json"),
    checkpoint_dir: Path = Path("models/checkpoints/domain_adapt"),
    model_name: str = DINOV3_MODEL,
    epochs: int = 3,
    batch_size: int = 8,
    effective_batch_size: int = 64,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    unfrozen_blocks: int = 2,
    max_tiles: int = 6000,
    seed: int = SEED,
    num_workers: int = 0,
    device_name: str = "mps",
    attn_implementation: str = "eager",
) -> dict[str, Any]:
    _seed_everything(seed)
    if initial_checkpoint is not None and not initial_checkpoint.exists():
        raise FileNotFoundError(f"Initial checkpoint does not exist: {initial_checkpoint}")
    tile_frame = pd.read_parquet(tile_manifest_path)
    if tile_frame.empty:
        raise RuntimeError("No road tiles are available for self-supervised adaptation.")
    if len(tile_frame) > max_tiles:
        tile_frame = tile_frame.sample(n=max_tiles, random_state=seed).reset_index(drop=True)

    embedder = HFBackboneEmbedder(
        model_name=model_name,
        checkpoint_path=initial_checkpoint,
        allow_fallback=False,
        attn_implementation=attn_implementation,
        move_to_device=False,
    )
    model = embedder.model
    device = _select_device(embedder.device, device_name)
    if device.type != "mps":
        raise RuntimeError(f"Domain adaptation requires MPS; selected device was {device.type}.")
    embedder.device = device
    model.to(device)
    hidden_size = int(getattr(model.config, "hidden_size", 768))
    projector = SimSiamProjector(hidden_size).to(device)
    predictor = SimSiamPredictor().to(device)
    _freeze_except_last_blocks(model, last_n=unfrozen_blocks)

    transform = _simsiam_transform(embedder.processor, embedder.input_size)
    dataset = SimSiamTileDataset(tile_frame, transform)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        generator=generator,
    )
    if len(loader) == 0:
        raise RuntimeError("Self-supervised DataLoader is empty; lower the batch size or add tiles.")
    trainable_backbone = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        [
            {"params": trainable_backbone, "lr": backbone_lr},
            {"params": list(projector.parameters()) + list(predictor.parameters()), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    accumulation_steps = max(1, math.ceil(effective_batch_size / batch_size))
    total_steps = max(1, math.ceil(len(loader) / accumulation_steps) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    config = SimSiamConfig(
        tile_manifest_path=str(tile_manifest_path),
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint is not None else "",
        output_checkpoint=str(output_checkpoint),
        output_metadata=str(output_metadata),
        model_name=embedder.model_name,
        epochs=epochs,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        accumulation_steps=accumulation_steps,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        unfrozen_blocks=unfrozen_blocks,
        max_tiles=max_tiles,
        seed=seed,
        device=device.type,
        attn_implementation=attn_implementation,
        checkpoint_dir=str(checkpoint_dir),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int | str]] = []

    Console().print(
        f"SimSiam domain adaptation on {device.type}; tiles={len(tile_frame)} "
        f"epochs={epochs} unfrozen_blocks={unfrozen_blocks}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        projector.train()
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(loader, desc=f"SimSiam epoch {epoch}/{epochs}", unit="batch")
        for batch_index, batch in enumerate(progress, start=1):
            loss = _simsiam_forward_loss(
                model=model,
                projector=projector,
                predictor=predictor,
                view_a=batch["view_a"],
                view_b=batch["view_b"],
                device=device,
            )
            (loss / accumulation_steps).backward()
            _assert_finite_trainable_gradients(model, "backbone")
            _assert_finite_trainable_gradients(projector, "projector")
            _assert_finite_trainable_gradients(predictor, "predictor")
            running_loss += float(loss.detach().cpu())
            if batch_index % accumulation_steps == 0 or batch_index == len(loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{running_loss / batch_index:.4f}")
        epoch_loss = running_loss / max(1, len(loader))
        history.append({"epoch": epoch, "loss": epoch_loss, "device": device.type})
        epoch_checkpoint = checkpoint_dir / f"epoch_{epoch}.pt"
        _save_simsiam_checkpoint(
            epoch_checkpoint,
            model=model,
            projector=projector,
            predictor=predictor,
            optimizer=optimizer,
            scheduler=scheduler,
            embedder=embedder,
            config=config,
            history=history,
            epoch=epoch,
        )
        shutil.copy2(epoch_checkpoint, output_checkpoint)
        output_metadata.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "history": history,
                    "checkpoint": str(output_checkpoint),
                    "latest_epoch_checkpoint": str(epoch_checkpoint),
                    "ssl_method": "SimSiam negative-cosine self-distillation on two strong augmentations of the same road tile.",
                },
                indent=2,
            )
            + "\n"
        )
    return {
        "checkpoint": str(output_checkpoint),
        "metadata": str(output_metadata),
        "epochs_trained": len(history),
        "history": history,
        "tiles_used": int(len(tile_frame)),
        "method": "SimSiam",
    }


def pseudo_label_tiles(
    tile_manifest_path: Path,
    *,
    output_manifest: Path = Path("data/processed/domain_adapt_pseudo_labels.parquet"),
    all_scores_path: Path = Path("data/processed/domain_adapt_pseudo_scores.parquet"),
    active_model_path: Path = Path("models/active_model.json"),
    k: int = 10,
    min_mean_cosine: float = 0.8,
    min_surface_margin: float = 0.10,
    min_quality_margin: float = 0.08,
    max_total: int = 3000,
    max_per_composite: int = 150,
    batch_size: int = 32,
    num_workers: int = 0,
    device_name: str = "mps",
) -> dict[str, Any]:
    artifacts = load_active_artifacts(active_model_path)
    ref_df, _ref_embeddings = load_reference_embeddings(artifacts.embeddings_path)
    import faiss

    try:
        faiss.omp_set_num_threads(1)
    except AttributeError:
        pass
    index = faiss.read_index(str(artifacts.faiss_index_path))
    embedder = HFBackboneEmbedder(
        model_name=artifacts.model_name,
        checkpoint_path=artifacts.checkpoint_path,
        allow_fallback=False,
        attn_implementation="eager",
        device_name=device_name,
    )
    if embedder.device.type != "mps":
        raise RuntimeError(f"Pseudo-labeling requires MPS; selected device was {embedder.device.type}.")

    tile_frame = pd.read_parquet(tile_manifest_path).reset_index(drop=True)
    dataset = TileInferenceDataset(tile_frame, embedder.processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scored_rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Pseudo-labeling tiles", unit="batch"):
        embeddings = embedder.embed_pixel_values(batch["pixel_values"]).numpy().astype("float32")
        distances, indices = index.search(embeddings, k)
        for local_index, sims, neighbor_idx in zip(batch["index"], distances, indices, strict=True):
            source = tile_frame.iloc[int(local_index)].to_dict()
            neighbors = ref_df.iloc[neighbor_idx]
            weights = np.maximum(sims.astype("float32"), 0.0) + 1e-6
            surface_type, surface_margin, surface_scores = _top_label_and_margin(
                neighbors["surface_type"].astype(str).tolist(), weights
            )
            quality_label, quality_margin, quality_scores = _top_label_and_margin(
                [str(value) for value in neighbors["quality"].astype(int).tolist()], weights
            )
            mean_cosine = float(np.mean(sims))
            accepted = bool(
                mean_cosine >= min_mean_cosine
                and surface_margin >= min_surface_margin
                and quality_margin >= min_quality_margin
            )
            scored_rows.append(
                {
                    **source,
                    "predicted_surface_type": surface_type,
                    "predicted_quality": int(quality_label),
                    "mean_neighbor_cosine": mean_cosine,
                    "surface_margin": float(surface_margin),
                    "quality_margin": float(quality_margin),
                    "surface_scores": json.dumps(surface_scores, sort_keys=True),
                    "quality_scores": json.dumps(quality_scores, sort_keys=True),
                    "nearest_neighbor": str(neighbors.iloc[0]["image_path"]),
                    "nearest_similarity": float(sims[0]),
                    "accepted": accepted,
                }
            )

    scored = pd.DataFrame(scored_rows)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    all_scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(all_scores_path, index=False)
    accepted = scored[scored["accepted"]].copy()
    if accepted.empty:
        pseudo = pd.DataFrame(
            columns=[
                "image_path",
                "source_dataset",
                "surface_type",
                "quality",
                "split",
                "mean_neighbor_cosine",
                "surface_margin",
                "quality_margin",
            ]
        )
    else:
        accepted["surface_type"] = accepted["predicted_surface_type"].astype(str)
        accepted["quality"] = accepted["predicted_quality"].astype("int64")
        accepted["source_dataset"] = "recorded_video_pseudo"
        accepted["split"] = "train"
        accepted["composite"] = accepted["surface_type"].astype(str) + "__q" + accepted["quality"].astype(str)
        accepted = accepted.sort_values(
            ["mean_neighbor_cosine", "surface_margin", "quality_margin"],
            ascending=False,
        )
        capped = accepted.groupby("composite", group_keys=False).head(max_per_composite)
        selected = _balanced_select(capped, group_column="composite", max_total=max_total)
        pseudo = selected[
            [
                "image_path",
                "source_dataset",
                "surface_type",
                "quality",
                "split",
                "mean_neighbor_cosine",
                "surface_margin",
                "quality_margin",
                "nearest_neighbor",
                "nearest_similarity",
                "session",
                "frame_index",
                "tile",
            ]
        ].copy()
    pseudo.to_parquet(output_manifest, index=False)
    counts = _pseudo_counts(pseudo)
    return {
        "pseudo_manifest": str(output_manifest),
        "all_scores": str(all_scores_path),
        "tiles_scored": int(len(scored)),
        "accepted_before_balance": int(len(accepted)),
        "selected": int(len(pseudo)),
        "thresholds": {
            "k": int(k),
            "min_mean_cosine": float(min_mean_cosine),
            "min_surface_margin": float(min_surface_margin),
            "min_quality_margin": float(min_quality_margin),
            "max_total": int(max_total),
            "max_per_composite": int(max_per_composite),
        },
        "counts": counts,
    }


def build_augmented_manifest(
    base_manifest_path: Path,
    pseudo_manifest_path: Path,
    *,
    output_path: Path = Path("data/processed/domain_adapt_manifest.parquet"),
) -> dict[str, Any]:
    base = pd.read_parquet(base_manifest_path)
    pseudo = pd.read_parquet(pseudo_manifest_path)
    columns = ["image_path", "source_dataset", "surface_type", "quality", "split"]
    combined = pd.concat([base[columns], pseudo[columns]], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return {
        "manifest": str(output_path),
        "base_rows": int(len(base)),
        "pseudo_rows": int(len(pseudo)),
        "combined_rows": int(len(combined)),
    }


def run_domain_adaptation_pipeline(
    *,
    video_paths: list[Path] | None = None,
    fps: float = 2.0,
    base_manifest_path: Path = Path("data/processed/manifest.parquet"),
    active_model_path: Path = Path("models/active_model.json"),
    current_metrics_path: Path = Path("reports/phase2_metrics_dinov3_finetuned.json"),
    report_path: Path = Path("reports/DOMAIN_ADAPT.md"),
    force_extract: bool = False,
    ssl_epochs: int = 3,
    ssl_batch_size: int = 8,
    ssl_effective_batch_size: int = 64,
    ssl_max_tiles: int = 6000,
    ssl_unfrozen_blocks: int = 2,
    pseudo_min_mean_cosine: float = 0.8,
    pseudo_min_surface_margin: float = 0.10,
    pseudo_min_quality_margin: float = 0.08,
    pseudo_max_total: int = 3000,
    pseudo_max_per_composite: int = 150,
    finetune_epochs: int = 8,
    finetune_batch_size: int = 16,
    finetune_effective_batch_size: int = 128,
    seed: int = SEED,
    device_name: str = "mps",
) -> dict[str, Any]:
    console = Console()
    _seed_everything(seed)
    _require_mps(device_name)
    video_paths = video_paths or DEFAULT_VIDEO_PATHS
    active_config = json.loads(active_model_path.read_text())
    active_checkpoint = Path(str(active_config.get("checkpoint", "models/finetuned_dinov3.pt")))

    frames = extract_video_frames(video_paths, fps=fps, force=force_extract)
    tiles = build_road_tile_manifest(frames, force=force_extract)
    console.print(f"Extracted/loaded {len(frames)} frames and {len(tiles)} road tiles.")

    ssl_result = train_simsiam_domain_adapt(
        Path("data/processed/video_road_tiles_manifest.parquet"),
        initial_checkpoint=active_checkpoint,
        epochs=ssl_epochs,
        batch_size=ssl_batch_size,
        effective_batch_size=ssl_effective_batch_size,
        max_tiles=ssl_max_tiles,
        unfrozen_blocks=ssl_unfrozen_blocks,
        seed=seed,
        device_name=device_name,
    )

    pseudo_result = pseudo_label_tiles(
        Path("data/processed/video_road_tiles_manifest.parquet"),
        active_model_path=active_model_path,
        min_mean_cosine=pseudo_min_mean_cosine,
        min_surface_margin=pseudo_min_surface_margin,
        min_quality_margin=pseudo_min_quality_margin,
        max_total=pseudo_max_total,
        max_per_composite=pseudo_max_per_composite,
        device_name=device_name,
    )
    augmented = build_augmented_manifest(
        base_manifest_path,
        Path(pseudo_result["pseudo_manifest"]),
    )

    candidate_checkpoint = Path("models/domain_adapt_finetuned.pt")
    candidate_metadata = Path("models/domain_adapt_finetuned.json")
    finetune_result = train_supcon(
        manifest_path=Path(augmented["manifest"]),
        output_checkpoint=candidate_checkpoint,
        output_metadata=candidate_metadata,
        model_name=DINOV3_MODEL,
        initial_checkpoint=Path(ssl_result["checkpoint"]),
        epochs=finetune_epochs,
        batch_size=finetune_batch_size,
        effective_batch_size=finetune_effective_batch_size,
        backbone_lr=5e-5,
        head_lr=5e-4,
        unfrozen_blocks=4,
        device_name=device_name,
        run_name="domain_adapt_supcon",
        patience=3,
        attn_implementation="eager",
    )

    suffix = "dinov3_domain_adapt"
    embed_info = embed_manifest(
        manifest_path=base_manifest_path,
        output_path=Path(f"data/processed/embeddings_{suffix}.parquet"),
        faiss_index_path=Path(f"models/faiss_full_{suffix}.index"),
        metadata_path=Path(f"models/embedding_metadata_{suffix}.json"),
        model_name=DINOV3_MODEL,
        checkpoint_path=candidate_checkpoint,
        allow_fallback=False,
        attn_implementation="eager",
        device_name=device_name,
        batch_size=16,
        num_workers=0,
    )
    if embed_info.device != "mps":
        raise RuntimeError(f"Candidate embedding did not run on MPS: {embed_info.device}")

    cluster_meta = run_clustering(
        embeddings_path=Path(f"data/processed/embeddings_{suffix}.parquet"),
        centroids_path=Path(f"models/kmeans_centroids_{suffix}.npy"),
        assignments_path=Path(f"data/processed/cluster_assignments_{suffix}.parquet"),
        profile_path=Path(f"reports/cluster_profile_{suffix}.csv"),
        metadata_path=Path(f"models/cluster_metadata_{suffix}.json"),
    )
    candidate_metrics_path = Path("reports/domain_adapt_metrics.json")
    candidate_metrics = run_evaluation(
        embeddings_path=Path(f"data/processed/embeddings_{suffix}.parquet"),
        assignments_path=Path(f"data/processed/cluster_assignments_{suffix}.parquet"),
        embed_metadata_path=Path(f"models/embedding_metadata_{suffix}.json"),
        cluster_metadata_path=Path(f"models/cluster_metadata_{suffix}.json"),
        metrics_path=candidate_metrics_path,
        report_path=Path("reports/PHASE2_BASELINE_dinov3_domain_adapt.md"),
        umap_html_path=Path("reports/umap_scatter_dinov3_domain_adapt.html"),
        umap_png_path=Path("reports/umap_quality_dinov3_domain_adapt.png"),
    )
    current_metrics = _read_json(current_metrics_path)
    gate = _gate_candidate(
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        max_type_accuracy_regression=0.005,
    )
    active_after = dict(active_config)
    if gate["accepted"]:
        active_after.update(
            {
                "backbone": "dinov3",
                "model_name": DINOV3_MODEL,
                "checkpoint": str(candidate_checkpoint),
                "suffix": suffix,
            }
        )
        active_model_path.write_text(json.dumps(active_after, indent=2) + "\n")

    report = _domain_adapt_report(
        frames=frames,
        tiles=tiles,
        ssl_result=ssl_result,
        pseudo_result=pseudo_result,
        augmented=augmented,
        finetune_result=finetune_result,
        cluster_meta=cluster_meta,
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        gate=gate,
        active_before=active_config,
        active_after=active_after,
        suffix=suffix,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "frames_extracted": int(len(frames)),
        "tiles_extracted": int(len(tiles)),
        "ssl": ssl_result,
        "pseudo": pseudo_result,
        "augmented": augmented,
        "finetune": finetune_result,
        "candidate_metrics": candidate_metrics,
        "current_metrics": current_metrics,
        "gate": gate,
        "active_model": json.loads(active_model_path.read_text()),
        "report": str(report_path),
    }


def _simsiam_transform(processor: Any, input_size: int) -> transforms.Compose:
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.35, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.45, 0.45, 0.35, 0.08)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.35),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _simsiam_forward_loss(
    *,
    model: nn.Module,
    projector: SimSiamProjector,
    predictor: SimSiamPredictor,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    values = torch.cat([view_a, view_b], dim=0).to(device)
    try:
        outputs = model(pixel_values=values)
    except RuntimeError as exc:
        raise RuntimeError(f"MPS SimSiam forward failed without CPU fallback: {exc}") from exc
    cls = outputs.last_hidden_state[:, 0, :].float()
    if not torch.isfinite(cls).all():
        raise RuntimeError("Non-finite SimSiam backbone CLS embedding on MPS.")
    z = projector(cls)
    if not torch.isfinite(z).all():
        raise RuntimeError("Non-finite SimSiam projector output on MPS.")
    p = predictor(z)
    if not torch.isfinite(p).all():
        raise RuntimeError("Non-finite SimSiam predictor output on MPS.")
    p_a, p_b = p.chunk(2, dim=0)
    z_a, z_b = z.chunk(2, dim=0)
    loss = 0.5 * (_negative_cosine(p_a, z_b.detach()) + _negative_cosine(p_b, z_a.detach()))
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite SimSiam loss on MPS.")
    return loss


def _negative_cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    return -(prediction * target).sum(dim=1).mean()


def _assert_finite_trainable_gradients(model: nn.Module, module_name: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient in {module_name}.{name} on MPS.")


def _save_simsiam_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    projector: SimSiamProjector,
    predictor: SimSiamPredictor,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    embedder: HFBackboneEmbedder,
    config: SimSiamConfig,
    history: list[dict[str, float | int | str]],
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "requested_model": embedder.requested_model,
            "model_name": embedder.model_name,
            "backbone": embedder.backbone,
            "embedding_dim": int(getattr(model.config, "hidden_size", 768)),
            "config": asdict(config),
            "history": history,
            "epoch": epoch,
        },
        path,
    )


def _top_label_and_margin(labels: list[str], weights: np.ndarray) -> tuple[str, float, dict[str, float]]:
    scores: dict[str, float] = {}
    total = float(np.sum(weights))
    for label, weight in zip(labels, weights, strict=True):
        scores[str(label)] = scores.get(str(label), 0.0) + float(weight)
    probabilities = {label: value / max(total, 1e-12) for label, value in scores.items()}
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return top_label, float(top_score - second_score), probabilities


def _balanced_select(frame: pd.DataFrame, *, group_column: str, max_total: int) -> pd.DataFrame:
    groups = {
        key: group.reset_index(drop=True)
        for key, group in frame.groupby(group_column, sort=True)
    }
    offsets = {key: 0 for key in groups}
    selected: list[pd.Series] = []
    while len(selected) < max_total:
        made_progress = False
        for key in sorted(groups):
            offset = offsets[key]
            group = groups[key]
            if offset >= len(group):
                continue
            selected.append(group.iloc[offset])
            offsets[key] = offset + 1
            made_progress = True
            if len(selected) >= max_total:
                break
        if not made_progress:
            break
    if not selected:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def _pseudo_counts(pseudo: pd.DataFrame) -> dict[str, Any]:
    if pseudo.empty:
        return {"surface_type": {}, "quality": {}, "surface_type_quality": {}}
    composite = pseudo["surface_type"].astype(str) + "__q" + pseudo["quality"].astype(str)
    return {
        "surface_type": {str(k): int(v) for k, v in Counter(pseudo["surface_type"]).items()},
        "quality": {str(k): int(v) for k, v in Counter(pseudo["quality"].astype(int)).items()},
        "surface_type_quality": {str(k): int(v) for k, v in Counter(composite).items()},
    }


def _gate_candidate(
    *,
    current_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    max_type_accuracy_regression: float,
) -> dict[str, Any]:
    current_quality_f1 = _metric(current_metrics, "knn", "quality", "val_test", "macro_f1")
    candidate_quality_f1 = _metric(candidate_metrics, "knn", "quality", "val_test", "macro_f1")
    current_type_acc = _metric(current_metrics, "knn", "surface_type", "val_test", "accuracy")
    candidate_type_acc = _metric(candidate_metrics, "knn", "surface_type", "val_test", "accuracy")
    quality_pass = candidate_quality_f1 > current_quality_f1
    type_pass = candidate_type_acc >= current_type_acc - max_type_accuracy_regression
    return {
        "accepted": bool(quality_pass and type_pass),
        "quality_macro_f1_pass": bool(quality_pass),
        "type_accuracy_pass": bool(type_pass),
        "current_quality_macro_f1": float(current_quality_f1),
        "candidate_quality_macro_f1": float(candidate_quality_f1),
        "current_type_accuracy": float(current_type_acc),
        "candidate_type_accuracy": float(candidate_type_acc),
        "max_type_accuracy_regression": float(max_type_accuracy_regression),
        "quality_macro_f1_delta": float(candidate_quality_f1 - current_quality_f1),
        "type_accuracy_delta": float(candidate_type_acc - current_type_acc),
    }


def _domain_adapt_report(
    *,
    frames: pd.DataFrame,
    tiles: pd.DataFrame,
    ssl_result: dict[str, Any],
    pseudo_result: dict[str, Any],
    augmented: dict[str, Any],
    finetune_result: dict[str, Any],
    cluster_meta: dict[str, Any],
    current_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    gate: dict[str, Any],
    active_before: dict[str, Any],
    active_after: dict[str, Any],
    suffix: str,
) -> str:
    rows = [
        "# Domain Adaptation On Recorded Road Frames",
        "",
        "## Data",
        "",
        f"- Frames extracted: {len(frames)}",
        f"- Road tiles extracted: {len(tiles)}",
        f"- Pseudo-labeled tiles selected: {pseudo_result['selected']} of {pseudo_result['tiles_scored']} scored",
        f"- Augmented manifest rows: {augmented['combined_rows']} ({augmented['base_rows']} StreetSurfaceVis + {augmented['pseudo_rows']} pseudo)",
        "",
        "## Self-Supervised Adaptation",
        "",
        "- Method: SimSiam negative-cosine self-distillation with two strong augmentations of each road tile.",
        f"- Epochs: {ssl_result['epochs_trained']}",
        f"- Tiles used for SSL: {ssl_result['tiles_used']}",
        "- Conservative settings: backbone LR 1e-5, last 2 DINOv3 blocks unfrozen, MPS/eager attention, no CPU fallback.",
        f"- Checkpoint: `{ssl_result['checkpoint']}`",
        "",
        "## Pseudo-Labeling",
        "",
        f"- Reference: current active fine-tuned model `{active_before.get('checkpoint')}` against StreetSurfaceVis cosine kNN.",
        f"- Thresholds: mean neighbor cosine >= {pseudo_result['thresholds']['min_mean_cosine']:.2f}, "
        f"surface margin >= {pseudo_result['thresholds']['min_surface_margin']:.2f}, "
        f"quality margin >= {pseudo_result['thresholds']['min_quality_margin']:.2f}.",
        f"- Balancing: max_total={pseudo_result['thresholds']['max_total']}, "
        f"max_per_surface_quality={pseudo_result['thresholds']['max_per_composite']}.",
        "",
        "### Pseudo-Label Counts",
        "",
        "| Group | Counts |",
        "|---|---|",
        f"| surface_type | `{json.dumps(pseudo_result['counts']['surface_type'], sort_keys=True)}` |",
        f"| quality | `{json.dumps(pseudo_result['counts']['quality'], sort_keys=True)}` |",
        f"| surface_type+quality | `{json.dumps(pseudo_result['counts']['surface_type_quality'], sort_keys=True)}` |",
        "",
        "## Held-Out StreetSurfaceVis Metrics",
        "",
        "| Metric | Current active | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = [
        ("surface_type val+test accuracy", ("knn", "surface_type", "val_test", "accuracy")),
        ("surface_type val+test macro-F1", ("knn", "surface_type", "val_test", "macro_f1")),
        ("quality val+test accuracy", ("knn", "quality", "val_test", "accuracy")),
        ("quality val+test macro-F1", ("knn", "quality", "val_test", "macro_f1")),
        ("quality val+test MAE", ("knn", "quality", "val_test", "mae")),
        ("quality val+test off-by-one", ("knn", "quality", "val_test", "off_by_one_accuracy")),
        ("silhouette surface_type", ("silhouette", "surface_type")),
        ("silhouette surface_type+quality", ("silhouette", "surface_type_quality")),
    ]
    for label, path in metric_rows:
        current_value = _metric(current_metrics, *path)
        candidate_value = _metric(candidate_metrics, *path)
        rows.append(
            f"| {label} | {current_value:.4f} | {candidate_value:.4f} | {candidate_value - current_value:+.4f} |"
        )
    rows.extend(
        [
            "",
            "## Acceptance Gate",
            "",
            f"- Required: quality macro-F1 must improve above {gate['current_quality_macro_f1']:.4f}.",
            f"- Required: type accuracy must stay within {gate['max_type_accuracy_regression']:.4f} of "
            f"{gate['current_type_accuracy']:.4f}.",
            f"- Quality macro-F1 pass: {_yes_no(bool(gate['quality_macro_f1_pass']))} "
            f"({gate['candidate_quality_macro_f1']:.4f}, delta {gate['quality_macro_f1_delta']:+.4f}).",
            f"- Type accuracy pass: {_yes_no(bool(gate['type_accuracy_pass']))} "
            f"({gate['candidate_type_accuracy']:.4f}, delta {gate['type_accuracy_delta']:+.4f}).",
            f"- Gate verdict: **{'ACCEPTED' if gate['accepted'] else 'REJECTED'}**.",
            "",
            "## Active Model",
            "",
            f"- Before: `{active_before.get('checkpoint')}`",
            f"- After: `{active_after.get('checkpoint')}`",
            f"- Candidate suffix/artifacts: `{suffix}`",
            f"- Candidate best SupCon epoch: {finetune_result['best_epoch']}",
            f"- Candidate clustering k: {cluster_meta.get('chosen_k')}",
        ]
    )
    if not gate["accepted"]:
        rows.append("- The active model was not replaced because the strict gate did not pass.")
    else:
        rows.append("- The active model was replaced because the strict gate passed.")
    return "\n".join(rows) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")
    return json.loads(path.read_text())


def _metric(metrics: dict[str, Any], *path: str) -> float:
    value: Any = metrics
    for key in path:
        value = value[key]
    return float(value)


def _require_mps(device_name: str) -> None:
    if device_name != "mps":
        raise RuntimeError("This domain adaptation pipeline is configured for MPS only.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; refusing CPU fallback.")


def _session_slug(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
