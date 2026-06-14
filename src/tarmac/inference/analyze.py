from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import ExifTags, Image, ImageFile
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from tarmac.defect import (
    DEFECT_LABELS,
    DEFECT_LABEL_APPLICABILITY,
    infer_defect_domain,
    is_defect_label_applicable,
)
from tarmac.embedding.embedder import DINOV3_MODEL, HFBackboneEmbedder
from tarmac.crack.model import CrackHead
from tarmac.crack.segment import segment_cracks
from tarmac.embedding.tiling import make_embedding_inputs, tile_boxes

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class ActiveArtifacts:
    suffix: str
    model_name: str
    checkpoint_path: Path
    embeddings_path: Path
    faiss_index_path: Path
    centroids_path: Path
    metadata_path: Path


def analyze_path(
    input_path: Path,
    out_dir: Path | None = None,
    fps: float = 2.0,
    k: int = 10,
    non_road_threshold: float | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    region: str = "auto",
    crack_segmentation: bool = False,
    mm_per_pixel: float | None = None,
    defect_gating: bool = True,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    artifacts = load_active_artifacts()
    if out_dir is None:
        out_dir = Path("runs") / input_path.stem
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_dir / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    temp_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    if input_path.is_dir():
        frame_paths = sorted(
            p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        input_type = "directory"
    elif input_path.suffix.lower() in VIDEO_EXTENSIONS:
        ensure_ffmpeg()
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="tarmac_frames_")
        frame_paths = extract_video_frames(input_path, Path(temp_dir_ctx.name), fps=fps)
        input_type = "video"
    elif input_path.suffix.lower() in IMAGE_EXTENSIONS:
        frame_paths = [input_path]
        input_type = "photo"
    else:
        raise ValueError(f"Unsupported input type: {input_path}")

    if not frame_paths:
        raise ValueError(f"No analyzable images found in {input_path}")

    try:
        ref_df, ref_embeddings = load_reference_embeddings(artifacts.embeddings_path)
        centroids = np.load(artifacts.centroids_path).astype("float32")
        centroids = normalize_rows(centroids)
        embedder = HFBackboneEmbedder(
            model_name=artifacts.model_name,
            checkpoint_path=artifacts.checkpoint_path,
            allow_fallback=False,
            device_name=device,
            attn_implementation="eager",
        )
        import faiss

        try:
            faiss.omp_set_num_threads(1)
        except AttributeError:
            pass
        index = faiss.read_index(str(artifacts.faiss_index_path))
        if index.ntotal != len(ref_embeddings):
            raise RuntimeError(
                f"FAISS index rows ({index.ntotal}) do not match reference embeddings ({len(ref_embeddings)})"
            )
        if non_road_threshold is None:
            non_road_threshold = calibrate_non_road_threshold(artifacts.embeddings_path, index, k=k)
        crack_detector = load_crack_detector()
        learned_crack_segmenter_exists = Path("models/crack_seg_head.pt").exists()
        defect_detector = load_defect_detector()
        chosen_region = resolve_region_mode(
            requested_region=region,
            frame_paths=frame_paths,
            embedder=embedder,
            reference_df=ref_df,
            index=index,
            centroids=centroids,
            k=k,
            non_road_threshold=non_road_threshold,
        )
        Console().print(f"Region mode: {chosen_region} (requested: {region})")
        run_crack_segmentation = bool(
            crack_segmentation
            or (chosen_region == "full" and (crack_detector is not None or learned_crack_segmenter_exists))
        )
        rows, tile_rows = analyze_frames(
            frame_paths=frame_paths,
            input_type=input_type,
            out_dir=out_dir,
            thumbs_dir=thumbs_dir,
            embedder=embedder,
            reference_df=ref_df,
            index=index,
            centroids=centroids,
            k=k,
            non_road_threshold=non_road_threshold,
            batch_size=batch_size,
            crack_detector=crack_detector,
            defect_detector=defect_detector,
            region=chosen_region,
            crack_segmentation=run_crack_segmentation,
            mm_per_pixel=mm_per_pixel,
            defect_gating=defect_gating,
            device_name=device,
        )
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()

    frames_df = pd.DataFrame(rows)
    tiles_df = pd.DataFrame(tile_rows)
    results_path = out_dir / "results.parquet"
    tiles_path = out_dir / "tiles.parquet"
    frames_df.to_parquet(results_path, index=False)
    tiles_df.to_parquet(tiles_path, index=False)

    summary = build_summary(
        input_path=input_path,
        input_type=input_type,
        out_dir=out_dir,
        frames_df=frames_df,
        artifacts=artifacts,
        fps=fps,
        k=k,
        non_road_threshold=non_road_threshold,
        device=device,
        requested_region=region,
        region=chosen_region,
        crack_segmentation=run_crack_segmentation,
        defect_detector=defect_detector,
        defect_gating=defect_gating,
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_active_artifacts(active_model_path: Path = Path("models/active_model.json")) -> ActiveArtifacts:
    config = json.loads(active_model_path.read_text())
    backbone = str(config.get("backbone", "dinov3"))
    checkpoint = Path(config.get("checkpoint", "models/finetuned_dinov3.pt"))
    suffix = str(config.get("suffix") or f"{backbone}_finetuned")
    model_name = str(config.get("model_name") or DINOV3_MODEL)
    metadata_path = Path(f"models/embedding_metadata_{suffix}.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        model_name = str(metadata.get("model_name") or model_name)
    return ActiveArtifacts(
        suffix=suffix,
        model_name=model_name,
        checkpoint_path=checkpoint,
        embeddings_path=Path(f"data/processed/embeddings_{suffix}.parquet"),
        faiss_index_path=Path(f"models/faiss_full_{suffix}.index"),
        centroids_path=Path(f"models/kmeans_centroids_{suffix}.npy"),
        metadata_path=metadata_path,
    )


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for video analysis. Install it with: brew install ffmpeg")


def extract_video_frames(video_path: Path, frames_dir: Path, fps: float) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    return sorted(frames_dir.glob("frame_*.jpg"))


def load_reference_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    df = df[df["kind"] == "full"].reset_index(drop=True)
    embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")
    embeddings = normalize_rows(embeddings)
    return df.drop(columns=["embedding"]), embeddings


def calibrate_non_road_threshold(
    embeddings_path: Path,
    index: Any,
    k: int,
    target_pass_rate: float = 0.95,
    max_threshold: float = 0.45,
) -> float:
    df = pd.read_parquet(embeddings_path)
    tiles = df[(df["split"] == "val") & (df["kind"] != "full")]
    if tiles.empty:
        return max_threshold
    embeddings = np.vstack(tiles["embedding"].to_numpy()).astype("float32")
    embeddings = normalize_rows(embeddings)
    distances, _ = index.search(embeddings, k)
    mean_cosine = np.sort(distances.mean(axis=1))
    max_failures = max(0, int(np.ceil((1.0 - target_pass_rate) * len(mean_cosine))) - 1)
    threshold = float(mean_cosine[max_failures])
    return min(max_threshold, threshold)


def analyze_frames(
    frame_paths: list[Path],
    input_type: str,
    out_dir: Path,
    thumbs_dir: Path,
    embedder: HFBackboneEmbedder,
    reference_df: pd.DataFrame,
    index: Any,
    centroids: np.ndarray,
    k: int,
    non_road_threshold: float,
    batch_size: int,
    crack_detector: dict[str, Any] | None = None,
    defect_detector: dict[str, Any] | None = None,
    region: str = "lower_half",
    crack_segmentation: bool = False,
    mm_per_pixel: float | None = None,
    defect_gating: bool = True,
    device_name: str = "auto",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_records: list[dict[str, Any]] = []
    tile_records: list[dict[str, Any]] = []
    input_items: list[dict[str, Any]] = []

    for frame_index, path in enumerate(frame_paths):
        with Image.open(path) as image:
            exif = extract_exif(image)
            inputs = make_embedding_inputs(image, embedder.input_size, region=region)
            source_size = image.size
            thumb_name = f"frame_{frame_index:06d}.jpg"
            thumb_path = thumbs_dir / thumb_name
            save_thumbnail(image, thumb_path)
        for item in inputs:
            input_items.append(
                {
                    "frame_index": frame_index,
                    "source_path": str(path),
                    "filename": path.name,
                    "kind": item.kind,
                    "box": item.box,
                    "source_width": source_size[0],
                    "source_height": source_size[1],
                    "image": item.image,
                    "thumbnail_path": str(thumb_path.relative_to(out_dir)),
                    "timestamp": exif.get("timestamp"),
                    "latitude": exif.get("latitude"),
                    "longitude": exif.get("longitude"),
                }
            )

    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(input_items), batch_size), desc="Analyzing frames", unit="batch"):
        batch = input_items[start : start + batch_size]
        pixels = embedder.processor(images=[item["image"] for item in batch], return_tensors="pt")[
            "pixel_values"
        ]
        emb = embedder.embed_pixel_values(pixels).numpy().astype("float32")
        embeddings.extend(list(emb))

    by_frame: dict[int, dict[str, Any]] = {}
    for item, embedding in zip(input_items, embeddings, strict=True):
        frame = by_frame.setdefault(
            int(item["frame_index"]),
            {
                "frame_index": int(item["frame_index"]),
                "input_type": input_type,
                "source_path": item["source_path"],
                "filename": item["filename"],
                "thumbnail_path": item["thumbnail_path"],
                "timestamp": item["timestamp"],
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "tiles": [],
            },
        )
        if item["kind"] == "full":
            frame["embedding"] = embedding
            continue
        pred = predict_tile(
            embedding=embedding,
            reference_df=reference_df,
            index=index,
            centroids=centroids,
            k=k,
            threshold=non_road_threshold,
        )
        crack_prob = predict_crack_probability(embedding, crack_detector)
        crack_flag = bool(crack_prob >= float(crack_detector["threshold"])) if crack_detector else False
        defect_probs = predict_defect_probabilities(embedding, defect_detector)
        defect_domain = infer_defect_domain(
            source_path=str(item["source_path"]),
            filename=str(item["filename"]),
            surface_type=str(pred["surface_type"]),
        )
        tile_record = {
            "frame_index": int(item["frame_index"]),
            "tile": item["kind"],
            "tile_crack_prob": crack_prob,
            "tile_crack": crack_flag,
            "tile_box": list(item["box"]) if item["box"] is not None else None,
            "image_width": int(item["source_width"]),
            "image_height": int(item["source_height"]),
            "region": region,
            "defect_domain": defect_domain,
            "defect_gating_enabled": bool(defect_gating),
            **pred,
        }
        if defect_detector is not None:
            thresholds = defect_detector["thresholds"]
            for label in defect_detector["labels"]:
                raw_prob = float(defect_probs[label])
                applicable = True
                if defect_gating:
                    applicable = is_defect_label_applicable(
                        label=label,
                        surface_type=str(pred["surface_type"]),
                        domain=defect_domain,
                        source_path=str(item["source_path"]),
                        filename=str(item["filename"]),
                    )
                prob = raw_prob if applicable else 0.0
                tile_record[f"tile_defect_{label}_prob_raw"] = raw_prob
                tile_record[f"tile_defect_{label}_applicable"] = bool(applicable)
                tile_record[f"tile_defect_{label}_prob"] = prob
                tile_record[f"tile_defect_{label}"] = bool(
                    applicable and raw_prob >= float(thresholds[label])
                )
        frame["tiles"].append(tile_record)
        tile_records.append(tile_record)

    for frame_index in sorted(by_frame):
        frame = by_frame[frame_index]
        road_tiles = [tile for tile in frame["tiles"] if not tile["non_road"]]
        if road_tiles:
            qualities = [int(tile["predicted_quality"]) for tile in road_tiles]
            surfaces = [str(tile["surface_type"]) for tile in road_tiles]
            confidence = float(np.mean([float(tile["confidence"]) for tile in road_tiles]))
            predicted_quality = int(round(float(np.median(qualities))))
            surface_type = Counter(surfaces).most_common(1)[0][0]
            road_tile_count = len(road_tiles)
            crack_ratio = float(np.mean([bool(tile.get("tile_crack", False)) for tile in road_tiles]))
        else:
            predicted_quality = None
            surface_type = "non_road"
            confidence = 0.0
            road_tile_count = 0
            crack_ratio = 0.0
        defect_ratios: dict[str, float] = {}
        defect_types: list[str] = []
        if defect_detector is not None:
            for label in defect_detector["labels"]:
                ratio = (
                    float(np.mean([bool(tile.get(f"tile_defect_{label}", False)) for tile in road_tiles]))
                    if road_tiles
                    else 0.0
                )
                defect_ratios[label] = ratio
                if ratio > 0.0:
                    defect_types.append(label)
        record = {
            "frame_index": frame["frame_index"],
            "input_type": frame["input_type"],
            "source_path": frame["source_path"],
            "filename": frame["filename"],
            "thumbnail_path": frame["thumbnail_path"],
            "timestamp": frame["timestamp"],
            "latitude": frame["latitude"],
            "longitude": frame["longitude"],
            "predicted_quality": predicted_quality,
            "surface_type": surface_type,
            "confidence": confidence,
            "road_tile_count": road_tile_count,
            "tile_count": len(frame["tiles"]),
            "crack_ratio": crack_ratio,
            "frame_has_crack": bool(crack_ratio > 0.0),
            "tile_details": json.dumps(frame["tiles"]),
            "embedding": frame["embedding"].astype("float32"),
            "region": region,
            "defect_gating_enabled": bool(defect_gating),
        }
        if defect_detector is not None:
            for label, ratio in defect_ratios.items():
                record[f"defect_{label}_ratio"] = ratio
                record[f"frame_has_defect_{label}"] = bool(ratio > 0.0)
            record["structural_defects"] = json.dumps(defect_types)
            record["frame_has_structural_defect"] = bool(defect_types)
        if crack_segmentation:
            with Image.open(frame["source_path"]) as original:
                seg_dir = out_dir / "crackseg"
                overlay_name = f"frame_{frame_index:06d}_{Path(frame['filename']).stem}_crackseg.png"
                seg = segment_cracks(
                    original,
                    crack_head=crack_detector,
                    embedder=embedder,
                    mm_per_pixel=mm_per_pixel,
                    output_path=seg_dir / overlay_name,
                    batch_size=batch_size,
                    device_name=device_name,
                )
            measurements = seg.measurements
            record.update(
                {
                    "crackseg_overlay_path": str(Path("crackseg") / overlay_name),
                    "crack_segmenter": seg.segmenter,
                    "crack_area_px": int(measurements["crack_area_px"]),
                    "crack_area_pct": float(measurements["crack_area_pct"]),
                    "crack_length_px": int(measurements["total_length_px"]),
                    "crack_mean_width_px": float(measurements["mean_width_px"]),
                    "crack_max_width_px": float(measurements["max_width_px"]),
                    "crack_components": int(measurements["n_components"]),
                }
            )
            if "crack_area_mm2" in measurements:
                record.update(
                    {
                        "crack_area_mm2": float(measurements["crack_area_mm2"]),
                        "crack_length_mm": float(measurements["total_length_mm"]),
                        "crack_mean_width_mm": float(measurements["mean_width_mm"]),
                        "crack_max_width_mm": float(measurements["max_width_mm"]),
                    }
                )
        frame_records.append(record)
    return frame_records, tile_records


def load_crack_detector(
    checkpoint_path: Path = Path("models/crack_head.pt"),
    metrics_path: Path = Path("reports/crack_metrics.json"),
) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    state = torch.load(checkpoint_path, map_location="cpu")
    input_dim = int(state.get("input_dim", 768))
    head = CrackHead(input_dim=input_dim)
    head.load_state_dict(state["head_state_dict"])
    head.eval()
    threshold = 0.5
    if metrics_path.exists():
        try:
            threshold = float(json.loads(metrics_path.read_text()).get("threshold", threshold))
        except (json.JSONDecodeError, TypeError, ValueError):
            threshold = 0.5
    return {"head": head, "threshold": threshold, "checkpoint": str(checkpoint_path)}


def load_defect_detector(
    checkpoint_path: Path = Path("models/defect_head.pt"),
    metadata_path: Path = Path("models/defect_head.json"),
) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    from tarmac.defect.evaluate import load_defect_head

    head, thresholds_array, _metadata = load_defect_head(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
    )
    thresholds = {label: float(thresholds_array[index]) for index, label in enumerate(DEFECT_LABELS)}
    return {
        "head": head,
        "labels": DEFECT_LABELS,
        "thresholds": thresholds,
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path) if metadata_path.exists() else None,
    }


def resolve_region_mode(
    requested_region: str,
    frame_paths: list[Path],
    embedder: HFBackboneEmbedder,
    reference_df: pd.DataFrame,
    index: Any,
    centroids: np.ndarray,
    k: int,
    non_road_threshold: float,
    *,
    max_frames: int = 3,
) -> str:
    if requested_region in {"lower_half", "full"}:
        return requested_region
    if requested_region != "auto":
        raise ValueError("region must be one of: auto, lower_half, full")
    top_non_road: list[float] = []
    for path in frame_paths[:max_frames]:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            boxes = tile_boxes(*rgb.size, tile_cols=3, tile_rows=3, region="full")
            crops = [rgb.crop(box).resize((embedder.input_size, embedder.input_size)) for box in boxes]
        pixels = embedder.processor(images=crops, return_tensors="pt")["pixel_values"]
        embeddings = embedder.embed_pixel_values(pixels).numpy().astype("float32")
        top_preds = [
            predict_tile(
                embedding=embeddings[i],
                reference_df=reference_df,
                index=index,
                centroids=centroids,
                k=k,
                threshold=non_road_threshold,
            )
            for i in range(3)
        ]
        top_non_road.append(float(np.mean([bool(pred["non_road"]) for pred in top_preds])))
    if top_non_road and float(np.mean(top_non_road)) >= 0.67:
        return "lower_half"
    return "full"


@torch.inference_mode()
def predict_crack_probability(embedding: np.ndarray, crack_detector: dict[str, Any] | None) -> float:
    if crack_detector is None:
        return float("nan")
    head = crack_detector["head"]
    tensor = torch.from_numpy(embedding.reshape(1, -1).astype("float32"))
    return float(torch.sigmoid(head(tensor))[0].item())


@torch.inference_mode()
def predict_defect_probabilities(embedding: np.ndarray, defect_detector: dict[str, Any] | None) -> dict[str, float]:
    if defect_detector is None:
        return {}
    head = defect_detector["head"]
    tensor = torch.from_numpy(embedding.reshape(1, -1).astype("float32"))
    probs = torch.sigmoid(head(tensor))[0].detach().cpu().numpy()
    return {label: float(probs[index]) for index, label in enumerate(defect_detector["labels"])}


def predict_tile(
    embedding: np.ndarray,
    reference_df: pd.DataFrame,
    index: Any,
    centroids: np.ndarray,
    k: int,
    threshold: float,
) -> dict[str, Any]:
    query = normalize_rows(embedding.reshape(1, -1).astype("float32"))
    distances, indices = index.search(query, k)
    sims = distances[0].astype("float32")
    neighbor_idx = indices[0]
    neighbors = reference_df.iloc[neighbor_idx]
    confidence = float(np.mean(sims))
    weights = np.maximum(sims, 0.0) + 1e-6
    quality_scores: dict[int, float] = {}
    for quality, weight in zip(neighbors["quality"].astype(int), weights, strict=True):
        quality_scores[int(quality)] = quality_scores.get(int(quality), 0.0) + float(weight)
    predicted_quality = max(quality_scores.items(), key=lambda item: item[1])[0]
    surface_scores: dict[str, float] = {}
    for surface, weight in zip(neighbors["surface_type"].astype(str), weights, strict=True):
        surface_scores[surface] = surface_scores.get(surface, 0.0) + float(weight)
    surface_type = max(surface_scores.items(), key=lambda item: item[1])[0]
    cluster_scores = query @ centroids.T
    return {
        "predicted_quality": int(predicted_quality),
        "surface_type": surface_type,
        "confidence": confidence,
        "non_road": bool(confidence < threshold),
        "kmeans_cluster": int(np.argmax(cluster_scores[0])),
        "nearest_neighbor": str(neighbors.iloc[0]["image_path"]),
        "nearest_similarity": float(sims[0]),
    }


def build_summary(
    input_path: Path,
    input_type: str,
    out_dir: Path,
    frames_df: pd.DataFrame,
    artifacts: ActiveArtifacts,
    fps: float,
    k: int,
    non_road_threshold: float,
    device: str,
    requested_region: str,
    region: str,
    crack_segmentation: bool,
    defect_detector: dict[str, Any] | None = None,
    defect_gating: bool = True,
) -> dict[str, Any]:
    valid_quality = frames_df["predicted_quality"].dropna().astype(int)
    quality_distribution = {
        str(k): int(v) for k, v in valid_quality.value_counts().sort_index().items()
    }
    dominant_surface = (
        str(frames_df["surface_type"].mode().iloc[0]) if not frames_df.empty else "unknown"
    )
    crack_available = "crack_ratio" in frames_df.columns
    structural_labels = [
        label for label in DEFECT_LABELS if f"defect_{label}_ratio" in frames_df.columns
    ]
    summary = {
        "input_path": str(input_path),
        "input_type": input_type,
        "out_dir": str(out_dir),
        "results_parquet": str(out_dir / "results.parquet"),
        "tiles_parquet": str(out_dir / "tiles.parquet"),
        "frames_analyzed": int(len(frames_df)),
        "quality_distribution": quality_distribution,
        "dominant_surface_type": dominant_surface,
        "mean_confidence": float(frames_df["confidence"].mean()) if len(frames_df) else 0.0,
        "mean_crack_ratio": float(frames_df["crack_ratio"].mean()) if crack_available and len(frames_df) else None,
        "frames_with_crack": int(frames_df["frame_has_crack"].sum()) if crack_available and len(frames_df) else 0,
        "crack_head": str(Path("models/crack_head.pt")) if Path("models/crack_head.pt").exists() else None,
        "crack_segmenter": (
            str(frames_df["crack_segmenter"].mode().iloc[0])
            if "crack_segmenter" in frames_df.columns and not frames_df["crack_segmenter"].dropna().empty
            else ("dinov3_dense_head" if Path("models/crack_seg_head.pt").exists() else "classical")
        ),
        "active_suffix": artifacts.suffix,
        "checkpoint": str(artifacts.checkpoint_path),
        "reference_embeddings": str(artifacts.embeddings_path),
        "faiss_index": str(artifacts.faiss_index_path),
        "centroids": str(artifacts.centroids_path),
        "fps": float(fps),
        "k": int(k),
        "non_road_threshold": float(non_road_threshold),
        "device": device,
        "requested_region": requested_region,
        "region": region,
        "crack_segmentation": bool(crack_segmentation),
        "defect_gating_enabled": bool(defect_gating),
        "defect_label_applicability": DEFECT_LABEL_APPLICABILITY,
    }
    if structural_labels:
        summary["defect_head"] = defect_detector["checkpoint"] if defect_detector else str(Path("models/defect_head.pt"))
        summary["mean_defect_ratios"] = {
            label: float(frames_df[f"defect_{label}_ratio"].mean()) if len(frames_df) else 0.0
            for label in structural_labels
        }
        summary["frames_with_structural_defects"] = (
            int(frames_df["frame_has_structural_defect"].sum())
            if "frame_has_structural_defect" in frames_df.columns
            else 0
        )
    else:
        summary["defect_head"] = str(Path("models/defect_head.pt")) if Path("models/defect_head.pt").exists() else None
    return summary


def print_summary(summary: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Tarmac Analysis Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Frames analyzed", str(summary["frames_analyzed"]))
    table.add_row("Region mode", f"{summary.get('region')} (requested: {summary.get('requested_region')})")
    table.add_row("Dominant surface", str(summary["dominant_surface_type"]))
    table.add_row("Mean confidence", f"{summary['mean_confidence']:.3f}")
    if summary.get("mean_crack_ratio") is not None:
        table.add_row("Mean crack ratio", f"{summary['mean_crack_ratio']:.3f}")
        table.add_row("Frames with cracks", str(summary.get("frames_with_crack", 0)))
    if summary.get("mean_defect_ratios"):
        ratios = ", ".join(
            f"{label}={float(value):.2f}" for label, value in summary["mean_defect_ratios"].items()
        )
        table.add_row("Structural defects", ratios)
        table.add_row("Frames with structural defects", str(summary.get("frames_with_structural_defects", 0)))
    table.add_row("Quality distribution", json.dumps(summary["quality_distribution"]))
    table.add_row("Results", str(summary["results_parquet"]))
    table.add_row("Summary", str(Path(summary["out_dir"]) / "summary.json"))
    console.print(table)


def save_thumbnail(image: Image.Image, path: Path) -> None:
    thumb = image.convert("RGB")
    thumb.thumbnail((320, 320))
    thumb.save(path, format="JPEG", quality=82)


def extract_exif(image: Image.Image) -> dict[str, Any]:
    try:
        exif = image.getexif()
    except Exception:
        return {}
    if not exif:
        return {}
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    result: dict[str, Any] = {}
    timestamp = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if timestamp:
        try:
            result["timestamp"] = datetime.strptime(str(timestamp), "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            result["timestamp"] = str(timestamp)
    gps_info = tags.get("GPSInfo")
    if gps_info:
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
        lat = _gps_coord(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
        lon = _gps_coord(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
        if lat is not None and lon is not None:
            result["latitude"] = lat
            result["longitude"] = lon
    return result


def _gps_coord(value: Any, ref: Any) -> float | None:
    if not value:
        return None
    parts = [float(x) for x in value]
    coord = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if str(ref).upper() in {"S", "W"}:
        coord *= -1
    return coord


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
