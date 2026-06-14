from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from tarmac.defect import DEFECT_LABELS
from tarmac.embedding.embedder import HFBackboneEmbedder
from tarmac.inference.analyze import (
    analyze_frames,
    calibrate_non_road_threshold,
    load_active_artifacts,
    load_crack_detector,
    load_defect_detector,
    load_reference_embeddings,
    normalize_rows,
    resolve_region_mode,
)
from tarmac.inference.assess import SEED, condition_record
from tarmac.survey.report import build_reports
from tarmac.survey.stream import FrameSample, stream_sampled_frames, timestamp_sequence
from tarmac.survey.telemetry import (
    ROUTE_NOTICE,
    dead_reckon,
    extract_imu,
    interpolate_track,
    start_location,
    video_duration,
    write_telemetry_metadata,
)


@dataclass
class SurveyModelContext:
    out_dir: Path
    thumbs_dir: Path
    embedder: HFBackboneEmbedder
    reference_df: pd.DataFrame
    index: Any
    centroids: np.ndarray
    non_road_threshold: float
    crack_detector: dict[str, Any] | None
    defect_detector: dict[str, Any] | None
    region: str
    batch_size: int
    device: str
    active_suffix: str
    checkpoint: str


def run_survey(
    video_path: Path,
    *,
    out_dir: Path | None = None,
    fps: float = 1.0,
    clip_seconds: float | None = None,
    quality_threshold: int = 4,
    device: str = "cpu",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run the GPS/IMU road survey with the active fine-tuned DINOv3 pipeline."""
    _seed_everything(SEED)
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if quality_threshold < 1 or quality_threshold > 5:
        raise ValueError("--quality-threshold must be in the 1-5 quality grade range.")
    out_dir = (out_dir or Path("runs") / f"survey_{video_path.stem}").expanduser().resolve()
    _prepare_output_dir(out_dir)

    duration = video_duration(video_path)
    effective_duration = min(duration, float(clip_seconds)) if clip_seconds is not None else duration
    effective_duration = max(0.0, effective_duration)
    timestamps = timestamp_sequence(duration, fps=fps, clip_seconds=clip_seconds)

    start = start_location(video_path)
    imu_work_dir = out_dir / "_tmp_imu"
    imu = extract_imu(
        video_path,
        work_dir=imu_work_dir,
        clip_seconds=effective_duration if clip_seconds is not None else None,
        duration_seconds=effective_duration,
    )
    telemetry = dead_reckon(imu, start=start, duration_seconds=effective_duration)
    telemetry_path = out_dir / "telemetry.parquet"
    telemetry.to_parquet(telemetry_path, index=False)
    write_telemetry_metadata(
        out_dir,
        {
            "start_location": start.as_dict(),
            "telemetry_parse": imu.as_dict(),
            "route_notice": ROUTE_NOTICE,
        },
    )
    if imu_work_dir.exists():
        shutil.rmtree(imu_work_dir)

    context: SurveyModelContext | None = None
    pending: list[FrameSample] = []
    sample_records: list[dict[str, Any]] = []
    problem_records: list[dict[str, Any]] = []
    problem_dir = out_dir / "problem_images"
    problem_dir.mkdir(parents=True, exist_ok=True)

    frame_iter = stream_sampled_frames(
        video_path,
        out_dir=out_dir,
        fps=fps,
        clip_seconds=clip_seconds,
        jpeg_quality=2,
    )
    for frame in tqdm(frame_iter, total=len(timestamps), desc="Survey frames", unit="frame"):
        pending.append(frame)
        if len(pending) >= batch_size:
            context = context or _load_model_context(
                first_frame_paths=[item.frame_path for item in pending[: min(3, len(pending))]],
                out_dir=out_dir,
                batch_size=batch_size,
                device=device,
            )
            records, problems = _process_batch(
                pending,
                context=context,
                telemetry=telemetry,
                quality_threshold=quality_threshold,
                problem_dir=problem_dir,
            )
            sample_records.extend(records)
            problem_records.extend(problems)
            pending = []

    if pending:
        context = context or _load_model_context(
            first_frame_paths=[item.frame_path for item in pending[: min(3, len(pending))]],
            out_dir=out_dir,
            batch_size=batch_size,
            device=device,
        )
        records, problems = _process_batch(
            pending,
            context=context,
            telemetry=telemetry,
            quality_threshold=quality_threshold,
            problem_dir=problem_dir,
        )
        sample_records.extend(records)
        problem_records.extend(problems)

    _cleanup_temp_dirs(out_dir)
    samples = pd.DataFrame(sample_records)
    problems = pd.DataFrame(problem_records)
    samples_path = out_dir / "samples.parquet"
    problems_path = out_dir / "problems.parquet"
    samples.to_parquet(samples_path, index=False)
    if problems.empty:
        problems = pd.DataFrame(columns=samples.columns)
    problems.to_parquet(problems_path, index=False)
    track_path = _write_geojson(out_dir, telemetry=telemetry, samples=samples, problems=problems)

    summary = _build_summary(
        video_path=video_path,
        out_dir=out_dir,
        duration=duration,
        effective_duration=effective_duration,
        fps=fps,
        clip_seconds=clip_seconds,
        quality_threshold=quality_threshold,
        device=device,
        context=context,
        start=start.as_dict(),
        imu=imu.as_dict(),
        telemetry=telemetry,
        samples=samples,
        problems=problems,
        telemetry_path=telemetry_path,
        samples_path=samples_path,
        problems_path=problems_path,
        track_path=track_path,
    )
    summary_path = out_dir / "summary.json"
    summary["map_html"] = str(out_dir / "map.html")
    summary["problems_table_html"] = str(out_dir / "problems_table.html")
    summary["index_html"] = str(out_dir / "index.html")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_paths = build_reports(out_dir)
    summary.update({name: str(path) for name, path in report_paths.items()})
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def print_survey_summary(summary: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Tarmac Road Survey")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Samples analyzed", str(summary.get("samples_analyzed", 0)))
    table.add_row("Problems found", str(summary.get("problems_found", 0)))
    table.add_row("Mean speed", f"{float(summary.get('mean_speed_kmh', 0.0)):.1f} km/h")
    table.add_row("Telemetry status", str(summary.get("telemetry_parse", {}).get("status", "unknown")))
    table.add_row("Telemetry plausible", str(summary.get("telemetry_parse", {}).get("plausible", False)))
    table.add_row("Output index", str(summary.get("index_html")))
    console.print(table)
    warning = summary.get("telemetry_parse", {}).get("warning")
    if warning:
        console.print(f"[yellow]Telemetry warning:[/yellow] {warning}")
    preview = summary.get("problem_preview", [])
    if preview:
        preview_table = Table(title="Problem Preview")
        preview_table.add_column("t")
        preview_table.add_column("speed_kmh")
        preview_table.add_column("lat,lon")
        preview_table.add_column("issues")
        preview_table.add_column("quality")
        preview_table.add_column("surface")
        for row in preview:
            preview_table.add_row(
                str(row.get("timestamp")),
                f"{float(row.get('speed_kmh', 0.0)):.1f}",
                f"{float(row.get('lat', 0.0)):.6f}, {float(row.get('lon', 0.0)):.6f}",
                ", ".join(row.get("issues", [])),
                str(row.get("quality_grade")),
                str(row.get("surface_type")),
            )
        console.print(preview_table)


def _load_model_context(
    *,
    first_frame_paths: list[Path],
    out_dir: Path,
    batch_size: int,
    device: str,
) -> SurveyModelContext:
    torch.set_num_threads(1)
    artifacts = load_active_artifacts()
    if "dinov3" not in artifacts.suffix.lower() and "dinov3" not in artifacts.model_name.lower():
        raise RuntimeError(
            f"Survey requires the active fine-tuned DINOv3 model, got suffix={artifacts.suffix} "
            f"model={artifacts.model_name}"
        )
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
    non_road_threshold = calibrate_non_road_threshold(artifacts.embeddings_path, index, k=10)
    crack_detector = load_crack_detector()
    defect_detector = load_defect_detector()
    region = resolve_region_mode(
        requested_region="auto",
        frame_paths=first_frame_paths,
        embedder=embedder,
        reference_df=ref_df,
        index=index,
        centroids=centroids,
        k=10,
        non_road_threshold=non_road_threshold,
    )
    thumbs_dir = out_dir / "_tmp_thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    return SurveyModelContext(
        out_dir=out_dir,
        thumbs_dir=thumbs_dir,
        embedder=embedder,
        reference_df=ref_df,
        index=index,
        centroids=centroids,
        non_road_threshold=non_road_threshold,
        crack_detector=crack_detector,
        defect_detector=defect_detector,
        region=region,
        batch_size=batch_size,
        device=device,
        active_suffix=artifacts.suffix,
        checkpoint=str(artifacts.checkpoint_path),
    )


def _process_batch(
    frames: list[FrameSample],
    *,
    context: SurveyModelContext,
    telemetry: pd.DataFrame,
    quality_threshold: int,
    problem_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, _tile_rows = analyze_frames(
        frame_paths=[frame.frame_path for frame in frames],
        input_type="video",
        out_dir=context.out_dir,
        thumbs_dir=context.thumbs_dir,
        embedder=context.embedder,
        reference_df=context.reference_df,
        index=context.index,
        centroids=context.centroids,
        k=10,
        non_road_threshold=context.non_road_threshold,
        batch_size=context.batch_size,
        crack_detector=context.crack_detector,
        defect_detector=context.defect_detector,
        region=context.region,
        crack_segmentation=False,
        mm_per_pixel=None,
        defect_gating=True,
        device_name=context.device,
    )
    records: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for local_index, (frame, row) in enumerate(zip(frames, rows, strict=True)):
        row = dict(row)
        row["frame_index"] = frame.index
        telemetry_row = interpolate_track(telemetry, frame.timestamp_s)
        record = _sample_record(
            row=row,
            frame=frame,
            telemetry_row=telemetry_row,
            quality_threshold=quality_threshold,
            region=context.region,
        )
        if record["is_problem"]:
            image_rel, thumb_rel = _save_problem_assets(
                source=frame.frame_path,
                problem_dir=problem_dir,
                sample_index=frame.index,
                timestamp_s=frame.timestamp_s,
            )
            record["problem_image"] = image_rel
            record["thumbnail_image"] = thumb_rel
            problems.append(record.copy())
        records.append(record)
        _delete_quietly(frame.frame_path)
    return records, problems


def _sample_record(
    *,
    row: dict[str, Any],
    frame: FrameSample,
    telemetry_row: dict[str, Any],
    quality_threshold: int,
    region: str,
) -> dict[str, Any]:
    quality = _maybe_int(row.get("predicted_quality"))
    surface_type = str(row.get("surface_type") or "unknown")
    crack_flag = _maybe_bool(row.get("frame_has_crack")) or _maybe_bool(row.get("frame_has_defect_crack"))
    structural_defects = _structural_defects(row)
    quality_issue = quality is not None and quality >= quality_threshold
    issues: list[str] = []
    if crack_flag:
        issues.append("crack")
    issues.extend(structural_defects)
    if quality_issue:
        issues.append(f"quality_grade_{quality}")
    is_problem = bool(crack_flag or structural_defects or quality_issue)
    assessment = _assessment_fields(row)
    record: dict[str, Any] = {
        "frame_index": int(frame.index),
        "t": float(frame.timestamp_s),
        "timestamp": _timestamp_label(frame.timestamp_s),
        "lat": float(telemetry_row["lat"]),
        "lon": float(telemetry_row["lon"]),
        "speed_mps": float(telemetry_row["speed_mps"]),
        "speed_kmh": float(telemetry_row["speed_kmh"]),
        "heading_deg": float(telemetry_row["heading_deg"]),
        "telemetry_source": str(telemetry_row["telemetry_source"]),
        "route_approximate": bool(telemetry_row["route_approximate"]),
        "quality_grade": quality,
        "surface_type": surface_type,
        "confidence": _maybe_float(row.get("confidence")),
        "road_tile_count": _maybe_int(row.get("road_tile_count")) or 0,
        "tile_count": _maybe_int(row.get("tile_count")) or 0,
        "crack_detected": bool(crack_flag),
        "structural_defects": json.dumps(structural_defects),
        "issues": json.dumps(issues),
        "is_problem": is_problem,
        "problem_image": "",
        "thumbnail_image": "",
        "region": region,
        "route_notice": ROUTE_NOTICE,
        **assessment,
    }
    for label in DEFECT_LABELS:
        record[f"defect_{label}"] = bool(_maybe_bool(row.get(f"frame_has_defect_{label}")))
        ratio = _maybe_float(row.get(f"defect_{label}_ratio"))
        record[f"defect_{label}_ratio"] = ratio if ratio is not None else 0.0
    return record


def _assessment_fields(row: dict[str, Any]) -> dict[str, Any]:
    try:
        record = condition_record(row)
    except Exception:
        return {
            "overall_condition_grade": _maybe_int(row.get("predicted_quality")),
            "repair_priority": "unknown",
            "assessment_key_defects": "[]",
        }
    return {
        "overall_condition_grade": int(record.get("overall_condition_grade", 0)),
        "repair_priority": str(record.get("repair_priority", "none")),
        "assessment_key_defects": json.dumps(record.get("key_defects", [])),
    }


def _structural_defects(row: dict[str, Any]) -> list[str]:
    defects: list[str] = []
    for label in DEFECT_LABELS:
        if label == "crack":
            continue
        if _maybe_bool(row.get(f"frame_has_defect_{label}")):
            defects.append(label)
    try:
        parsed = json.loads(str(row.get("structural_defects") or "[]"))
        if isinstance(parsed, list):
            for label in parsed:
                label = str(label)
                if label != "crack" and label in DEFECT_LABELS and label not in defects:
                    defects.append(label)
    except json.JSONDecodeError:
        pass
    return defects


def _save_problem_assets(
    *,
    source: Path,
    problem_dir: Path,
    sample_index: int,
    timestamp_s: float,
) -> tuple[str, str]:
    stem = f"problem_{sample_index:06d}_t{timestamp_s:010.3f}"
    image_path = problem_dir / f"{stem}.jpg"
    thumb_path = problem_dir / f"{stem}_thumb.jpg"
    shutil.copy2(source, image_path)
    with Image.open(source) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail((360, 240))
        thumb.save(thumb_path, format="JPEG", quality=82)
    return f"problem_images/{image_path.name}", f"problem_images/{thumb_path.name}"


def _write_geojson(
    out_dir: Path,
    *,
    telemetry: pd.DataFrame,
    samples: pd.DataFrame,
    problems: pd.DataFrame,
) -> Path:
    route_coords = [
        [float(row.lon), float(row.lat)]
        for row in telemetry.itertuples()
        if pd.notna(getattr(row, "lat", None)) and pd.notna(getattr(row, "lon", None))
    ]
    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "properties": {
                "name": "IMU-estimated route",
                "approximate": True,
                "notice": ROUTE_NOTICE,
                "telemetry_source": str(telemetry["telemetry_source"].iloc[0]) if not telemetry.empty else "unknown",
            },
            "geometry": {"type": "LineString", "coordinates": route_coords},
        }
    ]
    for row in problems.itertuples():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "frame_index": int(getattr(row, "frame_index", 0)),
                    "t": float(getattr(row, "t", 0.0)),
                    "timestamp": str(getattr(row, "timestamp", "")),
                    "speed_kmh": float(getattr(row, "speed_kmh", 0.0)),
                    "quality_grade": _maybe_int(getattr(row, "quality_grade", None)),
                    "surface_type": str(getattr(row, "surface_type", "unknown")),
                    "issues": _json_list(getattr(row, "issues", "[]")),
                    "problem_image": str(getattr(row, "problem_image", "")),
                    "thumbnail_image": str(getattr(row, "thumbnail_image", "")),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(getattr(row, "lon", 0.0)), float(getattr(row, "lat", 0.0))],
                },
            }
        )
    path = out_dir / "track.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n", encoding="utf-8")
    return path


def _build_summary(
    *,
    video_path: Path,
    out_dir: Path,
    duration: float,
    effective_duration: float,
    fps: float,
    clip_seconds: float | None,
    quality_threshold: int,
    device: str,
    context: SurveyModelContext | None,
    start: dict[str, Any],
    imu: dict[str, Any],
    telemetry: pd.DataFrame,
    samples: pd.DataFrame,
    problems: pd.DataFrame,
    telemetry_path: Path,
    samples_path: Path,
    problems_path: Path,
    track_path: Path,
) -> dict[str, Any]:
    quality_counts = Counter(
        str(int(q)) for q in samples["quality_grade"].dropna().tolist()
    ) if "quality_grade" in samples else Counter()
    issue_counts: Counter[str] = Counter()
    for value in problems["issues"].tolist() if "issues" in problems else []:
        issue_counts.update(_json_list(value))
    preview = []
    for row in problems.head(8).itertuples():
        preview.append(
            {
                "timestamp": str(getattr(row, "timestamp", "")),
                "speed_kmh": float(getattr(row, "speed_kmh", 0.0)),
                "lat": float(getattr(row, "lat", 0.0)),
                "lon": float(getattr(row, "lon", 0.0)),
                "issues": _json_list(getattr(row, "issues", "[]")),
                "quality_grade": _maybe_int(getattr(row, "quality_grade", None)),
                "surface_type": str(getattr(row, "surface_type", "unknown")),
            }
        )
    return {
        "run_name": out_dir.name,
        "input_path": str(video_path),
        "out_dir": str(out_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "fps": float(fps),
        "clip_seconds": clip_seconds,
        "video_duration_seconds": float(duration),
        "effective_duration_seconds": float(effective_duration),
        "quality_threshold": int(quality_threshold),
        "device": device,
        "active_suffix": context.active_suffix if context else None,
        "checkpoint": context.checkpoint if context else None,
        "region": context.region if context else None,
        "start_location": start,
        "route_notice": ROUTE_NOTICE,
        "telemetry_parse": imu,
        "telemetry_parquet": str(telemetry_path),
        "samples_parquet": str(samples_path),
        "problems_parquet": str(problems_path),
        "track_geojson": str(track_path),
        "problem_images_dir": str(out_dir / "problem_images"),
        "samples_analyzed": int(len(samples)),
        "problems_found": int(len(problems)),
        "mean_speed_kmh": float(samples["speed_kmh"].mean()) if len(samples) else 0.0,
        "telemetry_mean_speed_kmh": float(telemetry["speed_kmh"].mean()) if len(telemetry) else 0.0,
        "quality_distribution": dict(sorted(quality_counts.items())),
        "problem_issue_counts": dict(sorted(issue_counts.items())),
        "problem_preview": preview,
    }


def _prepare_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in ["problem_images", "_tmp_frames", "_tmp_thumbnails", "_tmp_imu"]:
        path = out_dir / child
        if path.exists():
            shutil.rmtree(path)
    for filename in [
        "telemetry.parquet",
        "telemetry_metadata.json",
        "track.geojson",
        "samples.parquet",
        "problems.parquet",
        "summary.json",
        "map.html",
        "problems_table.html",
        "index.html",
    ]:
        path = out_dir / filename
        if path.exists():
            path.unlink()


def _cleanup_temp_dirs(out_dir: Path) -> None:
    for child in ["_tmp_frames", "_tmp_thumbnails"]:
        path = out_dir / child
        if path.exists():
            shutil.rmtree(path)


def _delete_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _timestamp_label(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _maybe_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
