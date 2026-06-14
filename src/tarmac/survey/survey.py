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

from tarmac.crack.segment import segment_cracks
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
from tarmac.survey.gps_sources import (
    GpsSource,
    GpsSourceType,
    detect_gps_source,
    gps_source_status,
    interpolate_track,
    no_geo_track,
    route_notice_for_source,
)
from tarmac.survey.telemetry import (
    ROUTE_NOTICE,
    dead_reckon,
    extract_imu,
    video_duration,
    write_telemetry_metadata,
)

DEFAULT_CRACK_PROB = 0.6
DEFAULT_MIN_CRACK_AREA = 0.3
DEFAULT_MIN_CRACK_COMPONENT_LENGTH_PX = 64
DEFAULT_CRACK_SEG_CHECKPOINT = Path("models/crack_seg_head.pt")


@dataclass(frozen=True)
class CrackConfirmationConfig:
    crack_prob: float = DEFAULT_CRACK_PROB
    min_crack_area: float = DEFAULT_MIN_CRACK_AREA
    min_crack_length_px: int = DEFAULT_MIN_CRACK_COMPONENT_LENGTH_PX
    checkpoint_path: Path = DEFAULT_CRACK_SEG_CHECKPOINT


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
    crack_prob: float = DEFAULT_CRACK_PROB,
    min_crack_area: float = DEFAULT_MIN_CRACK_AREA,
    min_crack_length_px: int = DEFAULT_MIN_CRACK_COMPONENT_LENGTH_PX,
    device: str = "cpu",
    batch_size: int = 8,
    gps_sidecar: Path | None = None,
    gps_source: str = "auto",
) -> dict[str, Any]:
    """Run the GPS/IMU road survey with the active fine-tuned DINOv3 pipeline."""
    _seed_everything(SEED)
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if quality_threshold < 1 or quality_threshold > 5:
        raise ValueError("--quality-threshold must be in the 1-5 quality grade range.")
    if crack_prob < 0.0 or crack_prob > 1.0:
        raise ValueError("--crack-prob must be in the 0-1 probability range.")
    if min_crack_area < 0.0:
        raise ValueError("--min-crack-area must be non-negative.")
    if min_crack_length_px < 0:
        raise ValueError("--min-crack-length-px must be non-negative.")
    crack_confirmation = CrackConfirmationConfig(
        crack_prob=float(crack_prob),
        min_crack_area=float(min_crack_area),
        min_crack_length_px=int(min_crack_length_px),
    )
    out_dir = (out_dir or Path("runs") / f"survey_{video_path.stem}").expanduser().resolve()
    _prepare_output_dir(out_dir)

    duration = video_duration(video_path)
    effective_duration = min(duration, float(clip_seconds)) if clip_seconds is not None else duration
    effective_duration = max(0.0, effective_duration)
    timestamps = timestamp_sequence(duration, fps=fps, clip_seconds=clip_seconds)

    gps = detect_gps_source(video_path, sidecar=gps_sidecar, source_hint=gps_source)
    route_notice = route_notice_for_source(gps)
    imu_work_dir = out_dir / "_tmp_imu"
    imu_payload: dict[str, Any] | None = None
    if gps.source_type == GpsSourceType.IMU_DEADRECKON:
        if gps.start is None:
            raise RuntimeError("IMU dead-reckoning requires a start GPS point.")
        imu = extract_imu(
            video_path,
            work_dir=imu_work_dir,
            clip_seconds=effective_duration if clip_seconds is not None else None,
            duration_seconds=effective_duration,
        )
        telemetry = dead_reckon(imu, start=gps.start, duration_seconds=effective_duration)
        telemetry["notice"] = route_notice
        imu_payload = imu.as_dict()
    elif gps.track is not None:
        telemetry = gps.track.copy()
        telemetry = telemetry[telemetry["t"].astype(float) <= effective_duration].reset_index(drop=True)
        if telemetry.empty:
            telemetry = gps.track.copy()
        telemetry["notice"] = route_notice
    else:
        telemetry = no_geo_track(effective_duration, start=gps.start, reason=gps.reason)
        telemetry["notice"] = route_notice
    telemetry_parse = gps_source_status(gps, telemetry_parse=imu_payload)
    telemetry_path = out_dir / "telemetry.parquet"
    telemetry.to_parquet(telemetry_path, index=False)
    write_telemetry_metadata(
        out_dir,
        {
            "gps_source": gps.as_dict(),
            "start_location": gps.start.as_dict() if gps.start is not None else None,
            "telemetry_parse": telemetry_parse,
            "route_notice": route_notice,
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
                crack_confirmation=crack_confirmation,
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
            crack_confirmation=crack_confirmation,
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
        crack_confirmation=crack_confirmation,
        device=device,
        context=context,
        gps=gps,
        route_notice=route_notice,
        telemetry_parse=telemetry_parse,
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
    gps_source = summary.get("gps_source", {})
    table.add_row("GPS source", f"{gps_source.get('type', 'unknown')} - {gps_source.get('reason', '')}")
    table.add_row(
        f"Mean speed {summary.get('speed_label', '')}".strip(),
        f"{float(summary.get('mean_speed_kmh', 0.0)):.1f} km/h",
    )
    table.add_row("Confirmed cracks", str(summary.get("confirmed_crack_count", 0)))
    table.add_row("Telemetry status", str(summary.get("telemetry_parse", {}).get("status", "unknown")))
    table.add_row("Telemetry plausible", str(summary.get("telemetry_parse", {}).get("plausible", False)))
    table.add_row("Output index", str(summary.get("index_html")))
    console.print(table)
    speed_warning = summary.get("speed_warning")
    if speed_warning:
        console.print(f"[yellow]Speed warning:[/yellow] {speed_warning}")
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
                _format_lat_lon(row.get("lat"), row.get("lon")),
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
    requested_region: str = "auto",
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
        requested_region=requested_region,
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
    crack_confirmation: CrackConfirmationConfig,
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
        confirmation = _confirm_crack_for_image(
            frame.frame_path,
            row=row,
            context=context,
            config=crack_confirmation,
            force_segmentation=False,
        )
        record = _sample_record(
            row=row,
            frame=frame,
            telemetry_row=telemetry_row,
            quality_threshold=quality_threshold,
            crack_confirmation=confirmation,
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
    crack_confirmation: dict[str, Any],
    region: str,
) -> dict[str, Any]:
    quality = _maybe_int(row.get("predicted_quality"))
    surface_type = str(row.get("surface_type") or "unknown")
    crack_classifier_flag = _maybe_bool(row.get("frame_has_crack")) or _maybe_bool(row.get("frame_has_defect_crack"))
    crack_flag = bool(crack_confirmation.get("crack_confirmed", False))
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
        "crack_confirmed": bool(crack_flag),
        "crack_classifier_detected": bool(crack_classifier_flag),
        "crack_candidate": bool(crack_confirmation.get("crack_candidate", False)),
        "crack_classifier_prob": _finite_or_none(crack_confirmation.get("crack_classifier_prob")),
        "crack_classifier_threshold": _finite_or_none(crack_confirmation.get("crack_classifier_threshold")),
        "crack_min_area_pct": _finite_or_none(crack_confirmation.get("crack_min_area_pct")),
        "crack_min_component_length_px": _maybe_int(crack_confirmation.get("crack_min_component_length_px")) or 0,
        "crack_area_px": _maybe_int(crack_confirmation.get("crack_area_px")) or 0,
        "crack_area_pct": _finite_or_zero(crack_confirmation.get("crack_area_pct")),
        "crack_length_px": _maybe_int(crack_confirmation.get("crack_length_px")) or 0,
        "crack_max_component_length_px": _maybe_int(crack_confirmation.get("crack_max_component_length_px")) or 0,
        "crack_components": _maybe_int(crack_confirmation.get("crack_components")) or 0,
        "crack_segmenter": str(crack_confirmation.get("crack_segmenter", "")),
        "crack_confirmation_reason": str(crack_confirmation.get("crack_confirmation_reason", "")),
        "structural_defects": json.dumps(structural_defects),
        "issues": json.dumps(issues),
        "is_problem": is_problem,
        "problem_image": "",
        "thumbnail_image": "",
        "region": region,
        "route_notice": str(telemetry_row.get("route_notice") or ROUTE_NOTICE),
        **assessment,
    }
    for label in DEFECT_LABELS:
        record[f"defect_{label}"] = bool(_maybe_bool(row.get(f"frame_has_defect_{label}")))
        ratio = _maybe_float(row.get(f"defect_{label}_ratio"))
        record[f"defect_{label}_ratio"] = ratio if ratio is not None else 0.0
    return record


def _confirm_crack_for_image(
    image_path: Path,
    *,
    row: dict[str, Any],
    context: SurveyModelContext,
    config: CrackConfirmationConfig,
    force_segmentation: bool,
) -> dict[str, Any]:
    classifier_prob = _max_tile_crack_probability(row)
    candidate = bool(classifier_prob is not None and classifier_prob >= config.crack_prob)
    confirmation = _empty_crack_confirmation(
        classifier_prob=classifier_prob,
        config=config,
        candidate=candidate,
    )
    if not candidate and not force_segmentation:
        confirmation["crack_confirmation_reason"] = "classifier_below_threshold"
        return confirmation

    with Image.open(image_path) as image:
        result = segment_cracks(
            image,
            crack_head=context.crack_detector,
            embedder=context.embedder,
            mm_per_pixel=None,
            output_path=None,
            batch_size=context.batch_size,
            learned_checkpoint_path=config.checkpoint_path,
            prefer_learned=True,
            device_name=context.device,
        )
    measurements = result.measurements
    area_pct = float(measurements.get("crack_area_pct", 0.0) or 0.0)
    component_length = int(measurements.get("max_component_length_px", 0) or 0)
    confirmed = bool(
        candidate
        and area_pct >= config.min_crack_area
        and component_length >= config.min_crack_length_px
    )
    if confirmed:
        reason = "confirmed"
    elif not candidate:
        reason = "classifier_below_threshold"
    elif area_pct < config.min_crack_area:
        reason = "seg_area_below_threshold"
    else:
        reason = "component_length_below_threshold"
    confirmation.update(
        {
            "crack_confirmed": confirmed,
            "crack_area_px": int(measurements.get("crack_area_px", 0) or 0),
            "crack_area_pct": area_pct,
            "crack_length_px": int(measurements.get("total_length_px", 0) or 0),
            "crack_max_component_length_px": component_length,
            "crack_max_component_area_px": int(measurements.get("max_component_area_px", 0) or 0),
            "crack_components": int(measurements.get("n_components", 0) or 0),
            "crack_segmenter": result.segmenter,
            "crack_confirmation_reason": reason,
        }
    )
    return confirmation


def _empty_crack_confirmation(
    *,
    classifier_prob: float | None,
    config: CrackConfirmationConfig,
    candidate: bool,
) -> dict[str, Any]:
    return {
        "crack_confirmed": False,
        "crack_candidate": bool(candidate),
        "crack_classifier_prob": classifier_prob,
        "crack_classifier_threshold": float(config.crack_prob),
        "crack_min_area_pct": float(config.min_crack_area),
        "crack_min_component_length_px": int(config.min_crack_length_px),
        "crack_area_px": 0,
        "crack_area_pct": 0.0,
        "crack_length_px": 0,
        "crack_max_component_length_px": 0,
        "crack_max_component_area_px": 0,
        "crack_components": 0,
        "crack_segmenter": "",
        "crack_confirmation_reason": "not_evaluated",
    }


def _max_tile_crack_probability(row: dict[str, Any]) -> float | None:
    tiles = _tile_details(row.get("tile_details"))
    road_probs: list[float] = []
    all_probs: list[float] = []
    for tile in tiles:
        prob = _maybe_float(tile.get("tile_crack_prob"))
        if prob is None:
            continue
        all_probs.append(prob)
        if not _maybe_bool(tile.get("non_road")):
            road_probs.append(prob)
    values = road_probs or all_probs
    return max(values) if values else None


def _tile_details(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


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
        if _valid_lat_lon(getattr(row, "lat", None), getattr(row, "lon", None))
    ]
    features: list[dict[str, Any]] = []
    if len(route_coords) >= 2:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": "Survey route",
                    "approximate": bool(telemetry["route_approximate"].iloc[0]) if not telemetry.empty else False,
                    "notice": str(telemetry["notice"].iloc[0]) if "notice" in telemetry and not telemetry.empty else "",
                    "telemetry_source": str(telemetry["telemetry_source"].iloc[0]) if not telemetry.empty else "unknown",
                },
                "geometry": {"type": "LineString", "coordinates": route_coords},
            }
        )
    for row in problems.itertuples():
        lat = getattr(row, "lat", None)
        lon = getattr(row, "lon", None)
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
                    "crack_confirmed": bool(getattr(row, "crack_confirmed", False)),
                    "crack_classifier_prob": _maybe_float(getattr(row, "crack_classifier_prob", None)),
                    "crack_area_pct": _maybe_float(getattr(row, "crack_area_pct", None)),
                    "crack_max_component_length_px": _maybe_int(
                        getattr(row, "crack_max_component_length_px", None)
                    ),
                    "crack_confirmation_reason": str(getattr(row, "crack_confirmation_reason", "")),
                },
                "geometry": (
                    {"type": "Point", "coordinates": [float(lon), float(lat)]}
                    if _valid_lat_lon(lat, lon)
                    else None
                ),
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
    crack_confirmation: CrackConfirmationConfig,
    device: str,
    context: SurveyModelContext | None,
    gps: GpsSource,
    route_notice: str,
    telemetry_parse: dict[str, Any],
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
    crack_count = int(
        sum("crack" in _json_list(value) for value in problems["issues"].tolist())
    ) if "issues" in problems else 0
    preview = []
    for row in problems.head(8).itertuples():
        preview.append(
            {
                "timestamp": str(getattr(row, "timestamp", "")),
                "speed_kmh": float(getattr(row, "speed_kmh", 0.0)),
                "lat": _json_float_or_none(getattr(row, "lat", None)),
                "lon": _json_float_or_none(getattr(row, "lon", None)),
                "issues": _json_list(getattr(row, "issues", "[]")),
                "quality_grade": _maybe_int(getattr(row, "quality_grade", None)),
                "surface_type": str(getattr(row, "surface_type", "unknown")),
                "crack_area_pct": _maybe_float(getattr(row, "crack_area_pct", None)),
                "crack_max_component_length_px": _maybe_int(
                    getattr(row, "crack_max_component_length_px", None)
                ),
                "crack_confirmation_reason": str(getattr(row, "crack_confirmation_reason", "")),
            }
        )
    mean_speed_kmh = float(samples["speed_kmh"].mean()) if len(samples) else 0.0
    speed_label = _speed_label(gps.source_type)
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
        "crack_confirmation": {
            "enabled": True,
            "crack_prob": float(crack_confirmation.crack_prob),
            "min_crack_area_pct": float(crack_confirmation.min_crack_area),
            "min_crack_component_length_px": int(crack_confirmation.min_crack_length_px),
            "checkpoint": str(crack_confirmation.checkpoint_path),
        },
        "device": device,
        "active_suffix": context.active_suffix if context else None,
        "checkpoint": context.checkpoint if context else None,
        "region": context.region if context else None,
        "gps_source": gps.as_dict(),
        "start_location": gps.start.as_dict() if gps.start is not None else None,
        "route_notice": route_notice,
        "telemetry_parse": telemetry_parse,
        "telemetry_parquet": str(telemetry_path),
        "samples_parquet": str(samples_path),
        "problems_parquet": str(problems_path),
        "track_geojson": str(track_path),
        "problem_images_dir": str(out_dir / "problem_images"),
        "samples_analyzed": int(len(samples)),
        "problems_found": int(len(problems)),
        "confirmed_crack_count": crack_count,
        "mean_speed_kmh": mean_speed_kmh,
        "speed_label": speed_label,
        "speed_warning": _speed_warning(
            mean_speed_kmh,
            sample_count=len(samples),
            gps_source_type=gps.source_type.value,
        ),
        "telemetry_mean_speed_kmh": float(telemetry["speed_kmh"].mean()) if len(telemetry) else 0.0,
        "quality_distribution": dict(sorted(quality_counts.items())),
        "problem_issue_counts": dict(sorted(issue_counts.items())),
        "problem_preview": preview,
    }


def confirm_survey_problems(
    out_dir: Path,
    *,
    crack_prob: float = DEFAULT_CRACK_PROB,
    min_crack_area: float = DEFAULT_MIN_CRACK_AREA,
    min_crack_length_px: int = DEFAULT_MIN_CRACK_COMPONENT_LENGTH_PX,
    quality_threshold: int | None = None,
    device: str = "cpu",
    batch_size: int = 8,
    rebuild_reports: bool = True,
) -> dict[str, Any]:
    """Re-check saved survey problem images without touching the source video."""
    _seed_everything(SEED)
    out_dir = out_dir.expanduser().resolve()
    samples_path = out_dir / "samples.parquet"
    problems_path = out_dir / "problems.parquet"
    telemetry_path = out_dir / "telemetry.parquet"
    summary_path = out_dir / "summary.json"
    for path in [samples_path, problems_path, telemetry_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing survey artifact: {path}")
    if crack_prob < 0.0 or crack_prob > 1.0:
        raise ValueError("--crack-prob must be in the 0-1 probability range.")
    if min_crack_area < 0.0:
        raise ValueError("--min-crack-area must be non-negative.")
    if min_crack_length_px < 0:
        raise ValueError("--min-crack-length-px must be non-negative.")

    samples = pd.read_parquet(samples_path)
    problems = pd.read_parquet(problems_path)
    telemetry = pd.read_parquet(telemetry_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    threshold = int(quality_threshold or summary.get("quality_threshold", 4))
    config = CrackConfirmationConfig(
        crack_prob=float(crack_prob),
        min_crack_area=float(min_crack_area),
        min_crack_length_px=int(min_crack_length_px),
    )

    problem_rows = problems.to_dict("records")
    problem_paths = [_survey_artifact_path(out_dir, str(row.get("problem_image", ""))) for row in problem_rows]
    missing = [str(path) for path in problem_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing saved problem image(s): {missing[:5]}")

    confirmation_rows: list[dict[str, Any]] = []
    context: SurveyModelContext | None = None
    if problem_paths:
        first_region = str(problem_rows[0].get("region") or "auto")
        context = _load_model_context(
            first_frame_paths=problem_paths[: min(3, len(problem_paths))],
            out_dir=out_dir,
            batch_size=batch_size,
            device=device,
            requested_region=first_region,
        )
        for start in tqdm(range(0, len(problem_rows), batch_size), desc="Confirming cracks", unit="batch"):
            batch_rows = problem_rows[start : start + batch_size]
            batch_paths = problem_paths[start : start + batch_size]
            analyzed_rows, _tile_rows = analyze_frames(
                frame_paths=batch_paths,
                input_type="survey_problem_image",
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
            for original, image_path, analyzed in zip(batch_rows, batch_paths, analyzed_rows, strict=True):
                confirmation = _confirm_crack_for_image(
                    image_path,
                    row=dict(analyzed),
                    context=context,
                    config=config,
                    force_segmentation=True,
                )
                confirmation_rows.append(
                    _confirmed_problem_record(
                        original,
                        confirmation=confirmation,
                        quality_threshold=threshold,
                    )
                )
    _cleanup_temp_dirs(out_dir)

    confirmations = pd.DataFrame(confirmation_rows)
    if confirmations.empty:
        confirmations = pd.DataFrame(columns=list(problems.columns))
    confirmed = confirmations[confirmations["is_problem"].astype(bool)].copy() if not confirmations.empty else confirmations
    confirmations_path = out_dir / "problem_confirmations.parquet"
    confirmed_path = out_dir / "problems_confirmed.parquet"
    confirmations.to_parquet(confirmations_path, index=False)
    confirmed.to_parquet(confirmed_path, index=False)
    track_path = _write_geojson(out_dir, telemetry=telemetry, samples=samples, problems=confirmed)

    before_cracks = _count_issue(problems, "crack")
    after_cracks = _count_issue(confirmed, "crack")
    issue_counts = _issue_counts(confirmed)
    raw_issue_counts = _issue_counts(problems)
    mean_speed_kmh = float(samples["speed_kmh"].mean()) if len(samples) and "speed_kmh" in samples else 0.0
    gps_source_type = str(summary.get("gps_source", {}).get("type", GpsSourceType.IMU_DEADRECKON.value))
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "quality_threshold": threshold,
            "original_problems_found": int(len(problems)),
            "problems_before_confirmation": int(len(problems)),
            "problems_after_confirmation": int(len(confirmed)),
            "problems_found": int(len(confirmed)),
            "crack_count_before_confirmation": before_cracks,
            "crack_count_after_confirmation": after_cracks,
            "confirmed_crack_count": after_cracks,
            "unconfirmed_crack_count": int(confirmations.get("crack_unconfirmed", pd.Series(dtype=bool)).sum()),
            "raw_problem_issue_counts": raw_issue_counts,
            "problem_issue_counts": issue_counts,
            "confirmed_problem_issue_counts": issue_counts,
            "crack_confirmation": {
                "enabled": True,
                "crack_prob": float(config.crack_prob),
                "min_crack_area_pct": float(config.min_crack_area),
                "min_crack_component_length_px": int(config.min_crack_length_px),
                "checkpoint": str(config.checkpoint_path),
                "rechecked_saved_problem_images": True,
            },
            "problem_confirmations_parquet": str(confirmations_path),
            "problems_confirmed_parquet": str(confirmed_path),
            "track_geojson": str(track_path),
            "problem_preview": _problem_preview(confirmed),
            "mean_speed_kmh": mean_speed_kmh,
            "speed_label": summary.get("speed_label") or _speed_label(gps_source_type),
            "speed_warning": _speed_warning(
                mean_speed_kmh,
                sample_count=len(samples),
                gps_source_type=gps_source_type,
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if rebuild_reports:
        report_paths = build_reports(out_dir)
        summary.update({name: str(path) for name, path in report_paths.items()})
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _confirmed_problem_record(
    original: dict[str, Any],
    *,
    confirmation: dict[str, Any],
    quality_threshold: int,
) -> dict[str, Any]:
    updated = dict(original)
    raw_issues = _json_list(original.get("issues", "[]"))
    structural_defects = [label for label in _json_list(original.get("structural_defects", "[]")) if label != "crack"]
    quality = _maybe_int(original.get("quality_grade"))
    quality_issue = quality is not None and quality >= quality_threshold
    crack_confirmed = bool(confirmation.get("crack_confirmed", False))
    issues: list[str] = []
    if crack_confirmed:
        issues.append("crack")
    issues.extend(structural_defects)
    if quality_issue:
        issues.append(f"quality_grade_{quality}")
    updated.update(confirmation)
    updated["raw_issues"] = json.dumps(raw_issues)
    updated["raw_crack_detected"] = bool(_maybe_bool(original.get("crack_detected")) or "crack" in raw_issues)
    updated["crack_detected"] = crack_confirmed
    updated["crack_confirmed"] = crack_confirmed
    updated["crack_unconfirmed"] = bool(updated["raw_crack_detected"] and not crack_confirmed)
    updated["structural_defects"] = json.dumps(structural_defects)
    updated["issues"] = json.dumps(issues)
    updated["is_problem"] = bool(issues)
    return updated


def _survey_artifact_path(out_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = out_dir / path
    return path.resolve()


def _issue_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if "issues" in frame:
        for value in frame["issues"].tolist():
            counts.update(_json_list(value))
    return dict(sorted(counts.items()))


def _count_issue(frame: pd.DataFrame, issue: str) -> int:
    if "issues" not in frame:
        return 0
    return int(sum(issue in _json_list(value) for value in frame["issues"].tolist()))


def _problem_preview(problems: pd.DataFrame, *, limit: int = 8) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for row in problems.head(limit).itertuples():
        preview.append(
            {
                "timestamp": str(getattr(row, "timestamp", "")),
                "speed_kmh": float(getattr(row, "speed_kmh", 0.0)),
                "lat": _json_float_or_none(getattr(row, "lat", None)),
                "lon": _json_float_or_none(getattr(row, "lon", None)),
                "issues": _json_list(getattr(row, "issues", "[]")),
                "quality_grade": _maybe_int(getattr(row, "quality_grade", None)),
                "surface_type": str(getattr(row, "surface_type", "unknown")),
                "crack_area_pct": _maybe_float(getattr(row, "crack_area_pct", None)),
                "crack_max_component_length_px": _maybe_int(
                    getattr(row, "crack_max_component_length_px", None)
                ),
                "crack_confirmation_reason": str(getattr(row, "crack_confirmation_reason", "")),
            }
        )
    return preview


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


def _finite_or_none(value: Any) -> float | None:
    return _maybe_float(value)


def _finite_or_zero(value: Any) -> float:
    number = _maybe_float(value)
    return number if number is not None else 0.0


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


def _valid_lat_lon(lat: Any, lon: Any) -> bool:
    lat_value = _maybe_float(lat)
    lon_value = _maybe_float(lon)
    return lat_value is not None and lon_value is not None


def _json_float_or_none(value: Any) -> float | None:
    number = _maybe_float(value)
    return float(number) if number is not None else None


def _format_lat_lon(lat: Any, lon: Any) -> str:
    lat_value = _maybe_float(lat)
    lon_value = _maybe_float(lon)
    if lat_value is None or lon_value is None:
        return "n/a"
    return f"{lat_value:.6f}, {lon_value:.6f}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _speed_label(gps_source_type: GpsSourceType | str | None) -> str:
    value = gps_source_type.value if isinstance(gps_source_type, GpsSourceType) else str(gps_source_type or "")
    if value == GpsSourceType.IMU_DEADRECKON.value:
        return "est. (IMU, unreliable)"
    if value == GpsSourceType.NONE.value:
        return "(no GPS)"
    return "(GPS)"


def _speed_warning(mean_speed_kmh: float, *, sample_count: int, gps_source_type: str | None = None) -> str | None:
    if gps_source_type != GpsSourceType.IMU_DEADRECKON.value:
        return None
    if sample_count > 1 and mean_speed_kmh < 5.0:
        return (
            "Mean IMU-estimated speed is below 5 km/h for this moving survey; "
            "treat speed and distance as unreliable."
        )
    return None
