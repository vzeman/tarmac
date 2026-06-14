from __future__ import annotations

import html
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw


TILE_SIZE = 1024
MAX_LOD_LEVELS = 4
GAP_COLOR = (245, 130, 32)
BACKGROUND_COLOR = (18, 24, 32)
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class StripBuildResult:
    run_dir: Path
    html_path: Path
    manifest_path: Path
    ribbon_width: int
    ribbon_height: int
    lods: list[dict[str, Any]]


@dataclass(frozen=True)
class _FrameBand:
    sample_order: int
    frame_index: int
    image_path: Path
    source_width: int
    source_height: int
    band_y0: int
    band_y1: int
    y0: int
    y1: int
    distance_start_m: float | None
    distance_end_m: float | None


def build_strip_view(
    run_dir: Path,
    *,
    band_frac: float = 0.5,
    ribbon_width: int = 512,
    tile_size: int = TILE_SIZE,
    max_lod_levels: int = MAX_LOD_LEVELS,
) -> StripBuildResult:
    """Build a continuous tiled strip viewer for a completed survey run."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not 0.0 < band_frac <= 1.0:
        raise ValueError("--band-frac must be greater than 0 and at most 1.")
    if ribbon_width < 64:
        raise ValueError("--ribbon-width must be at least 64 pixels.")
    if tile_size < 256:
        raise ValueError("tile_size must be at least 256 pixels.")

    samples_path = run_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing survey samples: {samples_path}")
    samples = _load_samples(samples_path)
    if samples.empty:
        raise ValueError(f"No sampled frames found in {samples_path}")

    frame_paths = [_resolve_frame_path(run_dir, row) for row in samples.itertuples()]
    first_width, first_height = _image_size(frame_paths[0])
    crop_height = max(1, int(round(first_height * float(band_frac))))
    band_y0 = max(0, first_height - crop_height)
    band_y1 = min(first_height, band_y0 + crop_height)
    base_band_height = max(1, int(round((band_y1 - band_y0) * int(ribbon_width) / first_width)))
    spacing = _compute_spacing(samples, base_band_height=base_band_height)

    strip_dir = run_dir / "strip"
    tiles_dir = strip_dir / "tiles"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    frame_bands, position_map, gap_markers, ribbon_height = _position_frames(
        run_dir=run_dir,
        samples=samples,
        frame_paths=frame_paths,
        first_width=first_width,
        first_height=first_height,
        band_y0=band_y0,
        band_y1=band_y1,
        spacing=spacing,
    )
    lods = _build_lods(
        tiles_dir=tiles_dir,
        frame_bands=frame_bands,
        gap_markers=gap_markers,
        ribbon_width=int(ribbon_width),
        ribbon_height=ribbon_height,
        tile_size=tile_size,
        max_lod_levels=max_lod_levels,
    )
    manifest = _manifest_payload(
        run_dir=run_dir,
        samples=samples,
        position_map=position_map,
        gap_markers=gap_markers,
        lods=lods,
        ribbon_width=int(ribbon_width),
        ribbon_height=ribbon_height,
        band_frac=float(band_frac),
        band_y0=band_y0,
        band_y1=band_y1,
        base_band_height=base_band_height,
        spacing=spacing,
        tile_size=tile_size,
    )
    manifest_path = strip_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    html_path = run_dir / "strip.html"
    html_path.write_text(_viewer_html(manifest), encoding="utf-8")
    _update_summary(run_dir, html_path=html_path, manifest_path=manifest_path)
    _ensure_index_link(run_dir)
    return StripBuildResult(
        run_dir=run_dir,
        html_path=html_path,
        manifest_path=manifest_path,
        ribbon_width=int(ribbon_width),
        ribbon_height=ribbon_height,
        lods=lods,
    )


def _load_samples(samples_path: Path) -> pd.DataFrame:
    samples = pd.read_parquet(samples_path)
    sort_cols = [column for column in ["t", "frame_index"] if column in samples.columns]
    if sort_cols:
        samples = samples.sort_values(sort_cols, kind="stable")
    return samples.reset_index(drop=True)


def _resolve_frame_path(run_dir: Path, row: Any) -> Path:
    frame_value = str(getattr(row, "frame_image", "") or getattr(row, "frame_thumbnail", "") or "")
    frame_path = _run_path(run_dir, frame_value) if frame_value else None
    if frame_path is not None and frame_path.exists():
        return frame_path

    frame_index = _maybe_int(getattr(row, "frame_index", None))
    if frame_index is not None:
        matches = sorted((run_dir / "frames").glob(f"frame_{frame_index:06d}_*.jpg"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Missing sampled frame image for frame_index={frame_index}: {frame_value}")


def _run_path(run_dir: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _compute_spacing(samples: pd.DataFrame, *, base_band_height: int) -> dict[str, Any]:
    pair_distances = _pair_distances(samples)
    pair_times = _pair_times(samples)
    positive_distances = [value for value in pair_distances if value is not None and value > 0.05]
    positive_times = [value for value in pair_times if value is not None and value > 0.0]
    median_distance = _median(positive_distances)
    median_dt = _median(positive_times) or 1.0
    has_gps_spacing = median_distance is not None and median_distance >= 0.2
    speed_distances = _speed_distances(samples, pair_times)
    fallback_distance = median_distance or _median([value for value in speed_distances if value and value > 0.05])

    if has_gps_spacing and fallback_distance:
        pixels_per_meter = float(base_band_height) / float(fallback_distance)
        distance_source = "gps"
    elif fallback_distance:
        pixels_per_meter = float(base_band_height) / float(fallback_distance)
        distance_source = "speed"
    else:
        pixels_per_meter = None
        distance_source = "uniform"

    distances: list[float | None] = []
    for gps_distance, speed_distance in zip(pair_distances, speed_distances, strict=True):
        value: float | None = None
        if distance_source == "gps" and gps_distance is not None:
            value = gps_distance
        elif distance_source in {"gps", "speed"} and speed_distance is not None:
            value = speed_distance
        elif fallback_distance is not None:
            value = fallback_distance
        distances.append(value)
    if samples.shape[0] > 0:
        distances.append(fallback_distance)

    gap_markers: list[dict[str, Any]] = []
    gap_distance_threshold = max(25.0, float(median_distance or 0.0) * 5.0)
    gap_time_threshold = max(5.0, float(median_dt) * 3.0)
    for index, (distance, dt) in enumerate(zip(pair_distances, pair_times, strict=True)):
        reasons: list[str] = []
        if dt is not None and dt > gap_time_threshold:
            reasons.append("time_gap")
        if distance is not None and distance > gap_distance_threshold:
            reasons.append("gps_gap")
        if reasons:
            gap_markers.append(
                {
                    "after_sample_order": index,
                    "reason": "+".join(reasons),
                    "distance_m": _json_float(distance),
                    "dt_s": _json_float(dt),
                }
            )

    return {
        "distance_source": distance_source,
        "pixels_per_meter": pixels_per_meter,
        "base_band_height": int(base_band_height),
        "distances_m": distances,
        "median_distance_m": median_distance,
        "median_dt_s": median_dt,
        "gap_specs": gap_markers,
        "gap_height_px": max(4, int(round(base_band_height * 0.06))),
        "min_frame_height_px": 8 if distance_source != "uniform" else int(base_band_height),
    }


def _pair_distances(samples: pd.DataFrame) -> list[float | None]:
    distances: list[float | None] = []
    rows = list(samples.itertuples())
    for current, nxt in zip(rows, rows[1:], strict=False):
        lat1 = _maybe_float(getattr(current, "lat", None))
        lon1 = _maybe_float(getattr(current, "lon", None))
        lat2 = _maybe_float(getattr(nxt, "lat", None))
        lon2 = _maybe_float(getattr(nxt, "lon", None))
        if _valid_lat_lon(lat1, lon1) and _valid_lat_lon(lat2, lon2):
            distances.append(_haversine_m(lat1, lon1, lat2, lon2))
        else:
            distances.append(None)
    return distances


def _pair_times(samples: pd.DataFrame) -> list[float | None]:
    times: list[float | None] = []
    rows = list(samples.itertuples())
    for current, nxt in zip(rows, rows[1:], strict=False):
        t0 = _maybe_float(getattr(current, "t", None))
        t1 = _maybe_float(getattr(nxt, "t", None))
        times.append(t1 - t0 if t0 is not None and t1 is not None else None)
    return times


def _speed_distances(samples: pd.DataFrame, pair_times: list[float | None]) -> list[float | None]:
    rows = list(samples.itertuples())
    values: list[float | None] = []
    for current, nxt, dt in zip(rows, rows[1:], pair_times, strict=False):
        if dt is None or dt < 0.0:
            values.append(None)
            continue
        speed0 = _maybe_float(getattr(current, "speed_mps", None))
        speed1 = _maybe_float(getattr(nxt, "speed_mps", None))
        speeds = [value for value in [speed0, speed1] if value is not None and value >= 0.0]
        if speeds:
            values.append(float(sum(speeds) / len(speeds)) * float(dt))
        else:
            values.append(None)
    return values


def _position_frames(
    *,
    run_dir: Path,
    samples: pd.DataFrame,
    frame_paths: list[Path],
    first_width: int,
    first_height: int,
    band_y0: int,
    band_y1: int,
    spacing: dict[str, Any],
) -> tuple[list[_FrameBand], list[dict[str, Any]], list[dict[str, Any]], int]:
    pixels_per_meter = spacing.get("pixels_per_meter")
    base_band_height = int(spacing["base_band_height"])
    min_height = int(spacing["min_frame_height_px"])
    distances = list(spacing["distances_m"])
    gap_specs = {int(item["after_sample_order"]): item for item in spacing["gap_specs"]}
    gap_height = int(spacing["gap_height_px"])

    frame_bands: list[_FrameBand] = []
    position_map: list[dict[str, Any]] = []
    gap_markers: list[dict[str, Any]] = []
    y = 0
    distance_m = 0.0
    for sample_order, (row, frame_path) in enumerate(zip(samples.itertuples(), frame_paths, strict=True)):
        source_width, source_height = _image_size(frame_path)
        if source_width != first_width or source_height != first_height:
            crop_height = max(1, int(round(source_height * ((band_y1 - band_y0) / first_height))))
            row_band_y0 = max(0, source_height - crop_height)
            row_band_y1 = min(source_height, row_band_y0 + crop_height)
        else:
            row_band_y0 = band_y0
            row_band_y1 = band_y1

        segment_distance = distances[sample_order] if sample_order < len(distances) else None
        if pixels_per_meter is not None and segment_distance is not None:
            frame_height = max(min_height, int(round(max(0.0, float(segment_distance)) * float(pixels_per_meter))))
        else:
            frame_height = base_band_height
        y0 = y
        y1 = y + max(1, frame_height)
        distance_start = distance_m if pixels_per_meter is not None else None
        if segment_distance is not None and pixels_per_meter is not None:
            distance_m += max(0.0, float(segment_distance))
        distance_end = distance_m if pixels_per_meter is not None else None
        frame_index = _maybe_int(getattr(row, "frame_index", None)) or sample_order
        frame_bands.append(
            _FrameBand(
                sample_order=sample_order,
                frame_index=frame_index,
                image_path=frame_path,
                source_width=source_width,
                source_height=source_height,
                band_y0=row_band_y0,
                band_y1=row_band_y1,
                y0=y0,
                y1=y1,
                distance_start_m=distance_start,
                distance_end_m=distance_end,
            )
        )
        position_map.append(
            _frame_manifest_record(
                run_dir=run_dir,
                row=row,
                sample_order=sample_order,
                frame_index=frame_index,
                y0=y0,
                y1=y1,
                distance_start_m=distance_start,
                distance_end_m=distance_end,
            )
        )
        y = y1
        if sample_order in gap_specs:
            gap_y0 = y
            gap_y1 = y + gap_height
            gap = dict(gap_specs[sample_order])
            gap.update({"y0": gap_y0, "y1": gap_y1})
            gap_markers.append(gap)
            y = gap_y1

    return frame_bands, position_map, gap_markers, max(1, y)


def _frame_manifest_record(
    *,
    run_dir: Path,
    row: Any,
    sample_order: int,
    frame_index: int,
    y0: int,
    y1: int,
    distance_start_m: float | None,
    distance_end_m: float | None,
) -> dict[str, Any]:
    issues = _json_list(getattr(row, "issues", "[]"))
    frame_image = str(getattr(row, "frame_image", "") or getattr(row, "frame_thumbnail", "") or "")
    problem_image = str(getattr(row, "problem_image", "") or "")
    thumbnail_image = str(getattr(row, "thumbnail_image", "") or "")
    marked_image = _marked_image_for_row(run_dir, row)
    return {
        "sample_order": int(sample_order),
        "frame_index": int(frame_index),
        "t": _json_float(getattr(row, "t", None)) or 0.0,
        "timestamp": str(getattr(row, "timestamp", "") or _timestamp_label(_maybe_float(getattr(row, "t", None)) or 0.0)),
        "y0": int(y0),
        "y1": int(y1),
        "y_center": float((y0 + y1) / 2.0),
        "height_px": int(y1 - y0),
        "distance_start_m": _json_float(distance_start_m),
        "distance_end_m": _json_float(distance_end_m),
        "distance_center_m": (
            _json_float((float(distance_start_m) + float(distance_end_m)) / 2.0)
            if distance_start_m is not None and distance_end_m is not None
            else None
        ),
        "lat": _json_float(getattr(row, "lat", None)),
        "lon": _json_float(getattr(row, "lon", None)),
        "speed_kmh": _json_float(getattr(row, "speed_kmh", None)),
        "heading_deg": _json_float(getattr(row, "heading_deg", None)),
        "quality_grade": _maybe_int(getattr(row, "quality_grade", None)),
        "surface_type": str(getattr(row, "surface_type", "unknown") or "unknown"),
        "confidence": _json_float(getattr(row, "confidence", None)),
        "crack_detected": bool(getattr(row, "crack_detected", False)),
        "crack_confirmed": bool(getattr(row, "crack_confirmed", False)),
        "crack_area_pct": _json_float(getattr(row, "crack_area_pct", None)),
        "crack_length_px": _maybe_int(getattr(row, "crack_length_px", None)),
        "is_problem": bool(getattr(row, "is_problem", False)),
        "issues": issues,
        "frame_image": frame_image,
        "problem_image": problem_image,
        "thumbnail_image": thumbnail_image,
        "marked_image": marked_image,
        "marked_image_filename": Path(marked_image).name if marked_image else "",
        "route_approximate": bool(getattr(row, "route_approximate", False)),
        "telemetry_source": str(getattr(row, "telemetry_source", "") or ""),
    }


def _marked_image_for_row(run_dir: Path, row: Any) -> str:
    explicit = str(getattr(row, "crack_overlay_image", "") or "")
    if explicit:
        return explicit
    problem_image = str(getattr(row, "problem_image", "") or "")
    if not problem_image:
        return ""
    problem_path = _run_path(run_dir, problem_image)
    if problem_path is None:
        return ""
    marked_path = problem_path.with_name(f"{problem_path.stem}_marked{problem_path.suffix}")
    if marked_path.exists():
        return str(marked_path.relative_to(run_dir))
    return ""


def _build_lods(
    *,
    tiles_dir: Path,
    frame_bands: list[_FrameBand],
    gap_markers: list[dict[str, Any]],
    ribbon_width: int,
    ribbon_height: int,
    tile_size: int,
    max_lod_levels: int,
) -> list[dict[str, Any]]:
    levels = max(1, int(max_lod_levels))
    lods: list[dict[str, Any]] = []
    for level in range(levels):
        scale = 2**level
        width = max(1, int(math.ceil(ribbon_width / scale)))
        height = max(1, int(math.ceil(ribbon_height / scale)))
        cols = max(1, int(math.ceil(width / tile_size)))
        rows = max(1, int(math.ceil(height / tile_size)))
        level_dir = tiles_dir / f"z{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                tile = _render_tile(
                    frame_bands=frame_bands,
                    gap_markers=gap_markers,
                    ribbon_width=ribbon_width,
                    level=level,
                    col=col,
                    row=row,
                    lod_width=width,
                    lod_height=height,
                    tile_size=tile_size,
                )
                tile.save(level_dir / f"{index}.jpg", format="JPEG", quality=84, optimize=True)
        lods.append(
            {
                "level": int(level),
                "scale": int(scale),
                "width": int(width),
                "height": int(height),
                "cols": int(cols),
                "rows": int(rows),
                "tile_count": int(cols * rows),
            }
        )
    return lods


def _render_tile(
    *,
    frame_bands: list[_FrameBand],
    gap_markers: list[dict[str, Any]],
    ribbon_width: int,
    level: int,
    col: int,
    row: int,
    lod_width: int,
    lod_height: int,
    tile_size: int,
) -> Image.Image:
    scale = 2**level
    lod_x0 = col * tile_size
    lod_y0 = row * tile_size
    lod_x1 = min(lod_width, lod_x0 + tile_size)
    lod_y1 = min(lod_height, lod_y0 + tile_size)
    tile_width = max(1, lod_x1 - lod_x0)
    tile_height = max(1, lod_y1 - lod_y0)
    tile = Image.new("RGB", (tile_width, tile_height), BACKGROUND_COLOR)
    base_x0 = lod_x0 * scale
    base_x1 = min(ribbon_width, lod_x1 * scale)
    for frame in frame_bands:
        base_y0 = max(frame.y0, lod_y0 * scale)
        base_y1 = min(frame.y1, lod_y1 * scale)
        if base_y1 <= base_y0 or base_x1 <= base_x0:
            continue
        dest_x0 = int(math.floor(base_x0 / scale - lod_x0))
        dest_x1 = int(math.ceil(base_x1 / scale - lod_x0))
        dest_y0 = int(math.floor(base_y0 / scale - lod_y0))
        dest_y1 = int(math.ceil(base_y1 / scale - lod_y0))
        dest_width = max(1, min(tile_width, dest_x1) - max(0, dest_x0))
        dest_height = max(1, min(tile_height, dest_y1) - max(0, dest_y0))
        crop = _crop_frame_for_base_region(frame, base_x0, base_x1, base_y0, base_y1, ribbon_width)
        band = crop.resize((dest_width, dest_height), Image.Resampling.LANCZOS)
        tile.paste(band, (max(0, dest_x0), max(0, dest_y0)))

    if gap_markers:
        draw = ImageDraw.Draw(tile)
        for gap in gap_markers:
            gap_y0 = float(gap.get("y0", 0))
            gap_y1 = float(gap.get("y1", 0))
            if gap_y1 < lod_y0 * scale or gap_y0 > lod_y1 * scale:
                continue
            y0 = int(math.floor(max(gap_y0, lod_y0 * scale) / scale - lod_y0))
            y1 = int(math.ceil(min(gap_y1, lod_y1 * scale) / scale - lod_y0))
            draw.rectangle((0, max(0, y0), tile_width, min(tile_height, max(y0 + 1, y1))), fill=GAP_COLOR)
    return tile


def _crop_frame_for_base_region(
    frame: _FrameBand,
    base_x0: float,
    base_x1: float,
    base_y0: float,
    base_y1: float,
    ribbon_width: int,
) -> Image.Image:
    source_band_height = max(1, frame.band_y1 - frame.band_y0)
    frame_height = max(1, frame.y1 - frame.y0)
    sx0 = int(math.floor((base_x0 / ribbon_width) * frame.source_width))
    sx1 = int(math.ceil((base_x1 / ribbon_width) * frame.source_width))
    sy0 = int(math.floor(frame.band_y0 + ((base_y0 - frame.y0) / frame_height) * source_band_height))
    sy1 = int(math.ceil(frame.band_y0 + ((base_y1 - frame.y0) / frame_height) * source_band_height))
    sx0 = min(max(0, sx0), frame.source_width - 1)
    sx1 = min(max(sx0 + 1, sx1), frame.source_width)
    sy0 = min(max(frame.band_y0, sy0), frame.band_y1 - 1)
    sy1 = min(max(sy0 + 1, sy1), frame.band_y1)
    with Image.open(frame.image_path) as image:
        return image.convert("RGB").crop((sx0, sy0, sx1, sy1))


def _manifest_payload(
    *,
    run_dir: Path,
    samples: pd.DataFrame,
    position_map: list[dict[str, Any]],
    gap_markers: list[dict[str, Any]],
    lods: list[dict[str, Any]],
    ribbon_width: int,
    ribbon_height: int,
    band_frac: float,
    band_y0: int,
    band_y1: int,
    base_band_height: int,
    spacing: dict[str, Any],
    tile_size: int,
) -> dict[str, Any]:
    total_distance = None
    distances = [row.get("distance_end_m") for row in position_map if row.get("distance_end_m") is not None]
    if distances:
        total_distance = max(float(value) for value in distances)
    return {
        "version": 1,
        "kind": "tarmac-continuous-strip",
        "run_name": run_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "samples": "samples.parquet",
            "frames_dir": "frames",
            "sample_count": int(len(samples)),
        },
        "ribbon": {
            "width": int(ribbon_width),
            "height": int(ribbon_height),
            "band_frac": float(band_frac),
            "band_y0": int(band_y0),
            "band_y1": int(band_y1),
            "base_band_height": int(base_band_height),
            "distance_source": str(spacing.get("distance_source", "uniform")),
            "pixels_per_meter": _json_float(spacing.get("pixels_per_meter")),
            "median_distance_m": _json_float(spacing.get("median_distance_m")),
            "median_dt_s": _json_float(spacing.get("median_dt_s")),
            "total_distance_m": _json_float(total_distance),
        },
        "tile_size": int(tile_size),
        "tiles_root": "strip/tiles",
        "lods": lods,
        "gap_markers": gap_markers,
        "position_to_frame": position_map,
        "viewer": {
            "max_tile_cache": 60,
            "tile_path_template": "strip/tiles/z{level}/{index}.jpg",
        },
    }


def _viewer_html(manifest: dict[str, Any]) -> str:
    title = html.escape(f"Tarmac continuous strip - {manifest.get('run_name', 'survey')}")
    manifest_json = json.dumps(manifest, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101820;
      --panel: rgba(17, 24, 39, 0.88);
      --line: rgba(255, 255, 255, 0.16);
      --text: #eef2f7;
      --muted: #aab4c0;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    #viewer { width: 100vw; height: 100vh; display: block; cursor: grab; touch-action: none; background: #111820; }
    #viewer.dragging { cursor: grabbing; }
    .topbar { position: fixed; left: 12px; top: 12px; display: flex; gap: 8px; align-items: center; z-index: 3; }
    .readout { position: fixed; right: 12px; top: 12px; z-index: 3; max-width: min(520px, calc(100vw - 24px)); padding: 9px 11px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); color: var(--text); font-size: 13px; line-height: 1.35; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25); overflow-wrap: anywhere; }
    button, .link { min-width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); color: var(--text); font-weight: 700; text-decoration: none; display: inline-flex; align-items: center; justify-content: center; padding: 0 10px; cursor: pointer; }
    button:hover, .link:hover { border-color: rgba(56, 189, 248, 0.8); }
    .popup { position: fixed; left: 16px; bottom: 16px; z-index: 5; width: min(760px, calc(100vw - 32px)); max-height: min(620px, calc(100vh - 32px)); overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: rgba(15, 23, 42, 0.96); box-shadow: 0 18px 60px rgba(0,0,0,0.45); display: none; }
    .popup.open { display: block; }
    .popup header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid var(--line); }
    .popup h2 { margin: 0; font-size: 16px; line-height: 1.2; }
    .popup .body { padding: 12px; }
    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-bottom: 12px; color: var(--muted); font-size: 13px; }
    .meta b { color: var(--text); font-weight: 700; }
    .images { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
    figure { margin: 0; }
    figcaption { margin-bottom: 6px; color: var(--muted); font-size: 12px; }
    figure img { width: 100%; max-height: 420px; object-fit: contain; background: #05070a; border: 1px solid var(--line); border-radius: 6px; display: block; }
    .empty { min-height: 180px; border: 1px solid var(--line); border-radius: 6px; display: flex; align-items: center; justify-content: center; color: var(--muted); background: #05070a; }
    .legend { position: fixed; left: 12px; bottom: 12px; z-index: 3; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); color: var(--muted); font-size: 12px; }
    .swatch { width: 18px; height: 8px; border-radius: 3px; display: inline-block; margin-right: 4px; border: 1px solid rgba(255,255,255,0.16); }
  </style>
</head>
<body>
  <canvas id="viewer"></canvas>
  <div class="topbar">
    <a class="link" href="index.html" title="Back to run index">Index</a>
    <button id="zoomIn" title="Zoom in">+</button>
    <button id="zoomOut" title="Zoom out">-</button>
    <button id="reset" title="Fit strip">Fit</button>
  </div>
  <div id="readout" class="readout"></div>
  <div class="legend">
    <span><i class="swatch" style="background:#1a9850"></i>Q1</span>
    <span><i class="swatch" style="background:#91cf60"></i>Q2</span>
    <span><i class="swatch" style="background:#fee08b"></i>Q3</span>
    <span><i class="swatch" style="background:#fc8d59"></i>Q4</span>
    <span><i class="swatch" style="background:#d73027"></i>Q5</span>
    <span><i class="swatch" style="background:#e11d48"></i>crack</span>
    <span><i class="swatch" style="background:#f58220"></i>gap</span>
  </div>
  <aside id="popup" class="popup" aria-live="polite">
    <header>
      <h2 id="popupTitle"></h2>
      <button id="popupClose" title="Close">x</button>
    </header>
    <div class="body">
      <div id="popupMeta" class="meta"></div>
      <div class="images">
        <figure>
          <figcaption>Original frame</figcaption>
          <img id="originalImage" alt="Original survey frame">
        </figure>
        <figure>
          <figcaption>Marked crack image</figcaption>
          <div id="markedEmpty" class="empty">No marked image</div>
          <img id="markedImage" alt="Marked crack overlay" style="display:none">
        </figure>
      </div>
    </div>
  </aside>
  <script id="manifest-data" type="application/json">__MANIFEST_JSON__</script>
  <script>
    const manifest = JSON.parse(document.getElementById('manifest-data').textContent);
    const canvas = document.getElementById('viewer');
    const ctx = canvas.getContext('2d', { alpha: false });
    const readout = document.getElementById('readout');
    const popup = document.getElementById('popup');
    const popupTitle = document.getElementById('popupTitle');
    const popupMeta = document.getElementById('popupMeta');
    const originalImage = document.getElementById('originalImage');
    const markedImage = document.getElementById('markedImage');
    const markedEmpty = document.getElementById('markedEmpty');
    const qualityColors = { 1: '#1a9850', 2: '#91cf60', 3: '#fee08b', 4: '#fc8d59', 5: '#d73027' };
    const frames = manifest.position_to_frame || [];
    const lods = manifest.lods || [];
    const tileSize = manifest.tile_size || 1024;
    const maxTiles = manifest.viewer?.max_tile_cache || 60;
    // LRU tile cache: only viewport-intersecting tiles are requested, and old
    // non-visible image references are dropped so the browser can reclaim them.
    const cache = new Map();
    const pointers = new Map();
    let cssWidth = 1;
    let cssHeight = 1;
    let dpr = 1;
    let renderQueued = false;
    let dragging = false;
    let dragStart = null;
    let pinchStart = null;
    let activeLod = lods[0] || { level: 0, scale: 1, width: manifest.ribbon.width, height: manifest.ribbon.height, cols: 1, rows: 1 };
    const view = { x: 0, y: 0, scale: 1 };

    function resize() {
      dpr = Math.max(1, window.devicePixelRatio || 1);
      cssWidth = Math.max(1, window.innerWidth);
      cssHeight = Math.max(1, window.innerHeight);
      canvas.width = Math.round(cssWidth * dpr);
      canvas.height = Math.round(cssHeight * dpr);
      canvas.style.width = cssWidth + 'px';
      canvas.style.height = cssHeight + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      clampView();
      requestRender();
    }

    function fitView() {
      const widthScale = (cssWidth * 0.76) / Math.max(1, manifest.ribbon.width);
      view.scale = clamp(widthScale, minScale(), 2.5);
      view.x = -Math.max(20, (cssWidth / view.scale - manifest.ribbon.width) / 2);
      view.y = 0;
      clampView();
      requestRender();
    }

    function minScale() {
      return Math.max(0.015, Math.min(cssWidth / Math.max(1, manifest.ribbon.width * 3), cssHeight / Math.max(1, manifest.ribbon.height)));
    }

    function maxScale() {
      return 8;
    }

    function clampView() {
      const worldW = cssWidth / view.scale;
      const worldH = cssHeight / view.scale;
      const marginX = Math.max(48 / view.scale, manifest.ribbon.width * 0.2);
      const marginY = Math.max(80 / view.scale, 200);
      view.x = clamp(view.x, -marginX, Math.max(marginX, manifest.ribbon.width - worldW + marginX));
      view.y = clamp(view.y, -marginY, Math.max(marginY, manifest.ribbon.height - worldH + marginY));
    }

    function chooseLod() {
      let chosen = lods[0] || activeLod;
      const basePixelsPerScreenPixel = 1 / Math.max(view.scale, 0.00001);
      for (const lod of lods) {
        if (lod.scale <= basePixelsPerScreenPixel * 1.35) chosen = lod;
      }
      return chosen;
    }

    function requestRender() {
      if (renderQueued) return;
      renderQueued = true;
      requestAnimationFrame(render);
    }

    function render() {
      renderQueued = false;
      activeLod = chooseLod();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#101820';
      ctx.fillRect(0, 0, cssWidth, cssHeight);
      clampView();
      const visible = drawTiles(activeLod);
      drawMarkers();
      drawRibbonBorder();
      evictTiles(visible);
      updateReadout();
    }

    function drawTiles(lod) {
      // Virtualized tile loading: compute the current viewport in the selected
      // LOD and request just those tile indexes.
      const visible = new Set();
      const factor = lod.scale || Math.pow(2, lod.level || 0);
      const x0 = Math.max(0, Math.floor((view.x / factor) / tileSize));
      const y0 = Math.max(0, Math.floor((view.y / factor) / tileSize));
      const x1 = Math.min(lod.cols - 1, Math.floor(((view.x + cssWidth / view.scale) / factor) / tileSize));
      const y1 = Math.min(lod.rows - 1, Math.floor(((view.y + cssHeight / view.scale) / factor) / tileSize));
      for (let row = y0; row <= y1; row++) {
        for (let col = x0; col <= x1; col++) {
          const index = row * lod.cols + col;
          const key = `${lod.level}:${index}`;
          visible.add(key);
          const entry = requestTile(lod, index, key);
          if (!entry.loaded || !entry.image) continue;
          const dx = (col * tileSize * factor - view.x) * view.scale;
          const dy = (row * tileSize * factor - view.y) * view.scale;
          const dw = entry.image.naturalWidth * factor * view.scale;
          const dh = entry.image.naturalHeight * factor * view.scale;
          ctx.drawImage(entry.image, dx, dy, dw, dh);
        }
      }
      return visible;
    }

    function requestTile(lod, index, key) {
      let entry = cache.get(key);
      if (entry) {
        entry.last = performance.now();
        return entry;
      }
      const image = new Image();
      entry = { image, loaded: false, failed: false, last: performance.now() };
      cache.set(key, entry);
      image.onload = () => {
        entry.loaded = true;
        entry.last = performance.now();
        requestRender();
      };
      image.onerror = () => {
        entry.failed = true;
      };
      image.src = tileUrl(lod.level, index);
      return entry;
    }

    function tileUrl(level, index) {
      return `${manifest.tiles_root}/z${level}/${index}.jpg`;
    }

    function evictTiles(visible) {
      if (cache.size <= maxTiles) return;
      const entries = Array.from(cache.entries())
        .filter(([key]) => !visible.has(key))
        .sort((a, b) => a[1].last - b[1].last);
      while (cache.size > maxTiles && entries.length) {
        const [key, entry] = entries.shift();
        if (entry.image) {
          entry.image.onload = null;
          entry.image.onerror = null;
          entry.image.src = '';
          entry.image = null;
        }
        cache.delete(key);
      }
    }

    function drawMarkers() {
      const top = view.y;
      const bottom = view.y + cssHeight / view.scale;
      const stripLeft = (0 - view.x) * view.scale;
      const stripRight = (manifest.ribbon.width - view.x) * view.scale;
      for (const frame of frames) {
        if (frame.y1 < top || frame.y0 > bottom) continue;
        const y = (frame.y_center - view.y) * view.scale;
        const q = Number(frame.quality_grade || 0);
        const color = frame.crack_confirmed ? '#e11d48' : (qualityColors[q] || '#94a3b8');
        const h = Math.max(2, Math.min(12, (frame.y1 - frame.y0) * view.scale));
        ctx.fillStyle = color;
        ctx.fillRect(stripLeft, y - h / 2, Math.max(5, 10 * view.scale), h);
        if (frame.is_problem || frame.crack_confirmed || q >= 4) {
          ctx.beginPath();
          ctx.arc(stripRight + 10, y, frame.crack_confirmed ? 5 : 4, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = 'rgba(0,0,0,0.5)';
          ctx.stroke();
        }
      }
      for (const gap of manifest.gap_markers || []) {
        const y0 = (gap.y0 - view.y) * view.scale;
        const y1 = (gap.y1 - view.y) * view.scale;
        if (y1 < 0 || y0 > cssHeight) continue;
        ctx.fillStyle = '#f58220';
        ctx.fillRect(stripLeft, y0, Math.max(1, stripRight - stripLeft), Math.max(2, y1 - y0));
      }
    }

    function drawRibbonBorder() {
      const x = (0 - view.x) * view.scale;
      const y = (0 - view.y) * view.scale;
      const w = manifest.ribbon.width * view.scale;
      const h = manifest.ribbon.height * view.scale;
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, w, h);
    }

    function updateReadout() {
      const center = { x: view.x + cssWidth / (2 * view.scale), y: view.y + cssHeight / (2 * view.scale) };
      const frame = frameAtY(center.y);
      const distance = frame?.distance_center_m != null ? `${(frame.distance_center_m / 1000).toFixed(3)} km` : 'n/a';
      const gps = frame?.lat != null && frame?.lon != null ? `${frame.lat.toFixed(6)}, ${frame.lon.toFixed(6)}` : 'n/a';
      const quality = frame?.quality_grade != null ? `Q${frame.quality_grade}` : 'Q n/a';
      const tilesLoaded = Array.from(cache.values()).filter(v => v.loaded).length;
      readout.textContent = `LOD z${activeLod.level} | scale ${view.scale.toFixed(3)} | frame ${frame?.frame_index ?? 'n/a'} | ${distance} | GPS ${gps} | ${quality} | tiles ${tilesLoaded}/${maxTiles}`;
    }

    function frameAtY(y) {
      if (!frames.length) return null;
      let lo = 0;
      let hi = frames.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const frame = frames[mid];
        if (y < frame.y0) hi = mid - 1;
        else if (y > frame.y1) lo = mid + 1;
        else return frame;
      }
      const near = Math.max(0, Math.min(frames.length - 1, lo));
      const a = frames[near];
      const b = frames[Math.max(0, near - 1)];
      if (!b) return a;
      return Math.abs(a.y_center - y) < Math.abs(b.y_center - y) ? a : b;
    }

    function openFrame(frame) {
      if (!frame) return;
      popup.classList.add('open');
      popupTitle.textContent = `Frame ${frame.frame_index} · ${frame.timestamp || secondsLabel(frame.t || 0)}`;
      const distance = frame.distance_center_m != null ? `${frame.distance_center_m.toFixed(1)} m` : 'n/a';
      const gps = frame.lat != null && frame.lon != null ? `${frame.lat.toFixed(6)}, ${frame.lon.toFixed(6)}` : 'n/a';
      const issues = frame.issues?.length ? frame.issues.join(', ') : 'none';
      popupMeta.innerHTML = [
        ['Distance', distance],
        ['GPS', gps],
        ['Speed', frame.speed_kmh != null ? `${frame.speed_kmh.toFixed(1)} km/h` : 'n/a'],
        ['Quality', frame.quality_grade != null ? `Q${frame.quality_grade}` : 'n/a'],
        ['Surface', frame.surface_type || 'unknown'],
        ['Issues', issues],
        ['Crack area', frame.crack_area_pct != null ? `${frame.crack_area_pct.toFixed(3)}%` : 'n/a'],
        ['Marked file', frame.marked_image_filename || 'none']
      ].map(([label, value]) => `<div><b>${escapeHtml(label)}:</b> ${escapeHtml(value)}</div>`).join('');
      originalImage.src = frame.problem_image || frame.frame_image || frame.thumbnail_image || '';
      if (frame.marked_image) {
        markedImage.style.display = 'block';
        markedEmpty.style.display = 'none';
        markedImage.src = frame.marked_image;
      } else {
        markedImage.removeAttribute('src');
        markedImage.style.display = 'none';
        markedEmpty.style.display = 'flex';
      }
    }

    function zoomAt(clientX, clientY, factor) {
      const rect = canvas.getBoundingClientRect();
      const sx = clientX - rect.left;
      const sy = clientY - rect.top;
      const before = screenToWorld(sx, sy);
      view.scale = clamp(view.scale * factor, minScale(), maxScale());
      view.x = before.x - sx / view.scale;
      view.y = before.y - sy / view.scale;
      clampView();
      requestRender();
    }

    function screenToWorld(x, y) {
      return { x: view.x + x / view.scale, y: view.y + y / view.scale };
    }

    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      zoomAt(event.clientX, event.clientY, Math.exp(-event.deltaY * 0.001));
    }, { passive: false });

    canvas.addEventListener('pointerdown', event => {
      canvas.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (pointers.size === 1) {
        dragging = true;
        canvas.classList.add('dragging');
        dragStart = { clientX: event.clientX, clientY: event.clientY, x: view.x, y: view.y, moved: false };
      } else if (pointers.size === 2) {
        pinchStart = pinchSnapshot();
      }
    });

    canvas.addEventListener('pointermove', event => {
      if (!pointers.has(event.pointerId)) return;
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (pointers.size === 2 && pinchStart) {
        const current = pinchSnapshot();
        const nextScale = clamp(pinchStart.scale * (current.distance / Math.max(1, pinchStart.distance)), minScale(), maxScale());
        view.scale = nextScale;
        view.x = pinchStart.world.x - current.mid.x / view.scale;
        view.y = pinchStart.world.y - current.mid.y / view.scale;
        clampView();
        requestRender();
        return;
      }
      if (!dragging || !dragStart) return;
      const dx = event.clientX - dragStart.clientX;
      const dy = event.clientY - dragStart.clientY;
      if (Math.hypot(dx, dy) > 4) dragStart.moved = true;
      view.x = dragStart.x - dx / view.scale;
      view.y = dragStart.y - dy / view.scale;
      clampView();
      requestRender();
    });

    function endPointer(event) {
      const pointer = pointers.get(event.pointerId);
      pointers.delete(event.pointerId);
      if (pointers.size < 2) pinchStart = null;
      if (dragging && dragStart && pointer && !dragStart.moved) {
        const rect = canvas.getBoundingClientRect();
        const world = screenToWorld(pointer.x - rect.left, pointer.y - rect.top);
        openFrame(frameAtY(world.y));
      }
      if (!pointers.size) {
        dragging = false;
        canvas.classList.remove('dragging');
        dragStart = null;
      }
    }

    canvas.addEventListener('pointerup', endPointer);
    canvas.addEventListener('pointercancel', endPointer);
    canvas.addEventListener('pointerleave', event => {
      if (event.buttons === 0) endPointer(event);
    });

    function pinchSnapshot() {
      const rect = canvas.getBoundingClientRect();
      const values = Array.from(pointers.values()).slice(0, 2);
      const mid = { x: ((values[0].x + values[1].x) / 2) - rect.left, y: ((values[0].y + values[1].y) / 2) - rect.top };
      const distance = Math.hypot(values[0].x - values[1].x, values[0].y - values[1].y);
      return { mid, distance, scale: view.scale, world: screenToWorld(mid.x, mid.y) };
    }

    document.getElementById('zoomIn').addEventListener('click', () => zoomAt(cssWidth / 2, cssHeight / 2, 1.25));
    document.getElementById('zoomOut').addEventListener('click', () => zoomAt(cssWidth / 2, cssHeight / 2, 0.8));
    document.getElementById('reset').addEventListener('click', fitView);
    document.getElementById('popupClose').addEventListener('click', () => popup.classList.remove('open'));
    window.addEventListener('resize', resize);

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
    }

    function secondsLabel(seconds) {
      const minutes = Math.floor(seconds / 60);
      const rest = seconds - minutes * 60;
      return `${String(minutes).padStart(2, '0')}:${rest.toFixed(2).padStart(5, '0')}`;
    }

    function clamp(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    resize();
    fitView();
  </script>
</body>
</html>
"""
    return template.replace("__TITLE__", title).replace("__MANIFEST_JSON__", manifest_json)


def _update_summary(run_dir: Path, *, html_path: Path, manifest_path: Path) -> None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    summary["strip_html"] = str(html_path)
    summary["strip_manifest"] = str(manifest_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _ensure_index_link(run_dir: Path) -> None:
    index_path = run_dir / "index.html"
    if not index_path.exists():
        return
    text = index_path.read_text(encoding="utf-8")
    if 'href="strip.html"' in text:
        return
    link = '      <a href="strip.html">Open continuous strip</a>\n'
    if '      <a href="summary.json">Open summary JSON</a>' in text:
        text = text.replace('      <a href="summary.json">Open summary JSON</a>\n', link + '      <a href="summary.json">Open summary JSON</a>\n', 1)
    elif "    </nav>" in text:
        text = text.replace("    </nav>", link + "    </nav>", 1)
    else:
        text += f"\n{link}"
    index_path.write_text(text, encoding="utf-8")


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _valid_lat_lon(lat: float | None, lon: float | None) -> bool:
    return (
        lat is not None
        and lon is not None
        and math.isfinite(lat)
        and math.isfinite(lon)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
        and not (abs(lat) < 1e-12 and abs(lon) < 1e-12)
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)] if str(value) else []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _json_float(value: Any) -> float | None:
    number = _maybe_float(value)
    if number is None or not math.isfinite(number):
        return None
    return float(number)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
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


def _timestamp_label(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"
