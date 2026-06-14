from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pandas as pd
from PIL import Image


TILE_SIZE = 1024
MAX_LOD_LEVELS = 4
GAP_COLOR = (245, 130, 32)
BACKGROUND_COLOR = (18, 24, 32)
EARTH_RADIUS_M = 6_371_008.8
DEFAULT_STRIP_FPS = 15.0
DEFAULT_BAND_PX = 24
DEFAULT_ROAD_BAND_FRAC = 0.5


@dataclass(frozen=True)
class StripBuildResult:
    run_dir: Path
    html_path: Path
    manifest_path: Path
    ribbon_width: int
    ribbon_height: int
    band_count: int
    strip_fps: float
    lods: list[dict[str, Any]]


@dataclass(frozen=True)
class _SourceBand:
    band_index: int
    t: float
    y0: int
    y1: int


@dataclass(frozen=True)
class _VideoInfo:
    width: int
    height: int
    fps: float | None
    duration_s: float | None


def build_strip_view(
    run_dir: Path,
    *,
    band_frac: float = DEFAULT_ROAD_BAND_FRAC,
    ribbon_width: int = 512,
    strip_fps: float = DEFAULT_STRIP_FPS,
    band_px: int = DEFAULT_BAND_PX,
    tile_size: int = TILE_SIZE,
    max_lod_levels: int = MAX_LOD_LEVELS,
) -> StripBuildResult:
    """Build a continuous push-broom tiled strip viewer for a completed survey run."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not 0.0 < band_frac <= 1.0:
        raise ValueError("--band-frac must be greater than 0 and at most 1.")
    if ribbon_width < 64:
        raise ValueError("--ribbon-width must be at least 64 pixels.")
    if strip_fps < 0.0:
        raise ValueError("--strip-fps must be zero or greater.")
    if band_px < 1:
        raise ValueError("--band-px must be at least 1 pixel.")
    if tile_size < 256:
        raise ValueError("tile_size must be at least 256 pixels.")

    samples_path = run_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing survey samples: {samples_path}")
    samples = _load_samples(samples_path)
    if samples.empty:
        raise ValueError(f"No sampled frames found in {samples_path}")

    summary = _load_summary(run_dir)
    source_video = _resolve_source_video(run_dir, summary)
    video_info = _probe_video(source_video)
    effective_duration = _effective_duration(summary, video_info)
    source_band_y0, source_band_y1, band_center_frac = _source_band_bounds(
        video_info.height,
        band_px=int(band_px),
        band_frac=float(band_frac),
    )
    strip_dir = run_dir / "strip"
    tiles_dir = strip_dir / "tiles"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    writer = _ProgressiveTileWriter(
        tiles_dir=tiles_dir,
        ribbon_width=int(ribbon_width),
        tile_size=tile_size,
    )
    source_bands: list[_SourceBand] = []
    frame_interval_s = _frame_interval_s(strip_fps=float(strip_fps), video_info=video_info)
    for band_index, band_image in enumerate(
        _stream_source_bands(
            video_path=source_video,
            ribbon_width=int(ribbon_width),
            band_px=int(band_px),
            strip_fps=float(strip_fps),
            source_band_y0=source_band_y0,
            source_band_y1=source_band_y1,
            duration_s=effective_duration,
        )
    ):
        y0, y1 = writer.append_band(band_image)
        source_bands.append(
            _SourceBand(
                band_index=band_index,
                t=float(band_index) * frame_interval_s,
                y0=y0,
                y1=y1,
            )
        )
    ribbon_height = writer.finalize()
    if not source_bands:
        raise RuntimeError(f"ffmpeg produced no frames for source video: {source_video}")

    lods = _build_lod_pyramid_from_base(
        tiles_dir=tiles_dir,
        ribbon_width=int(ribbon_width),
        ribbon_height=ribbon_height,
        tile_size=tile_size,
        max_lod_levels=max_lod_levels,
    )
    marker_rows = _load_problem_marker_rows(run_dir, summary)
    position_map, gap_markers, spacing = _dense_position_map(
        run_dir=run_dir,
        samples=samples,
        problem_rows=marker_rows,
        source_bands=source_bands,
        ribbon_height=ribbon_height,
        band_px=int(band_px),
    )
    manifest = _dense_manifest_payload(
        run_dir=run_dir,
        samples=samples,
        source_video=source_video,
        video_info=video_info,
        position_map=position_map,
        gap_markers=gap_markers,
        lods=lods,
        ribbon_width=int(ribbon_width),
        ribbon_height=ribbon_height,
        band_frac=float(band_frac),
        band_center_frac=band_center_frac,
        source_band_y0=source_band_y0,
        source_band_y1=source_band_y1,
        strip_fps=float(strip_fps),
        effective_strip_fps=(1.0 / frame_interval_s) if frame_interval_s > 0 else None,
        band_px=int(band_px),
        band_count=len(source_bands),
        duration_s=effective_duration,
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
        band_count=len(source_bands),
        strip_fps=float(strip_fps),
        lods=lods,
    )


def _load_samples(samples_path: Path) -> pd.DataFrame:
    samples = pd.read_parquet(samples_path)
    sort_cols = [column for column in ["t", "frame_index"] if column in samples.columns]
    if sort_cols:
        samples = samples.sort_values(sort_cols, kind="stable")
    return samples.reset_index(drop=True)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing survey summary: {summary_path}")
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid survey summary JSON: {summary_path}") from exc


def _resolve_source_video(run_dir: Path, summary: dict[str, Any]) -> Path:
    input_path = str(summary.get("input_path", "") or "")
    if not input_path:
        raise FileNotFoundError(f"Missing input_path in {run_dir / 'summary.json'}")
    candidates: list[Path] = []
    path = Path(input_path).expanduser()
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([run_dir / path, Path.cwd() / path])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Source video from summary input_path does not exist: {input_path}")


def _probe_video(video_path: Path) -> _VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to build strip-view tiles.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed for {video_path}: {stderr}") from exc
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    width = _maybe_int(stream.get("width")) or 0
    height = _maybe_int(stream.get("height")) or 0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine source video dimensions for {video_path}")
    fps = _parse_frame_rate(str(stream.get("avg_frame_rate") or "")) or _parse_frame_rate(str(stream.get("r_frame_rate") or ""))
    duration_s = _json_float(stream.get("duration")) or _json_float((payload.get("format") or {}).get("duration"))
    return _VideoInfo(width=width, height=height, fps=fps, duration_s=duration_s)


def _parse_frame_rate(value: str) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            den = float(denominator)
            if den == 0.0:
                return None
            rate = float(numerator) / den
        else:
            rate = float(value)
    except ValueError:
        return None
    if not math.isfinite(rate) or rate <= 0.0:
        return None
    return float(rate)


def _effective_duration(summary: dict[str, Any], video_info: _VideoInfo) -> float | None:
    effective = _json_float(summary.get("effective_duration_seconds"))
    video_duration = _json_float(summary.get("video_duration_seconds")) or video_info.duration_s
    clip_seconds = _json_float(summary.get("clip_seconds"))
    if clip_seconds is not None:
        return max(0.0, clip_seconds)
    if effective is None:
        return video_info.duration_s
    if video_duration is not None and effective >= float(video_duration) - 0.01:
        return None
    return max(0.0, effective)


def _source_band_bounds(source_height: int, *, band_px: int, band_frac: float) -> tuple[int, int, float]:
    crop_height = min(max(1, int(band_px)), max(1, int(source_height)))
    # Preserve the old "lower road band" intent: a 0.5 fraction centers the thin
    # push-broom sample at 75% frame height, usually the lower-middle roadway.
    center_frac = min(0.98, max(0.02, 1.0 - float(band_frac) / 2.0))
    center_y = int(round(float(source_height) * center_frac))
    y0 = min(max(0, center_y - crop_height // 2), max(0, int(source_height) - crop_height))
    return y0, y0 + crop_height, center_frac


def _frame_interval_s(strip_fps: float, video_info: _VideoInfo) -> float:
    if strip_fps > 0.0:
        return 1.0 / float(strip_fps)
    native_fps = video_info.fps or 30.0
    return 1.0 / max(0.001, float(native_fps))


def _stream_source_bands(
    *,
    video_path: Path,
    ribbon_width: int,
    band_px: int,
    strip_fps: float,
    source_band_y0: int,
    source_band_y1: int,
    duration_s: float | None,
) -> Iterator[Image.Image]:
    source_band_height = max(1, int(source_band_y1) - int(source_band_y0))
    filters: list[str] = []
    if strip_fps > 0.0:
        filters.append(f"fps={strip_fps:g}")
    filters.extend(
        [
            f"crop=iw:{source_band_height}:0:{int(source_band_y0)}",
            f"scale={int(ribbon_width)}:{int(band_px)}:flags=lanczos",
            "format=rgb24",
        ]
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
    ]
    if duration_s is not None and duration_s > 0.0:
        command.extend(["-t", f"{float(duration_s):.6f}"])
    command.extend(
        [
            "-vf",
            ",".join(filters),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    frame_size = int(ribbon_width) * int(band_px) * 3
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to build strip-view tiles.") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                process.kill()
                raise RuntimeError(
                    f"ffmpeg produced a partial raw frame for {video_path}: "
                    f"{len(chunk)} of {frame_size} bytes"
                )
            yield Image.frombytes("RGB", (int(ribbon_width), int(band_px)), chunk)
    finally:
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}")


class _ProgressiveTileWriter:
    def __init__(self, *, tiles_dir: Path, ribbon_width: int, tile_size: int) -> None:
        self.tiles_dir = tiles_dir
        self.ribbon_width = int(ribbon_width)
        self.tile_size = int(tile_size)
        self.cols = max(1, int(math.ceil(self.ribbon_width / self.tile_size)))
        self.level_dir = self.tiles_dir / "z0"
        self.level_dir.mkdir(parents=True, exist_ok=True)
        self.row_index = 0
        self.row_y = 0
        self.total_height = 0
        self._tiles = self._new_row_tiles()

    def append_band(self, band: Image.Image) -> tuple[int, int]:
        if band.mode != "RGB":
            band = band.convert("RGB")
        if band.width != self.ribbon_width:
            band = band.resize((self.ribbon_width, band.height), Image.Resampling.LANCZOS)
        band_y0 = self.total_height
        source_y = 0
        remaining = band.height
        while remaining > 0:
            available = self.tile_size - self.row_y
            take = min(remaining, available)
            piece = band.crop((0, source_y, self.ribbon_width, source_y + take))
            self._paste_piece(piece, self.row_y)
            self.row_y += take
            self.total_height += take
            source_y += take
            remaining -= take
            if self.row_y >= self.tile_size:
                self._flush_row(self.tile_size)
        return band_y0, self.total_height

    def finalize(self) -> int:
        if self.row_y > 0 or self.total_height == 0:
            self._flush_row(max(1, self.row_y))
        return max(1, self.total_height)

    def _new_row_tiles(self) -> list[Image.Image]:
        return [
            Image.new(
                "RGB",
                (self._tile_width(col), self.tile_size),
                BACKGROUND_COLOR,
            )
            for col in range(self.cols)
        ]

    def _tile_width(self, col: int) -> int:
        x0 = col * self.tile_size
        x1 = min(self.ribbon_width, x0 + self.tile_size)
        return max(1, x1 - x0)

    def _paste_piece(self, piece: Image.Image, dest_y: int) -> None:
        for col, tile in enumerate(self._tiles):
            x0 = col * self.tile_size
            x1 = min(self.ribbon_width, x0 + self.tile_size)
            tile.paste(piece.crop((x0, 0, x1, piece.height)), (0, dest_y))

    def _flush_row(self, height: int) -> None:
        row_height = min(self.tile_size, max(1, int(height)))
        for col, tile in enumerate(self._tiles):
            index = self.row_index * self.cols + col
            output = tile.crop((0, 0, tile.width, row_height))
            output.save(self.level_dir / f"{index}.jpg", format="JPEG", quality=84, optimize=True)
        self.row_index += 1
        self.row_y = 0
        self._tiles = self._new_row_tiles()


def _build_lod_pyramid_from_base(
    *,
    tiles_dir: Path,
    ribbon_width: int,
    ribbon_height: int,
    tile_size: int,
    max_lod_levels: int,
) -> list[dict[str, Any]]:
    levels = max(1, int(max_lod_levels))
    lods: list[dict[str, Any]] = []
    for level in range(levels):
        scale = 2**level
        width = max(1, int(math.ceil(int(ribbon_width) / scale)))
        height = max(1, int(math.ceil(int(ribbon_height) / scale)))
        cols = max(1, int(math.ceil(width / tile_size)))
        rows = max(1, int(math.ceil(height / tile_size)))
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
        if level > 0:
            _build_lod_level_from_previous(
                tiles_dir=tiles_dir,
                previous=lods[level - 1],
                current=lods[level],
                tile_size=tile_size,
            )
    return lods


def _build_lod_level_from_previous(
    *,
    tiles_dir: Path,
    previous: dict[str, Any],
    current: dict[str, Any],
    tile_size: int,
) -> None:
    level_dir = tiles_dir / f"z{int(current['level'])}"
    level_dir.mkdir(parents=True, exist_ok=True)
    prev_level_dir = tiles_dir / f"z{int(previous['level'])}"
    prev_width = int(previous["width"])
    prev_height = int(previous["height"])
    prev_cols = int(previous["cols"])
    for row in range(int(current["rows"])):
        for col in range(int(current["cols"])):
            x0 = col * tile_size
            y0 = row * tile_size
            x1 = min(int(current["width"]), x0 + tile_size)
            y1 = min(int(current["height"]), y0 + tile_size)
            prev_x0 = x0 * 2
            prev_y0 = y0 * 2
            prev_x1 = min(prev_width, x1 * 2)
            prev_y1 = min(prev_height, y1 * 2)
            region = _read_lod_region(
                level_dir=prev_level_dir,
                cols=prev_cols,
                tile_size=tile_size,
                x0=prev_x0,
                y0=prev_y0,
                x1=prev_x1,
                y1=prev_y1,
            )
            tile_width = max(1, x1 - x0)
            tile_height = max(1, y1 - y0)
            tile = region.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            index = row * int(current["cols"]) + col
            tile.save(level_dir / f"{index}.jpg", format="JPEG", quality=84, optimize=True)


def _read_lod_region(
    *,
    level_dir: Path,
    cols: int,
    tile_size: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Image.Image:
    region_width = max(1, x1 - x0)
    region_height = max(1, y1 - y0)
    region = Image.new("RGB", (region_width, region_height), BACKGROUND_COLOR)
    col0 = max(0, x0 // tile_size)
    row0 = max(0, y0 // tile_size)
    col1 = max(0, (max(x0, x1 - 1)) // tile_size)
    row1 = max(0, (max(y0, y1 - 1)) // tile_size)
    for tile_row in range(row0, row1 + 1):
        for tile_col in range(col0, col1 + 1):
            index = tile_row * int(cols) + tile_col
            path = level_dir / f"{index}.jpg"
            if not path.exists():
                continue
            with Image.open(path) as tile_image:
                tile = tile_image.convert("RGB")
                tx0 = tile_col * tile_size
                ty0 = tile_row * tile_size
                crop_x0 = max(0, x0 - tx0)
                crop_y0 = max(0, y0 - ty0)
                crop_x1 = min(tile.width, x1 - tx0)
                crop_y1 = min(tile.height, y1 - ty0)
                if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                    continue
                dest_x = tx0 + crop_x0 - x0
                dest_y = ty0 + crop_y0 - y0
                region.paste(tile.crop((crop_x0, crop_y0, crop_x1, crop_y1)), (dest_x, dest_y))
    return region


def _load_problem_marker_rows(run_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        summary.get("problems_confirmed_parquet"),
        summary.get("problems_parquet"),
        "problems_confirmed.parquet",
        "problems.parquet",
    ]
    seen: set[Path] = set()
    for value in candidates:
        if not value:
            continue
        path = _run_path(run_dir, str(value))
        if path is None or path in seen:
            continue
        seen.add(path)
        if path.exists():
            frame = pd.read_parquet(path)
            if frame.empty:
                return []
            sort_cols = [column for column in ["t", "frame_index"] if column in frame.columns]
            if sort_cols:
                frame = frame.sort_values(sort_cols, kind="stable")
            return [dict(row) for row in frame.to_dict("records")]
    return []


def _dense_position_map(
    *,
    run_dir: Path,
    samples: pd.DataFrame,
    problem_rows: list[dict[str, Any]],
    source_bands: list[_SourceBand],
    ribbon_height: int,
    band_px: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    band_times = [band.t for band in source_bands]
    band_centers = [float((band.y0 + band.y1) / 2.0) for band in source_bands]
    sample_centers = [_nearest_band_y(_maybe_float(getattr(row, "t", None)), band_times, band_centers) for row in samples.itertuples()]
    boundaries = _sample_y_boundaries(sample_centers, ribbon_height)
    distance_spans, spacing = _sample_distance_spans(samples, ribbon_height=ribbon_height, band_px=band_px)
    problem_by_sample = _problem_rows_by_sample(samples, problem_rows)

    position_map: list[dict[str, Any]] = []
    for sample_order, row in enumerate(samples.itertuples()):
        y0 = boundaries[sample_order]
        y1 = boundaries[sample_order + 1]
        frame_index = _maybe_int(getattr(row, "frame_index", None)) or sample_order
        distance_start, distance_end = distance_spans[sample_order]
        record = _frame_manifest_record(
            run_dir=run_dir,
            row=row,
            sample_order=sample_order,
            frame_index=frame_index,
            y0=y0,
            y1=y1,
            distance_start_m=distance_start,
            distance_end_m=distance_end,
        )
        problem = problem_by_sample.get(sample_order)
        if problem is not None:
            _apply_problem_overlay(record, problem, run_dir=run_dir)
        position_map.append(record)

    gap_markers = _dense_gap_markers(spacing.get("gap_specs", []), position_map, band_px=band_px)
    return position_map, gap_markers, spacing


def _nearest_band_y(t: float | None, band_times: list[float], band_centers: list[float]) -> float:
    if not band_centers:
        return 0.0
    if t is None:
        return band_centers[0]
    index = bisect_left(band_times, float(t))
    if index <= 0:
        return band_centers[0]
    if index >= len(band_times):
        return band_centers[-1]
    before = index - 1
    after = index
    return band_centers[before] if abs(band_times[before] - t) <= abs(band_times[after] - t) else band_centers[after]


def _sample_y_boundaries(sample_centers: list[float], ribbon_height: int) -> list[int]:
    if not sample_centers:
        return [0, max(1, int(ribbon_height))]
    boundaries = [0]
    count = len(sample_centers)
    for index in range(count - 1):
        midpoint = int(round((sample_centers[index] + sample_centers[index + 1]) / 2.0))
        remaining = count - index - 1
        upper = max(boundaries[-1], int(ribbon_height) - remaining)
        boundary = min(max(boundaries[-1] + 1, midpoint), upper) if ribbon_height >= count else max(boundaries[-1], midpoint)
        boundaries.append(boundary)
    boundaries.append(max(1, int(ribbon_height)))
    return boundaries


def _sample_distance_spans(
    samples: pd.DataFrame,
    *,
    ribbon_height: int,
    band_px: int,
) -> tuple[list[tuple[float | None, float | None]], dict[str, Any]]:
    pair_distances = _pair_distances(samples)
    pair_times = _pair_times(samples)
    speed_distances = _speed_distances(samples, pair_times)
    gps_median = _median([value for value in pair_distances if value is not None and value > 0.05])
    speed_median = _median([value for value in speed_distances if value is not None and value > 0.05])
    if gps_median is not None and gps_median >= 0.2:
        distance_source = "gps"
    elif speed_median is not None:
        distance_source = "speed"
    else:
        distance_source = "uniform"

    spans: list[tuple[float | None, float | None]] = []
    cumulative = 0.0
    for index in range(len(samples)):
        segment_distance: float | None = None
        if index < len(pair_distances):
            if distance_source == "gps":
                segment_distance = pair_distances[index] if pair_distances[index] is not None else speed_distances[index]
            elif distance_source == "speed":
                segment_distance = speed_distances[index]
        if segment_distance is not None and math.isfinite(segment_distance) and segment_distance >= 0.0:
            start = cumulative
            cumulative += float(segment_distance)
            spans.append((start, cumulative))
        elif distance_source == "uniform":
            spans.append((None, None))
        else:
            spans.append((cumulative, cumulative))

    median_dt = _median([value for value in pair_times if value is not None and value > 0.0])
    gap_specs = _gap_specs(pair_distances=pair_distances, pair_times=pair_times, median_distance=gps_median or speed_median, median_dt=median_dt)
    pixels_per_meter = float(ribbon_height) / cumulative if cumulative > 0.0 else None
    return spans, {
        "distance_source": distance_source,
        "pixels_per_meter": pixels_per_meter,
        "median_distance_m": gps_median or speed_median,
        "median_dt_s": median_dt,
        "total_distance_m": cumulative if cumulative > 0.0 else None,
        "gap_specs": gap_specs,
        "base_band_height": int(band_px),
    }


def _gap_specs(
    *,
    pair_distances: list[float | None],
    pair_times: list[float | None],
    median_distance: float | None,
    median_dt: float | None,
) -> list[dict[str, Any]]:
    gap_markers: list[dict[str, Any]] = []
    gap_distance_threshold = max(25.0, float(median_distance or 0.0) * 5.0)
    gap_time_threshold = max(5.0, float(median_dt or 1.0) * 3.0)
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
    return gap_markers


def _problem_rows_by_sample(samples: pd.DataFrame, problem_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    if not problem_rows or samples.empty:
        return {}
    sample_times = [_maybe_float(getattr(row, "t", None)) for row in samples.itertuples()]
    searchable_times = [float(value) if value is not None else math.inf for value in sample_times]
    frame_to_order: dict[int, int] = {}
    for sample_order, row in enumerate(samples.itertuples()):
        frame_index = _maybe_int(getattr(row, "frame_index", None))
        if frame_index is not None:
            frame_to_order[frame_index] = sample_order
    mapped: dict[int, dict[str, Any]] = {}
    for problem in problem_rows:
        frame_index = _maybe_int(problem.get("frame_index"))
        sample_order = frame_to_order.get(frame_index) if frame_index is not None else None
        if sample_order is None:
            sample_order = _nearest_sample_order(_maybe_float(problem.get("t")), searchable_times)
        if sample_order is not None:
            mapped[sample_order] = problem
    return mapped


def _nearest_sample_order(t: float | None, sample_times: list[float]) -> int | None:
    if not sample_times:
        return None
    if t is None or not math.isfinite(float(t)):
        return None
    finite = [value for value in sample_times if math.isfinite(value)]
    if not finite:
        return None
    index = bisect_left(sample_times, float(t))
    if index <= 0:
        return 0
    if index >= len(sample_times):
        return len(sample_times) - 1
    before = index - 1
    after = index
    before_value = sample_times[before]
    after_value = sample_times[after]
    if not math.isfinite(before_value):
        return after
    if not math.isfinite(after_value):
        return before
    return before if abs(before_value - t) <= abs(after_value - t) else after


def _apply_problem_overlay(record: dict[str, Any], problem: dict[str, Any], *, run_dir: Path) -> None:
    for key in [
        "quality_grade",
        "surface_type",
        "confidence",
        "crack_detected",
        "crack_confirmed",
        "crack_area_pct",
        "crack_length_px",
        "is_problem",
        "problem_image",
        "thumbnail_image",
        "telemetry_source",
    ]:
        if key in problem:
            value = problem.get(key)
            if key in {"quality_grade", "crack_length_px"}:
                record[key] = _maybe_int(value)
            elif key in {"confidence", "crack_area_pct"}:
                record[key] = _json_float(value)
            elif key in {"crack_detected", "crack_confirmed", "is_problem"}:
                record[key] = bool(value)
            else:
                record[key] = str(value or "")
    if "issues" in problem:
        record["issues"] = _json_list(problem.get("issues"))
    marked_image = _marked_image_for_row(run_dir, SimpleNamespace(**problem))
    if marked_image:
        record["marked_image"] = marked_image
        record["marked_image_filename"] = Path(marked_image).name


def _dense_gap_markers(
    gap_specs: list[dict[str, Any]],
    position_map: list[dict[str, Any]],
    *,
    band_px: int,
) -> list[dict[str, Any]]:
    gap_height = max(3, int(round(float(band_px) * 0.25)))
    markers: list[dict[str, Any]] = []
    for gap in gap_specs:
        sample_order = _maybe_int(gap.get("after_sample_order"))
        if sample_order is None or sample_order < 0 or sample_order >= len(position_map):
            continue
        center = int(position_map[sample_order]["y1"])
        item = dict(gap)
        item.update({"y0": max(0, center - gap_height // 2), "y1": center + gap_height})
        markers.append(item)
    return markers


def _dense_manifest_payload(
    *,
    run_dir: Path,
    samples: pd.DataFrame,
    source_video: Path,
    video_info: _VideoInfo,
    position_map: list[dict[str, Any]],
    gap_markers: list[dict[str, Any]],
    lods: list[dict[str, Any]],
    ribbon_width: int,
    ribbon_height: int,
    band_frac: float,
    band_center_frac: float,
    source_band_y0: int,
    source_band_y1: int,
    strip_fps: float,
    effective_strip_fps: float | None,
    band_px: int,
    band_count: int,
    duration_s: float | None,
    spacing: dict[str, Any],
    tile_size: int,
) -> dict[str, Any]:
    total_distance = spacing.get("total_distance_m")
    if total_distance is None:
        distances = [row.get("distance_end_m") for row in position_map if row.get("distance_end_m") is not None]
        if distances:
            total_distance = max(float(value) for value in distances)
    return {
        "version": 2,
        "kind": "tarmac-continuous-strip",
        "run_name": run_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_path": str(source_video),
            "samples": "samples.parquet",
            "sample_count": int(len(samples)),
            "source_video_width": int(video_info.width),
            "source_video_height": int(video_info.height),
            "source_video_fps": _json_float(video_info.fps),
            "source_video_duration_s": _json_float(video_info.duration_s),
        },
        "ribbon": {
            "width": int(ribbon_width),
            "height": int(ribbon_height),
            "band_px": int(band_px),
            "band_count": int(band_count),
            "strip_fps": float(strip_fps),
            "effective_strip_fps": _json_float(effective_strip_fps),
            "duration_s": _json_float(duration_s),
            "band_frac": float(band_frac),
            "band_center_frac": float(band_center_frac),
            "source_band_y0": int(source_band_y0),
            "source_band_y1": int(source_band_y1),
            "distance_source": str(spacing.get("distance_source", "uniform")),
            "pixels_per_meter": _json_float(spacing.get("pixels_per_meter")),
            "median_distance_m": _json_float(spacing.get("median_distance_m")),
            "median_dt_s": _json_float(spacing.get("median_dt_s")),
            "total_distance_m": _json_float(total_distance),
            "perspective_caveat": "Continuous push-broom source-video strip; perspective remains until calibrated top-down rectification.",
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


def _run_path(run_dir: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


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
