from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tarmac.survey.telemetry import ROUTE_NOTICE, StartLocation, start_location


class GpsSourceType(str, Enum):
    EMBEDDED_VIDEO = "embedded_video"
    SIDECAR = "sidecar"
    IMU_DEADRECKON = "imu_deadreckon"
    NONE = "none"


@dataclass
class GpsSource:
    source_type: GpsSourceType
    reason: str
    parser: str | None = None
    path: Path | None = None
    track: pd.DataFrame | None = None
    start: StartLocation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.source_type.value,
            "reason": self.reason,
            "parser": self.parser,
            "path": str(self.path) if self.path is not None else None,
            "sample_count": int(len(self.track)) if self.track is not None else 0,
            "start_location": self.start.as_dict() if self.start is not None else None,
        }
        payload.update(self.metadata)
        return payload


GPS_SOURCE_CHOICES = {"auto", "embedded", "sidecar", "imu", "none"}


def detect_gps_source(
    video_path: Path,
    sidecar: Path | None = None,
    *,
    source_hint: str = "auto",
) -> GpsSource:
    """Detect and parse the best available GPS source for a survey video."""
    video_path = video_path.expanduser().resolve()
    hint = str(source_hint or "auto").strip().lower()
    if hint not in GPS_SOURCE_CHOICES:
        raise ValueError(f"--gps-source must be one of {sorted(GPS_SOURCE_CHOICES)}, got {source_hint!r}")

    if hint == "none":
        return _none_source(video_path, "GPS source forced to none.")

    explicit_sidecar = sidecar.expanduser().resolve() if sidecar is not None else None
    if explicit_sidecar is not None:
        source = _parse_explicit_sidecar(explicit_sidecar)
        if hint in {"auto", "sidecar"} or source.source_type == GpsSourceType.SIDECAR:
            return source
        if hint == "embedded" and source.source_type == GpsSourceType.EMBEDDED_VIDEO:
            return source
        raise ValueError(f"Explicit --gps-sidecar {explicit_sidecar} is not compatible with --gps-source {hint}.")

    if hint in {"auto", "sidecar"}:
        source = _detect_sidecar(video_path)
        if source is not None:
            return source
        if hint == "sidecar":
            raise FileNotFoundError(f"No .track.json or .gpx sidecar found next to {video_path}")

    if hint in {"auto", "embedded"}:
        source = _detect_embedded(video_path)
        if source is not None:
            return source
        if hint == "embedded":
            raise RuntimeError(f"No embedded timed GPS track found in or next to {video_path}")

    if hint in {"auto", "imu"}:
        source = _detect_imu(video_path)
        if source is not None:
            return source
        if hint == "imu":
            raise RuntimeError(f"No usable IMU dead-reckoning source with a start GPS point found in {video_path}")

    return _none_source(video_path, "No sidecar, embedded timed GPS, or usable IMU start point was found.")


def parse_track_json(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", payload if isinstance(payload, list) else [])
    if not isinstance(frames, list):
        raise ValueError(f"track.json frames must be a list: {path}")
    rows: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        lat = _float_or_none(frame.get("lat", frame.get("latitude")))
        lon = _float_or_none(frame.get("lon", frame.get("lng", frame.get("longitude"))))
        if lat is None or lon is None:
            continue
        rows.append(
            {
                "utc_ms": _float_or_none(frame.get("utc_ms", frame.get("timestamp_ms"))),
                "t": _float_or_none(frame.get("t", frame.get("t_s", frame.get("time_s")))),
                "lat": lat,
                "lon": lon,
                "alt_m": _float_or_none(frame.get("alt", frame.get("alt_m", frame.get("altitude")))),
                "speed_mps": _speed_mps(frame),
                "heading_deg": _float_or_none(frame.get("heading", frame.get("heading_deg", frame.get("course")))),
            }
        )
    return _finalize_track(rows, source="track_json_sidecar", approximate=False)


def parse_gpx(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    root = ET.parse(path).getroot()
    rows: list[dict[str, Any]] = []
    for point in root.iter():
        if _strip_ns(point.tag) not in {"trkpt", "rtept", "wpt"}:
            continue
        lat = _float_or_none(point.attrib.get("lat"))
        lon = _float_or_none(point.attrib.get("lon"))
        if lat is None or lon is None:
            continue
        row: dict[str, Any] = {"lat": lat, "lon": lon, "alt_m": None, "utc_ms": None}
        for child in point.iter():
            tag = _strip_ns(child.tag).lower()
            text = (child.text or "").strip()
            if tag == "ele":
                row["alt_m"] = _float_or_none(text)
            elif tag == "time":
                parsed = _parse_datetime_ms(text)
                if parsed is not None:
                    row["utc_ms"] = parsed
            elif tag in {"speed", "gpxtpx:speed"}:
                row["speed_mps"] = _float_or_none(text)
            elif tag in {"course", "heading", "bearing"}:
                row["heading_deg"] = _float_or_none(text)
        rows.append(row)
    return _finalize_track(rows, source="gpx_sidecar", approximate=False)


def parse_dji_srt(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_line = next((line for line in lines if "-->" in line), "")
        if not time_line:
            continue
        start_text = time_line.split("-->", 1)[0].strip()
        t = _parse_srt_time(start_text)
        body = " ".join(line for line in lines if "-->" not in line and not line.isdigit())
        lat, lon, alt = _parse_srt_location(body)
        if lat is None or lon is None:
            continue
        utc_ms = _parse_datetime_ms(body)
        rows.append(
            {
                "t": t,
                "utc_ms": utc_ms,
                "lat": lat,
                "lon": lon,
                "alt_m": alt,
                "speed_mps": _extract_labeled_float(body, ["speed", "gps_speed", "vel"]),
                "heading_deg": _extract_labeled_float(body, ["heading", "course", "yaw"]),
            }
        )
    return _finalize_track(rows, source="dji_srt", approximate=False)


def parse_gpmf(video_path: Path) -> pd.DataFrame:
    if not _has_gpmf_stream(video_path) or shutil.which("exiftool") is None:
        return _empty_track()
    return _parse_exiftool_gps(video_path, source="gopro_gpmf_exiftool", prefer_gpmf=True)


def parse_generic_embedded_gps(video_path: Path) -> pd.DataFrame:
    return _parse_exiftool_gps(video_path, source="generic_embedded_gps_exiftool", prefer_gpmf=False)


def interpolate_track(track: pd.DataFrame, timestamp_s: float) -> dict[str, float | str | bool]:
    if track.empty:
        raise ValueError("Telemetry track is empty.")
    times = track["t"].astype(float).to_numpy()
    result: dict[str, float | str | bool] = {"t": float(timestamp_s)}
    for col in ["lat", "lon", "alt_m", "speed_mps", "speed_kmh"]:
        result[col] = _interp_numeric(track, times, col, float(timestamp_s))
    result["heading_deg"] = _interp_heading(track, times, float(timestamp_s))
    result["telemetry_source"] = str(track["telemetry_source"].iloc[0])
    result["route_approximate"] = bool(track["route_approximate"].iloc[0])
    result["route_notice"] = str(track["notice"].iloc[0]) if "notice" in track.columns else ""
    return result


def resample_track(track: pd.DataFrame, timestamps_s: Iterable[float]) -> pd.DataFrame:
    rows = [interpolate_track(track, float(timestamp_s)) for timestamp_s in timestamps_s]
    return pd.DataFrame(rows)


def no_geo_track(duration_seconds: float, *, start: StartLocation | None = None, reason: str) -> pd.DataFrame:
    duration_seconds = max(0.0, float(duration_seconds))
    t = np.arange(0.0, duration_seconds + 1e-6, 1.0, dtype=float)
    if len(t) == 0:
        t = np.array([0.0], dtype=float)
    if t[-1] < duration_seconds:
        t = np.append(t, duration_seconds)
    lat = np.full(len(t), np.nan, dtype=float)
    lon = np.full(len(t), np.nan, dtype=float)
    return pd.DataFrame(
        {
            "t": t,
            "lat": lat,
            "lon": lon,
            "alt_m": np.full(len(t), start.alt_m if start is not None and start.alt_m is not None else np.nan),
            "speed_mps": np.zeros(len(t), dtype=float),
            "speed_kmh": np.zeros(len(t), dtype=float),
            "heading_deg": np.full(len(t), np.nan, dtype=float),
            "telemetry_source": "no_gps",
            "route_approximate": False,
            "warning": reason,
            "notice": "No timed GPS track was found; map route is omitted.",
        }
    )


def gps_source_status(source: GpsSource, *, telemetry_parse: dict[str, Any] | None = None) -> dict[str, Any]:
    if source.source_type == GpsSourceType.IMU_DEADRECKON and telemetry_parse is not None:
        return telemetry_parse
    if source.source_type in {GpsSourceType.SIDECAR, GpsSourceType.EMBEDDED_VIDEO}:
        return {
            "parsed": True,
            "plausible": True,
            "status": source.parser or source.source_type.value,
            "warning": None,
            "sample_count": int(len(source.track)) if source.track is not None else 0,
        }
    return {
        "parsed": False,
        "plausible": False,
        "status": "no_gps",
        "warning": source.reason,
        "sample_count": 0,
    }


def route_notice_for_source(source: GpsSource) -> str:
    if source.source_type == GpsSourceType.IMU_DEADRECKON:
        return ROUTE_NOTICE
    if source.source_type == GpsSourceType.NONE:
        if source.start is not None:
            return "No timed GPS track was found; map shows only the single start point."
        return "No GPS track was found; map route is omitted."
    if source.source_type == GpsSourceType.SIDECAR:
        return f"GPS route from sidecar ({source.parser or 'sidecar'})."
    return f"GPS route from embedded timed metadata ({source.parser or 'embedded'})."


def start_from_track(track: pd.DataFrame, *, source: str) -> StartLocation | None:
    if track.empty:
        return None
    valid = track[pd.notna(track["lat"]) & pd.notna(track["lon"])]
    if valid.empty:
        return None
    first = valid.iloc[0]
    alt = _float_or_none(first.get("alt_m"))
    return StartLocation(lat=float(first["lat"]), lon=float(first["lon"]), alt_m=alt, source=source)


def _parse_explicit_sidecar(path: Path) -> GpsSource:
    if not path.exists():
        raise FileNotFoundError(f"GPS sidecar does not exist: {path}")
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".track", ".json"] or path.name.lower().endswith(".track.json"):
        track = parse_track_json(path)
        _require_track(track, path)
        return _track_source(GpsSourceType.SIDECAR, track, path, "track_json", "Explicit --gps-sidecar track.json.")
    if path.suffix.lower() == ".gpx":
        track = parse_gpx(path)
        _require_track(track, path)
        return _track_source(GpsSourceType.SIDECAR, track, path, "gpx", "Explicit --gps-sidecar GPX track.")
    if path.suffix.lower() == ".srt":
        track = parse_dji_srt(path)
        _require_track(track, path)
        return _track_source(GpsSourceType.EMBEDDED_VIDEO, track, path, "dji_srt", "Explicit DJI SRT timed GPS sidecar.")
    raise ValueError(f"Unsupported GPS sidecar type: {path}")


def _detect_sidecar(video_path: Path) -> GpsSource | None:
    for path, parser in _sidecar_candidates(video_path):
        if not path.exists():
            continue
        track = parse_track_json(path) if parser == "track_json" else parse_gpx(path)
        if len(track) > 0:
            reason = f"Found same-basename {path.name} GPS sidecar."
            return _track_source(GpsSourceType.SIDECAR, track, path, parser, reason)
    return None


def _detect_embedded(video_path: Path) -> GpsSource | None:
    for srt_path in _srt_candidates(video_path):
        if not srt_path.exists():
            continue
        track = parse_dji_srt(srt_path)
        if len(track) > 1:
            return _track_source(
                GpsSourceType.EMBEDDED_VIDEO,
                track,
                srt_path,
                "dji_srt",
                f"Found same-basename DJI SRT timed GPS sidecar: {srt_path.name}.",
            )

    embedded_srt = _extract_embedded_srt_track(video_path)
    if embedded_srt is not None and len(embedded_srt[1]) > 1:
        srt_path, track = embedded_srt
        return _track_source(
            GpsSourceType.EMBEDDED_VIDEO,
            track,
            srt_path,
            "embedded_dji_srt",
            "Extracted an embedded subtitle GPS stream with ffmpeg.",
        )

    gpmf_track = parse_gpmf(video_path)
    if len(gpmf_track) > 1:
        return _track_source(
            GpsSourceType.EMBEDDED_VIDEO,
            gpmf_track,
            video_path,
            "gopro_gpmf",
            "ExifTool found multiple GoPro GPMF GPS samples.",
        )

    generic_track = parse_generic_embedded_gps(video_path)
    if len(generic_track) > 1:
        return _track_source(
            GpsSourceType.EMBEDDED_VIDEO,
            generic_track,
            video_path,
            "generic_embedded_gps",
            "ExifTool found multiple embedded timed GPS samples.",
        )
    return None


def _detect_imu(video_path: Path) -> GpsSource | None:
    if not _has_imu_stream(video_path):
        return None
    try:
        start = start_location(video_path)
    except RuntimeError:
        return None
    return GpsSource(
        source_type=GpsSourceType.IMU_DEADRECKON,
        reason="No timed GPS track found; Apple-style IMU metadata and a single start GPS point are available.",
        parser="apple_live_photo_info",
        path=video_path,
        start=start,
    )


def _none_source(video_path: Path, reason: str) -> GpsSource:
    try:
        start = start_location(video_path)
    except RuntimeError:
        start = None
    if start is not None:
        reason = f"{reason} A single start GPS point is available but no timed route was found."
    return GpsSource(source_type=GpsSourceType.NONE, reason=reason, parser="none", path=video_path, start=start)


def _track_source(
    source_type: GpsSourceType,
    track: pd.DataFrame,
    path: Path,
    parser: str,
    reason: str,
) -> GpsSource:
    start = start_from_track(track, source=parser)
    return GpsSource(source_type=source_type, reason=reason, parser=parser, path=path, track=track, start=start)


def _sidecar_candidates(video_path: Path) -> list[tuple[Path, str]]:
    return [
        (video_path.with_suffix(".track.json"), "track_json"),
        (video_path.with_suffix(".gpx"), "gpx"),
        (video_path.with_suffix(".GPX"), "gpx"),
    ]


def _srt_candidates(video_path: Path) -> list[Path]:
    return [video_path.with_suffix(".srt"), video_path.with_suffix(".SRT")]


def _extract_embedded_srt_track(video_path: Path) -> tuple[Path, pd.DataFrame] | None:
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory(prefix="tarmac_srt_") as tmp:
        out = Path(tmp) / f"{video_path.stem}.srt"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:s:0",
            "-c:s",
            "srt",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        if not out.exists() or out.stat().st_size == 0:
            return None
        track = parse_dji_srt(out)
        return video_path, track


def _parse_exiftool_gps(video_path: Path, *, source: str, prefer_gpmf: bool) -> pd.DataFrame:
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        return _empty_track()
    cmd = [
        exiftool,
        "-api",
        "LargeFileSupport=1",
        "-ee",
        "-n",
        "-j",
        "-a",
        "-G1",
        str(video_path),
    ]
    try:
        payload = json.loads(subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=60))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return _empty_track()
    if not payload:
        return _empty_track()
    row = payload[0]
    lat_values = _metadata_values(row, ["GPSLatitude"], prefer_gpmf=prefer_gpmf)
    lon_values = _metadata_values(row, ["GPSLongitude"], prefer_gpmf=prefer_gpmf)
    if not lat_values or not lon_values:
        coord_values = _metadata_values(row, ["GPSCoordinates", "GPSPosition"], prefer_gpmf=prefer_gpmf)
        parsed = [_parse_gps_tuple(value) for value in coord_values]
        lat_values = [item[0] for item in parsed if item is not None]
        lon_values = [item[1] for item in parsed if item is not None]
        alt_values = [item[2] for item in parsed if item is not None]
    else:
        alt_values = _metadata_values(
            row,
            ["GPSAltitude", "AbsoluteAltitude", "RelativeAltitude", "Altitude"],
            prefer_gpmf=prefer_gpmf,
        )

    count = min(len(lat_values), len(lon_values))
    if count == 0:
        return _empty_track()
    speed_values = _metadata_values(row, ["GPSSpeed", "GPSHSpeed", "Speed"], prefer_gpmf=prefer_gpmf)
    heading_values = _metadata_values(
        row,
        ["GPSImgDirection", "GPSDestBearing", "GPSHeading", "Heading", "Course"],
        prefer_gpmf=prefer_gpmf,
    )
    time_values = _metadata_values(
        row,
        ["SampleTime", "GPSDateTime", "GPSDate/Time", "DateTimeOriginal", "CreateDate"],
        prefer_gpmf=prefer_gpmf,
    )
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        lat = _float_or_none(lat_values[idx])
        lon = _float_or_none(lon_values[idx])
        if lat is None or lon is None:
            continue
        time_value = time_values[idx] if idx < len(time_values) else None
        parsed_time = _parse_exif_time(time_value)
        rows.append(
            {
                "t": parsed_time[0],
                "utc_ms": parsed_time[1],
                "lat": lat,
                "lon": lon,
                "alt_m": _float_or_none(alt_values[idx]) if idx < len(alt_values) else None,
                "speed_mps": _float_or_none(speed_values[idx]) if idx < len(speed_values) else None,
                "heading_deg": _float_or_none(heading_values[idx]) if idx < len(heading_values) else None,
            }
        )
    return _finalize_track(rows, source=source, approximate=False)


def _metadata_values(row: dict[str, Any], names: list[str], *, prefer_gpmf: bool) -> list[Any]:
    normalized_names = {_normalize_key(name) for name in names}
    matches: list[tuple[int, list[Any]]] = []
    for key, value in row.items():
        group, tag = _split_exif_key(str(key))
        tag_norm = _normalize_key(tag)
        if tag_norm not in normalized_names and not any(tag_norm.startswith(name) for name in normalized_names):
            continue
        priority = 0
        group_norm = group.lower()
        if prefer_gpmf:
            priority = 0 if any(token in group_norm for token in ["gpmf", "gopro"]) else 1
        matches.append((priority, _as_list(value)))
    values: list[Any] = []
    for _priority, items in sorted(matches, key=lambda item: item[0]):
        values.extend(items)
    return values


def _finalize_track(rows: list[dict[str, Any]], *, source: str, approximate: bool) -> pd.DataFrame:
    if not rows:
        return _empty_track()
    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df = df[pd.notna(df["lat"]) & pd.notna(df["lon"])].copy()
    if df.empty:
        return _empty_track()

    if "utc_ms" in df and df["utc_ms"].notna().any():
        df["utc_ms"] = pd.to_numeric(df["utc_ms"], errors="coerce")
        first_utc = float(df["utc_ms"].dropna().iloc[0])
        df["t"] = pd.to_numeric(df.get("t"), errors="coerce")
        missing_t = df["t"].isna()
        df.loc[missing_t, "t"] = (df.loc[missing_t, "utc_ms"] - first_utc) / 1000.0
    else:
        df["utc_ms"] = np.nan
        df["t"] = pd.to_numeric(df.get("t"), errors="coerce")
    if df["t"].isna().all():
        df["t"] = np.arange(len(df), dtype=float)
    elif df["t"].isna().any():
        known = df["t"].dropna()
        df["t"] = df["t"].interpolate(limit_direction="both")
        if df["t"].isna().any():
            start = float(known.iloc[0]) if len(known) else 0.0
            df["t"] = np.arange(len(df), dtype=float) + start
    if len(df) > 1:
        times = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
        finite = times[np.isfinite(times)]
        if len(finite) == 0 or float(np.nanmax(finite) - np.nanmin(finite)) <= 1e-9:
            df["t"] = np.arange(len(df), dtype=float)
        elif len(np.unique(np.round(finite, 6))) < len(finite):
            df["t"] = np.maximum.accumulate(times + np.arange(len(times), dtype=float) * 1e-6)

    for col in ["alt_m", "speed_mps", "heading_deg"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.sort_values("t").drop_duplicates(subset=["t", "lat", "lon"]).reset_index(drop=True)
    df["t"] = df["t"].astype(float)
    if len(df) > 1:
        if df["speed_mps"].isna().all():
            df["speed_mps"] = _compute_speeds(df)
        else:
            df["speed_mps"] = df["speed_mps"].interpolate(limit_direction="both").fillna(0.0)
        if df["heading_deg"].isna().all():
            df["heading_deg"] = _compute_headings(df)
        else:
            df["heading_deg"] = df["heading_deg"].interpolate(limit_direction="both") % 360.0
    else:
        df["speed_mps"] = df["speed_mps"].fillna(0.0)
        df["heading_deg"] = df["heading_deg"].fillna(np.nan)
    df["speed_kmh"] = df["speed_mps"].astype(float) * 3.6
    df["telemetry_source"] = source
    df["route_approximate"] = bool(approximate)
    df["warning"] = None
    df["notice"] = "GPS route from measured timed GPS samples."
    columns = [
        "t",
        "utc_ms",
        "lat",
        "lon",
        "alt_m",
        "speed_mps",
        "speed_kmh",
        "heading_deg",
        "telemetry_source",
        "route_approximate",
        "warning",
        "notice",
    ]
    return df[columns]


def _compute_speeds(df: pd.DataFrame) -> np.ndarray:
    speed = np.zeros(len(df), dtype=float)
    for idx in range(1, len(df)):
        dt = float(df.loc[idx, "t"] - df.loc[idx - 1, "t"])
        if dt <= 0:
            continue
        speed[idx] = _haversine_m(
            float(df.loc[idx - 1, "lat"]),
            float(df.loc[idx - 1, "lon"]),
            float(df.loc[idx, "lat"]),
            float(df.loc[idx, "lon"]),
        ) / dt
    if len(speed) > 1:
        speed[0] = speed[1]
    return speed


def _compute_headings(df: pd.DataFrame) -> np.ndarray:
    heading = np.full(len(df), np.nan, dtype=float)
    for idx in range(1, len(df)):
        heading[idx] = _bearing_deg(
            float(df.loc[idx - 1, "lat"]),
            float(df.loc[idx - 1, "lon"]),
            float(df.loc[idx, "lat"]),
            float(df.loc[idx, "lon"]),
        )
    if len(heading) > 1:
        heading[0] = heading[1]
    return heading


def _interp_numeric(track: pd.DataFrame, times: np.ndarray, col: str, timestamp_s: float) -> float:
    if col not in track.columns:
        return float("nan")
    values = pd.to_numeric(track[col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    if not mask.any():
        return float("nan")
    if mask.sum() == 1:
        return float(values[mask][0])
    return float(np.interp(timestamp_s, times[mask], values[mask]))


def _interp_heading(track: pd.DataFrame, times: np.ndarray, timestamp_s: float) -> float:
    if "heading_deg" not in track.columns:
        return float("nan")
    values = pd.to_numeric(track["heading_deg"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    if not mask.any():
        return float("nan")
    if mask.sum() == 1:
        return float(values[mask][0] % 360.0)
    radians = np.unwrap(np.deg2rad(values[mask]))
    return float(np.rad2deg(np.interp(timestamp_s, times[mask], radians)) % 360.0)


def _require_track(track: pd.DataFrame, path: Path) -> None:
    if track.empty:
        raise ValueError(f"No GPS samples parsed from {path}")


def _empty_track() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "t",
            "utc_ms",
            "lat",
            "lon",
            "alt_m",
            "speed_mps",
            "speed_kmh",
            "heading_deg",
            "telemetry_source",
            "route_approximate",
            "warning",
            "notice",
        ]
    )


def _speed_mps(row: dict[str, Any]) -> float | None:
    for key in ["speed_mps", "speed"]:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    speed_kmh = _float_or_none(row.get("speed_kmh"))
    if speed_kmh is not None:
        return speed_kmh / 3.6
    return None


def _parse_srt_location(text: str) -> tuple[float | None, float | None, float | None]:
    lat = _extract_labeled_float(text, ["latitude", "lat"])
    lon = _extract_labeled_float(text, ["longitude", "lon", "lng"])
    alt = _extract_labeled_float(text, ["abs_alt", "altitude", "alt", "rel_alt"])
    if lat is not None and lon is not None:
        return lat, lon, alt
    gps_match = re.search(
        r"(?:gps|location|coordinates?)\s*[:=]\s*\(?\s*([-+]?\d+(?:\.\d+)?)\s*[, ]+\s*([-+]?\d+(?:\.\d+)?)(?:\s*[, ]+\s*([-+]?\d+(?:\.\d+)?))?",
        text,
        flags=re.IGNORECASE,
    )
    if gps_match:
        lat = _float_or_none(gps_match.group(1))
        lon = _float_or_none(gps_match.group(2))
        alt = _float_or_none(gps_match.group(3))
    return lat, lon, alt


def _extract_labeled_float(text: str, labels: list[str]) -> float | None:
    for label in labels:
        pattern = rf"(?:\[|\b){re.escape(label)}\s*[:=]\s*([-+]?\d+(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _float_or_none(match.group(1))
    return None


def _parse_srt_time(value: str) -> float | None:
    match = re.match(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})", value.strip())
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis = int(match.group(4).ljust(3, "0"))
    return float(hours * 3600 + minutes * 60 + seconds + millis / 1000.0)


def _parse_datetime_ms(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        value = float(text)
        if value > 10_000_000_000:
            return value
        return None
    raw = str(text)
    match = re.search(
        r"\d{4}[-:]\d{2}[-:]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
        raw,
    )
    if not match:
        return None
    value = match.group(0)
    if value[4] == ":":
        value = f"{value[0:4]}-{value[5:7]}-{value[8:]}"
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    if re.search(r"[+-]\d{4}$", value):
        value = value[:-5] + value[-5:-2] + ":" + value[-2:]
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp() * 1000.0)


def _parse_exif_time(value: Any) -> tuple[float | None, float | None]:
    utc_ms = _parse_datetime_ms(value)
    if utc_ms is not None:
        return None, utc_ms
    if value is None:
        return None, None
    text = str(value).strip()
    seconds = _float_or_none(text)
    if seconds is not None:
        return seconds, None
    match = re.match(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[,.](\d+))?", text)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        secs = int(match.group(3))
        frac = float(f"0.{match.group(4)}") if match.group(4) else 0.0
        return float(hours * 3600 + minutes * 60 + secs + frac), None
    return None, None


def _parse_gps_tuple(value: Any) -> tuple[float, float, float | None] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lat = _float_or_none(value[0])
        lon = _float_or_none(value[1])
        alt = _float_or_none(value[2]) if len(value) >= 3 else None
        return (lat, lon, alt) if lat is not None and lon is not None else None
    parts = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    if len(parts) < 2:
        return None
    return float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) >= 3 else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _split_exif_key(key: str) -> tuple[str, str]:
    if ":" in key:
        group, tag = key.rsplit(":", 1)
        return group, tag
    return "", key


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _has_gpmf_stream(video_path: Path) -> bool:
    for stream in _ffprobe_streams(video_path):
        tag = str(stream.get("codec_tag_string", "")).lower()
        codec = str(stream.get("codec_name", "")).lower()
        handler = str(stream.get("tags", {}).get("handler_name", "")).lower()
        if "gpmd" in {tag, codec} or "gpmf" in handler or "gopro" in handler:
            return True
    return False


def _has_imu_stream(video_path: Path) -> bool:
    for stream in _ffprobe_streams(video_path):
        if stream.get("codec_type") != "data" or stream.get("codec_tag_string") != "mebx":
            continue
        nb_frames = _int_or_none(stream.get("nb_frames")) or 0
        if nb_frames > 10:
            return True
    return False


def _ffprobe_streams(video_path: Path) -> list[dict[str, Any]]:
    if shutil.which("ffprobe") is None:
        return []
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-show_streams",
        "-print_format",
        "json",
        str(video_path),
    ]
    try:
        payload = json.loads(subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=30))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return []
    streams = payload.get("streams", [])
    return streams if isinstance(streams, list) else []


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_378_137.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
