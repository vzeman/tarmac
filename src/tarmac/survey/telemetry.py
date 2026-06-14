from __future__ import annotations

import json
import logging
import math
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
G_MPS2 = 9.80665
EARTH_RADIUS_M = 6_378_137.0
ROUTE_NOTICE = "Route is IMU-estimated (approximate, drifts) - no continuous GPS in source."


@dataclass(frozen=True)
class StartLocation:
    lat: float
    lon: float
    alt_m: float | None
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt_m": self.alt_m,
            "source": self.source,
        }


@dataclass
class IMUParseResult:
    samples: pd.DataFrame
    parsed: bool
    plausible: bool
    status: str
    warning: str | None
    stream_index: int | None
    packet_count: int
    sample_rate_hz: float | None
    accel_offsets: tuple[int, int, int] | None = None
    gyro_offsets: tuple[int, int, int] | None = None
    accel_magnitude_median: float | None = None
    accel_magnitude_p01: float | None = None
    accel_magnitude_p99: float | None = None
    sample_preview: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "plausible": self.plausible,
            "status": self.status,
            "warning": self.warning,
            "stream_index": self.stream_index,
            "packet_count": self.packet_count,
            "sample_rate_hz": self.sample_rate_hz,
            "accel_offsets": list(self.accel_offsets) if self.accel_offsets else None,
            "gyro_offsets": list(self.gyro_offsets) if self.gyro_offsets else None,
            "accel_magnitude_median": self.accel_magnitude_median,
            "accel_magnitude_p01": self.accel_magnitude_p01,
            "accel_magnitude_p99": self.accel_magnitude_p99,
            "sample_preview": self.sample_preview or [],
        }


def start_location(video_path: Path) -> StartLocation:
    """Read the single QuickTime start GPS point from the source video."""
    video_path = video_path.expanduser().resolve()
    exiftool = shutil.which("exiftool")
    if exiftool:
        cmd = [
            exiftool,
            "-api",
            "LargeFileSupport=1",
            "-n",
            "-j",
            "-QuickTime:GPSCoordinates",
            "-QuickTime:LocationISO6709",
            str(video_path),
        ]
        try:
            data = json.loads(subprocess.check_output(cmd, text=True))
            if data:
                row = data[0]
                gps_value = row.get("GPSCoordinates")
                parsed = _parse_gps_coordinates(gps_value)
                if parsed is not None:
                    lat, lon, alt = parsed
                    return StartLocation(lat=lat, lon=lon, alt_m=alt, source="exiftool:QuickTime:GPSCoordinates")
                iso_value = row.get("LocationISO6709")
                parsed = _parse_iso6709(iso_value)
                if parsed is not None:
                    lat, lon, alt = parsed
                    return StartLocation(lat=lat, lon=lon, alt_m=alt, source="exiftool:QuickTime:LocationISO6709")
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("ExifTool GPS read failed: %s", exc)

    probe = _ffprobe(video_path)
    tags = probe.get("format", {}).get("tags", {})
    iso_value = (
        tags.get("com.apple.quicktime.location.ISO6709")
        or tags.get("location")
        or tags.get("LocationISO6709")
    )
    parsed = _parse_iso6709(iso_value)
    if parsed is None:
        raise RuntimeError(f"Could not read QuickTime start GPS point from {video_path}")
    lat, lon, alt = parsed
    return StartLocation(lat=lat, lon=lon, alt_m=alt, source="ffprobe:format.tags.location.ISO6709")


def extract_imu(
    video_path: Path,
    *,
    work_dir: Path,
    clip_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> IMUParseResult:
    """Demux and parse the high-rate Apple Core Media metadata stream.

    The observed iPhone ProRes file stores the high-rate candidate motion stream as
    QuickTime Track 7 / ffprobe stream index 6. ExifTool names its key
    ``LivePhotoInfo`` and decodes the first 76 payload bytes with this unpack
    schema: ``VfVVf6c4lCCcclf4Vvv``. Apple does not publish the semantic names for
    these fields, so acceleration is only accepted when a candidate 3-axis float
    triplet has a gravity-like magnitude.
    """
    video_path = video_path.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        probe = _ffprobe(video_path)
        stream = _select_live_photo_stream(probe)
        if stream is None:
            return _empty_imu("No high-rate mebx LivePhotoInfo stream found.")

        raw_path = work_dir / f"imu_stream_{stream['index']}.bin"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if clip_seconds is not None:
            cmd.extend(["-t", _seconds_arg(clip_seconds)])
        cmd.extend(
            [
                "-i",
                str(video_path),
                "-map",
                f"0:{int(stream['index'])}",
                "-c",
                "copy",
                "-f",
                "data",
                str(raw_path),
            ]
        )
        subprocess.run(cmd, check=True)
        raw = raw_path.read_bytes()
        samples = _parse_live_photo_info_packets(
            raw,
            stream=stream,
            clip_seconds=clip_seconds,
            duration_seconds=duration_seconds,
        )
        try:
            raw_path.unlink()
        except OSError:
            pass
        if samples.empty:
            return _empty_imu(
                f"Metadata stream {stream['index']} demuxed but no LivePhotoInfo packets were parsed.",
                stream_index=int(stream["index"]),
            )

        enriched, validation = _promote_plausible_motion_fields(samples)
        warning = validation.get("warning")
        status = validation["status"]
        preview_cols = [
            col
            for col in [
                "t",
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "live_f0",
                "live_f1",
                "live_f2",
                "live_f3",
            ]
            if col in enriched.columns
        ]
        preview = enriched.head(5)[preview_cols].round(6).to_dict(orient="records")
        return IMUParseResult(
            samples=enriched,
            parsed=True,
            plausible=bool(validation["plausible"]),
            status=status,
            warning=warning,
            stream_index=int(stream["index"]),
            packet_count=int(len(enriched)),
            sample_rate_hz=_sample_rate(enriched),
            accel_offsets=validation.get("accel_offsets"),
            gyro_offsets=validation.get("gyro_offsets"),
            accel_magnitude_median=validation.get("accel_magnitude_median"),
            accel_magnitude_p01=validation.get("accel_magnitude_p01"),
            accel_magnitude_p99=validation.get("accel_magnitude_p99"),
            sample_preview=preview,
        )
    except (subprocess.CalledProcessError, OSError, RuntimeError, struct.error) as exc:
        return _empty_imu(f"IMU metadata parsing failed: {exc}")


def dead_reckon(
    imu: IMUParseResult,
    start: StartLocation,
    *,
    duration_seconds: float,
    nominal_speed_mps: float = 8.0,
) -> pd.DataFrame:
    """Build an approximate time-indexed route from parsed IMU or a fallback."""
    if (
        imu.parsed
        and imu.plausible
        and not imu.samples.empty
        and {"accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"}.issubset(imu.samples.columns)
    ):
        return _integrate_imu_route(imu.samples, start=start, duration_seconds=duration_seconds)
    warning = imu.warning or "No physically plausible accelerometer triplet was available."
    return _fallback_route(
        start=start,
        duration_seconds=duration_seconds,
        speed_mps=nominal_speed_mps,
        reason=warning,
    )


def video_duration(video_path: Path) -> float:
    probe = _ffprobe(video_path.expanduser().resolve())
    try:
        return float(probe["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not read video duration for {video_path}") from exc


def interpolate_track(track: pd.DataFrame, timestamp_s: float) -> dict[str, float | str | bool]:
    if track.empty:
        raise ValueError("Telemetry track is empty.")
    times = track["t"].astype(float).to_numpy()
    result: dict[str, float | str | bool] = {"t": float(timestamp_s)}
    for col in ["lat", "lon", "speed_mps", "speed_kmh", "heading_deg"]:
        result[col] = float(np.interp(float(timestamp_s), times, track[col].astype(float).to_numpy()))
    result["telemetry_source"] = str(track["telemetry_source"].iloc[0])
    result["route_approximate"] = bool(track["route_approximate"].iloc[0])
    return result


def _integrate_imu_route(samples: pd.DataFrame, *, start: StartLocation, duration_seconds: float) -> pd.DataFrame:
    df = samples.sort_values("t").copy()
    df = df[(df["t"] >= 0.0) & (df["t"] <= duration_seconds)].reset_index(drop=True)
    if len(df) < 2:
        return _fallback_route(
            start=start,
            duration_seconds=duration_seconds,
            speed_mps=8.0,
            reason="Too few plausible IMU samples to integrate.",
        )

    t = df["t"].astype(float).to_numpy()
    accel = df[["accel_x", "accel_y", "accel_z"]].astype(float).to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].astype(float).to_numpy()

    initial = accel[t <= min(3.0, max(0.1, t[-1]))]
    gravity = np.median(initial, axis=0) if len(initial) else np.median(accel[: min(len(accel), 240)], axis=0)
    gravity_norm = float(np.linalg.norm(gravity))
    if not math.isfinite(gravity_norm) or gravity_norm < 1e-6:
        gravity = np.array([0.0, 0.0, G_MPS2], dtype=float)
    else:
        gravity = gravity / gravity_norm * G_MPS2

    east = np.zeros(len(df), dtype=float)
    north = np.zeros(len(df), dtype=float)
    speed = np.zeros(len(df), dtype=float)
    heading = np.zeros(len(df), dtype=float)
    vel_e = 0.0
    vel_n = 0.0
    pos_e = 0.0
    pos_n = 0.0
    yaw = math.radians(90.0)
    linear_smooth = np.zeros(3, dtype=float)

    for i in range(1, len(df)):
        dt = float(t[i] - t[i - 1])
        if dt <= 0.0 or dt > 0.2:
            dt = 1.0 / max(_sample_rate(df) or 120.0, 1.0)
        yaw += float(gyro[i, 2]) * dt
        linear = accel[i] - gravity
        linear_smooth = 0.9 * linear_smooth + 0.1 * linear
        still = abs(float(np.linalg.norm(accel[i])) - G_MPS2) < 0.25 and float(np.linalg.norm(gyro[i])) < 0.035
        if still:
            vel_e *= 0.5
            vel_n *= 0.5
            if math.hypot(vel_e, vel_n) < 0.15:
                vel_e = 0.0
                vel_n = 0.0
        else:
            forward_acc = float(linear_smooth[1])
            lateral_acc = float(linear_smooth[0])
            vel_e += (math.cos(yaw) * forward_acc + math.cos(yaw + math.pi / 2.0) * lateral_acc) * dt
            vel_n += (math.sin(yaw) * forward_acc + math.sin(yaw + math.pi / 2.0) * lateral_acc) * dt
            vel_e *= 0.999
            vel_n *= 0.999
            current_speed = math.hypot(vel_e, vel_n)
            if current_speed > 35.0:
                scale = 35.0 / current_speed
                vel_e *= scale
                vel_n *= scale
        pos_e += vel_e * dt
        pos_n += vel_n * dt
        east[i] = pos_e
        north[i] = pos_n
        speed[i] = math.hypot(vel_e, vel_n)
        heading[i] = (math.degrees(yaw) + 360.0) % 360.0

    route = _enu_to_track(
        start=start,
        t=t,
        east_m=east,
        north_m=north,
        speed_mps=speed,
        heading_deg=heading,
        source="imu_live_photo_info_dead_reckoned",
        approximate=True,
        warning="Dead-reckoned from private Apple LivePhotoInfo fields; route drifts without GPS fixes.",
    )
    return route


def _fallback_route(
    *,
    start: StartLocation,
    duration_seconds: float,
    speed_mps: float,
    reason: str,
) -> pd.DataFrame:
    step = 1.0
    t = np.arange(0.0, max(duration_seconds, 0.0) + 1e-6, step, dtype=float)
    if len(t) == 0 or t[-1] < duration_seconds:
        t = np.append(t, duration_seconds)
    east_m = speed_mps * t
    north_m = np.zeros_like(east_m)
    speed = np.full_like(east_m, speed_mps, dtype=float)
    heading = np.full_like(east_m, 90.0, dtype=float)
    return _enu_to_track(
        start=start,
        t=t,
        east_m=east_m,
        north_m=north_m,
        speed_mps=speed,
        heading_deg=heading,
        source="fallback_nominal_straight",
        approximate=True,
        warning=reason,
    )


def _enu_to_track(
    *,
    start: StartLocation,
    t: np.ndarray,
    east_m: np.ndarray,
    north_m: np.ndarray,
    speed_mps: np.ndarray,
    heading_deg: np.ndarray,
    source: str,
    approximate: bool,
    warning: str,
) -> pd.DataFrame:
    lat0 = math.radians(start.lat)
    lat = start.lat + (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    lon = start.lon + (east_m / (EARTH_RADIUS_M * max(math.cos(lat0), 1e-9))) * (180.0 / math.pi)
    return pd.DataFrame(
        {
            "t": t.astype(float),
            "lat": lat.astype(float),
            "lon": lon.astype(float),
            "speed_mps": speed_mps.astype(float),
            "speed_kmh": (speed_mps * 3.6).astype(float),
            "heading_deg": heading_deg.astype(float),
            "telemetry_source": source,
            "route_approximate": bool(approximate),
            "warning": warning,
            "notice": ROUTE_NOTICE,
        }
    )


def _parse_live_photo_info_packets(
    raw: bytes,
    *,
    stream: dict[str, Any],
    clip_seconds: float | None,
    duration_seconds: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pos = 0
    packet_index = 0
    while pos + 8 <= len(raw):
        packet_len = struct.unpack_from(">I", raw, pos)[0]
        if packet_len < 88 or pos + packet_len > len(raw):
            break
        packet = raw[pos : pos + packet_len]
        key_id = struct.unpack_from(">I", packet, 4)[0]
        live_type = struct.unpack_from("<I", packet, 8)[0]
        if key_id == 1 and live_type == 3:
            row: dict[str, Any] = {
                "packet_index": packet_index,
                "packet_len": packet_len,
                "key_id": key_id,
                "live_type": live_type,
                "live_sample_delta_s": struct.unpack_from("<f", packet, 12)[0],
                "live_clock": struct.unpack_from("<I", packet, 16)[0],
                "live_reserved": struct.unpack_from("<I", packet, 20)[0],
            }
            for i in range(6):
                row[f"live_f{i}"] = struct.unpack_from("<f", packet, 24 + i * 4)[0]
            c0, c1, c2, c3 = struct.unpack_from("<bbbb", packet, 48)
            row.update({"live_c0": c0, "live_c1": c1, "live_c2": c2, "live_c3": c3})
            for i, off in enumerate(range(64, 80, 4)):
                row[f"live_tail_f{i}"] = struct.unpack_from("<f", packet, off)[0]
            row["live_tail_u0"] = struct.unpack_from("<I", packet, 80)[0]
            row["live_tail_h0"] = struct.unpack_from("<H", packet, 84)[0]
            row["live_tail_h1"] = struct.unpack_from("<H", packet, 86)[0]
            for i, off in enumerate(range(88, min(packet_len, 112), 4)):
                row[f"live_extra_f{i}"] = struct.unpack_from("<f", packet, off)[0]
            rows.append(row)
        pos += packet_len
        packet_index += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    stream_duration = _float_or_none(stream.get("duration"))
    effective_duration = min(
        value
        for value in [stream_duration, duration_seconds, clip_seconds]
        if value is not None and value > 0
    ) if any(value is not None and value > 0 for value in [stream_duration, duration_seconds, clip_seconds]) else None
    if effective_duration is not None and len(df) > 1:
        df["t"] = np.linspace(0.0, float(effective_duration), num=len(df), endpoint=False)
    else:
        sample_delta = float(df["live_sample_delta_s"].median())
        if not math.isfinite(sample_delta) or sample_delta <= 0.0:
            sample_delta = 1.0 / 120.0
        df["t"] = np.arange(len(df), dtype=float) * sample_delta
    return df


def _promote_plausible_motion_fields(samples: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = samples.copy()
    float_offsets: dict[int, str] = {
        24: "live_f0",
        28: "live_f1",
        32: "live_f2",
        36: "live_f3",
        40: "live_f4",
        44: "live_f5",
        64: "live_tail_f0",
        68: "live_tail_f1",
        72: "live_tail_f2",
        76: "live_tail_f3",
        88: "live_extra_f0",
        92: "live_extra_f1",
        96: "live_extra_f2",
        100: "live_extra_f3",
        104: "live_extra_f4",
        108: "live_extra_f5",
    }
    accel_candidate = _find_accel_candidate(df, float_offsets)
    gyro_candidate = _find_gyro_candidate(df, float_offsets)
    for col in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
        df[col] = np.nan

    validation: dict[str, Any] = {
        "plausible": False,
        "status": "live_photo_info_parsed_accel_not_plausible",
        "warning": (
            "Parsed Apple LivePhotoInfo packets, but no 3-axis float triplet had a "
            "gravity-like accelerometer magnitude near 9.8 m/s^2. Falling back to "
            "a nominal straight dead-reckoned route."
        ),
    }
    if gyro_candidate is not None:
        gyro_offsets, gyro_cols = gyro_candidate
        df[["gyro_x", "gyro_y", "gyro_z"]] = df[list(gyro_cols)].astype(float)
        validation["gyro_offsets"] = gyro_offsets

    if accel_candidate is None:
        return df, validation

    accel_offsets, accel_cols, scale, stats = accel_candidate
    accel_values = df[list(accel_cols)].astype(float).to_numpy() * float(scale)
    df[["accel_x", "accel_y", "accel_z"]] = accel_values
    if gyro_candidate is None:
        df[["gyro_x", "gyro_y", "gyro_z"]] = 0.0
    validation.update(
        {
            "plausible": True,
            "status": stats.get("status", "live_photo_info_parsed_plausible_accelerometer"),
            "warning": stats.get("warning"),
            "accel_offsets": accel_offsets,
            "accel_magnitude_median": stats["median"],
            "accel_magnitude_p01": stats["p01"],
            "accel_magnitude_p99": stats["p99"],
        }
    )
    return df, validation


def _find_accel_candidate(
    df: pd.DataFrame,
    float_offsets: dict[int, str],
) -> tuple[tuple[int, int, int], tuple[str, str, str], float, dict[str, float]] | None:
    ordered_offsets = sorted(float_offsets)
    candidates: list[tuple[float, tuple[int, int, int], tuple[str, str, str], float, dict[str, float]]] = []
    for offsets in zip(ordered_offsets, ordered_offsets[1:], ordered_offsets[2:], strict=False):
        if offsets[1] - offsets[0] != 4 or offsets[2] - offsets[1] != 4:
            continue
        cols = tuple(float_offsets[offset] for offset in offsets)
        values = df[list(cols)].astype(float).to_numpy()
        if not np.isfinite(values).all() or np.nanmax(np.abs(values)) > 1_000.0:
            continue
        raw_mag = np.linalg.norm(values, axis=1)
        raw_median = float(np.nanmedian(raw_mag))
        for scale in (1.0, G_MPS2):
            mag = raw_mag * scale
            p01, median, p99 = [float(x) for x in np.nanpercentile(mag, [1, 50, 99])]
            status = "live_photo_info_parsed_plausible_accelerometer"
            warning: str | None = None
            if not (7.0 <= median <= 12.8 and p01 > 0.2 and p99 < 35.0):
                window_n = min(len(mag), max(120, int(round(3.0 * (_sample_rate(df) or 120.0)))))
                window_mag = mag[:window_n]
                window_values = values[:window_n] * scale
                p01, median, p99 = [float(x) for x in np.nanpercentile(window_mag, [1, 50, 99])]
                if not (7.0 <= median <= 13.5 and p01 > 0.2 and p99 < 25.0):
                    continue
                status = "live_photo_info_parsed_initial_window_plausible"
                warning = (
                    "Apple LivePhotoInfo motion fields passed the gravity check only in the "
                    "opening near-rest window. Dead reckoning uses these private fields and is "
                    "especially approximate/drift-prone."
                )
                axis_std = np.nanstd(window_values, axis=0)
            else:
                axis_std = np.nanstd(values * scale, axis=0)
            if scale == G_MPS2 and not (0.7 <= raw_median <= 1.3):
                continue
            if float(np.nanmax(axis_std)) < 0.01:
                continue
            score = abs(median - G_MPS2) + 0.01 * (p99 - p01)
            candidates.append(
                (
                    score,
                    tuple(int(o) for o in offsets),
                    cols,
                    float(scale),
                    {"p01": p01, "median": median, "p99": p99, "status": status, "warning": warning},
                )
            )
    if not candidates:
        return None
    _, offsets, cols, scale, stats = sorted(candidates, key=lambda item: item[0])[0]
    return offsets, cols, scale, stats


def _find_gyro_candidate(
    df: pd.DataFrame,
    float_offsets: dict[int, str],
) -> tuple[tuple[int, int, int], tuple[str, str, str]] | None:
    preferred = (100, 104, 108)
    if all(offset in float_offsets for offset in preferred):
        cols = tuple(float_offsets[offset] for offset in preferred)
        values = df[list(cols)].astype(float).to_numpy()
        mag = np.linalg.norm(values, axis=1)
        if np.isfinite(values).all() and float(np.nanpercentile(mag, 99)) < 10.0:
            return preferred, cols
    ordered_offsets = sorted(float_offsets)
    candidates: list[tuple[float, tuple[int, int, int], tuple[str, str, str]]] = []
    for offsets in zip(ordered_offsets, ordered_offsets[1:], ordered_offsets[2:], strict=False):
        if offsets[1] - offsets[0] != 4 or offsets[2] - offsets[1] != 4:
            continue
        cols = tuple(float_offsets[offset] for offset in offsets)
        values = df[list(cols)].astype(float).to_numpy()
        if not np.isfinite(values).all():
            continue
        mag = np.linalg.norm(values, axis=1)
        p99 = float(np.nanpercentile(mag, 99))
        median = float(np.nanmedian(mag))
        if 0.001 <= median <= 2.0 and p99 < 10.0:
            candidates.append((abs(median - 0.05), tuple(int(o) for o in offsets), cols))
    if not candidates:
        return None
    _, offsets, cols = sorted(candidates, key=lambda item: item[0])[0]
    return offsets, cols


def _select_live_photo_stream(probe: dict[str, Any]) -> dict[str, Any] | None:
    streams = probe.get("streams", [])
    candidates = []
    for stream in streams:
        if stream.get("codec_type") != "data" or stream.get("codec_tag_string") != "mebx":
            continue
        nb_frames = _int_or_none(stream.get("nb_frames")) or 0
        if nb_frames <= 10:
            continue
        bit_rate = _int_or_none(stream.get("bit_rate")) or 0
        candidates.append((bit_rate, nb_frames, stream))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[0][2]


def _ffprobe(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        str(video_path),
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def _empty_imu(message: str, stream_index: int | None = None) -> IMUParseResult:
    return IMUParseResult(
        samples=pd.DataFrame(),
        parsed=False,
        plausible=False,
        status="fallback_nominal_straight",
        warning=message,
        stream_index=stream_index,
        packet_count=0,
        sample_rate_hz=None,
        sample_preview=[],
    )


def _sample_rate(df: pd.DataFrame) -> float | None:
    if len(df) < 2 or "t" not in df.columns:
        return None
    dt = np.diff(df["t"].astype(float).to_numpy())
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return None
    return float(1.0 / np.median(dt))


def _parse_gps_coordinates(value: Any) -> tuple[float, float, float | None] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lat = _float_or_none(value[0])
        lon = _float_or_none(value[1])
        alt = _float_or_none(value[2]) if len(value) >= 3 else None
        return (lat, lon, alt) if lat is not None and lon is not None else None
    parts = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    if len(parts) >= 2:
        lat = float(parts[0])
        lon = float(parts[1])
        alt = float(parts[2]) if len(parts) >= 3 else None
        return lat, lon, alt
    return None


def _parse_iso6709(value: Any) -> tuple[float, float, float | None] | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.match(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?$", text)
    if not match:
        return _parse_gps_coordinates(text)
    lat = float(match.group(1))
    lon = float(match.group(2))
    alt = float(match.group(3)) if match.group(3) is not None else None
    return lat, lon, alt


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


def _seconds_arg(seconds: float) -> str:
    return f"{float(seconds):.6f}"


def write_telemetry_metadata(out_dir: Path, payload: dict[str, Any]) -> Path:
    path = out_dir / "telemetry_metadata.json"
    payload = {**payload, "generated_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
