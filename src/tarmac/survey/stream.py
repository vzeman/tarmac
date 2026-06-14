from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from tarmac.survey.telemetry import video_duration


@dataclass(frozen=True)
class FrameSample:
    index: int
    timestamp_s: float
    frame_path: Path


def timestamp_sequence(duration_seconds: float, fps: float, clip_seconds: float | None = None) -> list[float]:
    if fps <= 0:
        raise ValueError("--fps must be positive.")
    effective_duration = min(duration_seconds, clip_seconds) if clip_seconds is not None else duration_seconds
    effective_duration = max(0.0, float(effective_duration))
    if effective_duration == 0.0:
        return [0.0]
    step = 1.0 / float(fps)
    return [float(t) for t in np.arange(0.0, effective_duration, step)]


def stream_sampled_frames(
    video_path: Path,
    *,
    out_dir: Path,
    fps: float,
    clip_seconds: float | None = None,
    jpeg_quality: int = 2,
) -> Iterator[FrameSample]:
    """Yield seek-extracted frames without decoding the full video."""
    video_path = video_path.expanduser().resolve()
    duration = video_duration(video_path)
    timestamps = timestamp_sequence(duration, fps=fps, clip_seconds=clip_seconds)
    temp_dir = out_dir / "_tmp_frames"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    for index, timestamp_s in enumerate(timestamps):
        frame_path = temp_dir / f"sample_{index:06d}_t{timestamp_s:010.3f}.jpg"
        extract_frame(video_path, timestamp_s=timestamp_s, output_path=frame_path, jpeg_quality=jpeg_quality)
        yield FrameSample(index=index, timestamp_s=timestamp_s, frame_path=frame_path)


def extract_frame(video_path: Path, *, timestamp_s: float, output_path: Path, jpeg_quality: int = 2) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(timestamp_s):.3f}",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        "-q:v",
        str(int(jpeg_quality)),
        "-y",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
