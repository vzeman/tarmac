from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pandas as pd

from tarmac.survey.gps_sources import GpsSourceType, detect_gps_source, interpolate_track


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tarmac_gps_sources_") as tmp:
        root = Path(tmp)
        sidecar_video = root / "sidecar.mp4"
        srt_video = root / "drone.mp4"
        none_video = root / "none.mp4"
        for path in [sidecar_video, srt_video, none_video]:
            _make_video(path)

        _write_track_json(sidecar_video.with_suffix(".track.json"))
        _write_dji_srt(srt_video.with_suffix(".SRT"))

        sidecar_source = detect_gps_source(sidecar_video)
        assert sidecar_source.source_type == GpsSourceType.SIDECAR, sidecar_source.as_dict()
        assert sidecar_source.parser == "track_json", sidecar_source.as_dict()
        sidecar_point = interpolate_track(sidecar_source.track, 2.0)  # type: ignore[arg-type]
        assert pd.notna(sidecar_point["lat"]) and pd.notna(sidecar_point["lon"]), sidecar_point

        srt_source = detect_gps_source(srt_video)
        assert srt_source.source_type == GpsSourceType.EMBEDDED_VIDEO, srt_source.as_dict()
        assert srt_source.parser == "dji_srt", srt_source.as_dict()
        srt_point = interpolate_track(srt_source.track, 2.0)  # type: ignore[arg-type]
        assert pd.notna(srt_point["lat"]) and pd.notna(srt_point["lon"]), srt_point

        none_source = detect_gps_source(none_video)
        assert none_source.source_type == GpsSourceType.NONE, none_source.as_dict()

        restore = _patch_survey_model()
        try:
            from typer.testing import CliRunner

            from tarmac.cli import app

            runner = CliRunner()

            sidecar_result = runner.invoke(
                app,
                [
                    "survey",
                    str(sidecar_video),
                    "--out",
                    str(root / "run_sidecar"),
                    "--fps",
                    "1",
                    "--clip-seconds",
                    "3",
                    "--device",
                    "cpu",
                ],
            )
            assert sidecar_result.exit_code == 0, sidecar_result.output
            sidecar_run = json.loads((root / "run_sidecar" / "summary.json").read_text(encoding="utf-8"))
            assert sidecar_run["gps_source"]["type"] == "sidecar", sidecar_run["gps_source"]
            sidecar_samples = pd.read_parquet(root / "run_sidecar" / "samples.parquet")
            assert sidecar_samples["lat"].notna().all() and sidecar_samples["lon"].notna().all()

            srt_result = runner.invoke(
                app,
                [
                    "survey",
                    str(srt_video),
                    "--out",
                    str(root / "run_srt"),
                    "--fps",
                    "1",
                    "--clip-seconds",
                    "3",
                    "--device",
                    "cpu",
                ],
            )
            assert srt_result.exit_code == 0, srt_result.output
            srt_run = json.loads((root / "run_srt" / "summary.json").read_text(encoding="utf-8"))
            assert srt_run["gps_source"]["type"] == "embedded_video", srt_run["gps_source"]
            srt_samples = pd.read_parquet(root / "run_srt" / "samples.parquet")
            assert srt_samples["lat"].notna().all() and srt_samples["lon"].notna().all()

            none_result = runner.invoke(
                app,
                [
                    "survey",
                    str(none_video),
                    "--out",
                    str(root / "run_none"),
                    "--gps-source",
                    "none",
                    "--fps",
                    "1",
                    "--clip-seconds",
                    "2",
                    "--device",
                    "cpu",
                ],
            )
            assert none_result.exit_code == 0, none_result.output
            none_run = json.loads((root / "run_none" / "summary.json").read_text(encoding="utf-8"))
            assert none_run["gps_source"]["type"] == "none", none_run["gps_source"]
            none_samples = pd.read_parquet(root / "run_none" / "samples.parquet")
            assert none_samples["lat"].isna().all() and none_samples["lon"].isna().all()
            assert "No GPS" in (root / "run_none" / "map.html").read_text(encoding="utf-8")
        finally:
            restore()

        print("gps source smoke ok")
        print(f"TYPE 2 sidecar detected: {sidecar_source.reason}")
        print(f"TYPE 1 DJI SRT detected: {srt_source.reason}")
        print(f"TYPE 3 no GPS detected: {none_source.reason}")
    return 0


def _make_video(path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for this smoke test.")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=5",
        "-t",
        "5",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True)


def _write_track_json(path: Path) -> None:
    frames = []
    base_utc = 1_700_000_000_000
    for idx in range(5):
        frames.append(
            {
                "utc_ms": base_utc + idx * 1000,
                "lat": 48.1000 + idx * 0.0001,
                "lon": 17.1000 + idx * 0.0001,
                "speed": 6.0,
                "heading": 45.0,
            }
        )
    path.write_text(json.dumps({"session": {"id": "smoke"}, "frames": frames, "imu": []}) + "\n", encoding="utf-8")


def _write_dji_srt(path: Path) -> None:
    blocks = []
    for idx in range(5):
        start = f"00:00:0{idx},000"
        end = f"00:00:0{idx},900"
        blocks.append(
            "\n".join(
                [
                    str(idx + 1),
                    f"{start} --> {end}",
                    (
                        f"<font size=\"36\">FrameCnt: {idx + 1} "
                        f"[latitude: {48.2000 + idx * 0.0001:.7f}] "
                        f"[longitude: {17.2000 + idx * 0.0001:.7f}] "
                        f"[abs_alt: {120.0 + idx:.1f}]</font>"
                    ),
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _patch_survey_model() -> Callable[[], None]:
    import tarmac.survey.survey as survey

    original_load = survey._load_model_context
    original_analyze = survey.analyze_frames
    original_confirm = survey._confirm_crack_for_image

    def fake_load_model_context(*, out_dir: Path, batch_size: int, device: str, **_: object) -> SimpleNamespace:
        thumbs_dir = out_dir / "_tmp_thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            out_dir=out_dir,
            thumbs_dir=thumbs_dir,
            embedder=None,
            reference_df=None,
            index=None,
            centroids=None,
            non_road_threshold=0.0,
            crack_detector=None,
            defect_detector=None,
            region="lower_half",
            batch_size=batch_size,
            device=device,
            active_suffix="smoke",
            checkpoint="smoke",
        )

    def fake_analyze_frames(frame_paths: list[Path], **_: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        rows = []
        for _path in frame_paths:
            rows.append(
                {
                    "predicted_quality": 1,
                    "surface_type": "asphalt",
                    "confidence": 1.0,
                    "road_tile_count": 1,
                    "tile_count": 1,
                    "tile_details": "[]",
                    "structural_defects": "[]",
                }
            )
        return rows, []

    def fake_confirm_crack_for_image(*args: object, **kwargs: object) -> dict[str, object]:
        config = kwargs["config"]
        return survey._empty_crack_confirmation(classifier_prob=None, config=config, candidate=False)

    survey._load_model_context = fake_load_model_context
    survey.analyze_frames = fake_analyze_frames
    survey._confirm_crack_for_image = fake_confirm_crack_for_image

    def restore() -> None:
        survey._load_model_context = original_load
        survey.analyze_frames = original_analyze
        survey._confirm_crack_for_image = original_confirm

    return restore


if __name__ == "__main__":
    raise SystemExit(main())
