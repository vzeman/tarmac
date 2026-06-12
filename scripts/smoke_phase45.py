from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_IMG_DIR = Path("/tmp/tarmac_test_imgs")
TEST_VIDEO_DIR = Path("/tmp/tarmac_video_frames")
TEST_VIDEO = Path("/tmp/tarmac_test.mp4")
DIR_RUN = Path("/tmp/tarmac_test_imgs_run")
VIDEO_RUN = Path("/tmp/tarmac_test_video_run")
PORT = 8765


def main() -> None:
    manifest = pd.read_parquet(ROOT / "data/processed/manifest.parquet")
    selected = select_test_images(manifest)
    prepare_images(selected)
    run(["tarmac", "analyze", str(TEST_IMG_DIR), "--out", str(DIR_RUN), "--batch-size", "16"])
    pred_table = predicted_vs_true(selected, DIR_RUN / "results.parquet")
    assert len(pred_table) == len(selected)
    assert pred_table["predicted_quality"].notna().all()
    print(pred_table.to_string(index=False))

    make_video(manifest)
    run(["tarmac", "analyze", str(TEST_VIDEO), "--out", str(VIDEO_RUN), "--fps", "2", "--batch-size", "16"])
    run(["tarmac", "report", str(VIDEO_RUN)])
    report_path = VIDEO_RUN / "report.html"
    assert report_path.exists(), report_path
    assert report_path.stat().st_size > 100_000, report_path.stat().st_size
    assert "plotly" in report_path.read_text().lower()

    check_streamlit()
    print(json.dumps({
        "image_run": str(DIR_RUN),
        "video": str(TEST_VIDEO),
        "video_run": str(VIDEO_RUN),
        "report": str(report_path),
    }, indent=2))


def select_test_images(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quality in sorted(manifest["quality"].unique()):
        group = manifest[manifest["quality"] == quality].sort_values(["surface_type", "image_path"])
        rows.extend(group.groupby("surface_type", sort=True).head(1).head(2).to_dict("records"))
    extras = (
        manifest.sort_values(["surface_type", "quality", "image_path"])
        .drop_duplicates("surface_type")
        .head(max(0, 12 - len(rows)))
        .to_dict("records")
    )
    rows.extend(extras)
    selected = pd.DataFrame(rows).drop_duplicates("image_path").head(12).reset_index(drop=True)
    if len(selected) < 12:
        remainder = manifest[~manifest["image_path"].isin(selected["image_path"])].head(12 - len(selected))
        selected = pd.concat([selected, remainder], ignore_index=True)
    return selected


def prepare_images(selected: pd.DataFrame) -> None:
    shutil.rmtree(TEST_IMG_DIR, ignore_errors=True)
    TEST_IMG_DIR.mkdir(parents=True)
    for row in selected.itertuples():
        src = ROOT / row.image_path
        dst = TEST_IMG_DIR / src.name
        shutil.copy2(src, dst)


def predicted_vs_true(selected: pd.DataFrame, results_path: Path) -> pd.DataFrame:
    results = pd.read_parquet(results_path)
    truth = selected.copy()
    truth["filename"] = truth["image_path"].map(lambda p: Path(p).name)
    table = truth.merge(
        results[["filename", "predicted_quality", "surface_type", "confidence"]],
        on="filename",
        how="left",
        suffixes=("_true", "_pred"),
    )
    return table[
        [
            "filename",
            "quality",
            "predicted_quality",
            "surface_type_true",
            "surface_type_pred",
            "confidence",
        ]
    ].rename(
        columns={
            "quality": "true_quality",
            "surface_type_true": "true_surface",
            "surface_type_pred": "pred_surface",
        }
    )


def make_video(manifest: pd.DataFrame) -> None:
    shutil.rmtree(TEST_VIDEO_DIR, ignore_errors=True)
    TEST_VIDEO_DIR.mkdir(parents=True)
    if TEST_VIDEO.exists():
        TEST_VIDEO.unlink()
    sample = manifest.sort_values(["quality", "surface_type", "image_path"]).head(30)
    for idx, row in enumerate(sample.itertuples()):
        shutil.copy2(ROOT / row.image_path, TEST_VIDEO_DIR / f"img_{idx:03d}.jpg")
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "2",
        "-i",
        str(TEST_VIDEO_DIR / "img_%03d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(TEST_VIDEO),
    ], use_uv=False)
    assert TEST_VIDEO.exists()


def check_streamlit() -> None:
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "streamlit",
            "run",
            "src/tarmac/ui/app.py",
            "--server.headless",
            "true",
            "--server.port",
            str(PORT),
        ],
        cwd=ROOT,
        env={**os.environ, "UV_CACHE_DIR": ".uv-cache"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 30
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}", timeout=2) as response:
                    assert response.status == 200
                    return
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        output = ""
        if proc.stdout is not None:
            output = proc.stdout.read()
        raise RuntimeError(f"Streamlit did not serve HTTP 200 within 30s: {last_error}\n{output}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def run(cmd: list[str], use_uv: bool = True) -> None:
    full_cmd = ["uv", "run", *cmd] if use_uv else cmd
    subprocess.run(full_cmd, cwd=ROOT, env={**os.environ, "UV_CACHE_DIR": ".uv-cache"}, check=True)


if __name__ == "__main__":
    main()
