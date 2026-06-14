from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "runs" / "smoke_assess_input"
RUN_DIR = ROOT / "runs" / "smoke_assess"
SEED = 42


def main() -> None:
    random.seed(SEED)
    prepare_inputs()
    run(
        [
            "uv",
            "run",
            "tarmac",
            "assess",
            str(INPUT_DIR),
            "--out",
            str(RUN_DIR),
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--mm-per-pixel",
            "0.5",
        ]
    )
    run(["uv", "run", "tarmac", "report", str(RUN_DIR)])
    assert_outputs()


def prepare_inputs() -> None:
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(ROOT / "data" / "processed" / "manifest.parquet")
    streets = manifest[manifest["source_dataset"] == "streetsurfacevis"].sort_values("image_path")
    street_paths: list[tuple[int, Path]] = []
    for quality in (1, 3, 5):
        candidates = streets[streets["quality"] == quality]["image_path"].tolist()
        if not candidates:
            raise AssertionError(f"Missing StreetSurfaceVis quality {quality} examples")
        street_paths.append((quality, ROOT / random.Random(SEED + quality).choice(candidates)))

    airport_root = ROOT / "data" / "raw" / "crackairport"
    airport_images = sorted(airport_root.rglob("train_images/*.jpg"))
    if len(airport_images) < 2:
        raise AssertionError("Need at least two local CrackAirport images for smoke_assess.py")
    airport_paths = random.Random(SEED).sample(airport_images, 2)

    for index, (quality, source) in enumerate(street_paths):
        target = INPUT_DIR / f"{index:02d}_streetsurfacevis_q{quality}_{source.name}"
        shutil.copy2(source, target)
    for offset, source in enumerate(airport_paths, start=len(street_paths)):
        target = INPUT_DIR / f"{offset:02d}_crackairport_{source.name}"
        shutil.copy2(source, target)


def run(command: list[str]) -> None:
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = ".uv-cache"
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def assert_outputs() -> None:
    assessment_json = RUN_DIR / "assessment.json"
    assessment_parquet = RUN_DIR / "assessment.parquet"
    report_html = RUN_DIR / "report.html"
    assert assessment_json.exists(), assessment_json
    assert assessment_parquet.exists(), assessment_parquet
    assert report_html.exists(), report_html

    payload = json.loads(assessment_json.read_text())
    records = payload.get("records", [])
    assert records, "assessment.json has no records"
    required = {"overall_condition_grade", "repair_priority", "rationale", "pci_proxy_descriptor"}
    for record in records:
        missing = sorted(required.difference(record))
        assert not missing, f"{record.get('filename')} missing {missing}"
        assert record["repair_priority"] in {"none", "monitor", "plan_repair", "urgent"}
        assert isinstance(record["rationale"], str) and "PCI-like" in record["rationale"]

    frame = pd.read_parquet(assessment_parquet)
    for column in required:
        assert column in frame.columns, f"assessment.parquet missing {column}"
    assert "Condition assessment" in report_html.read_text()
    print(
        f"smoke_assess ok: frames={len(records)} "
        f"mean_condition_grade={payload['summary']['mean_condition_grade']:.2f} "
        f"report={report_html}"
    )


if __name__ == "__main__":
    main()
