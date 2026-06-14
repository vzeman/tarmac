from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/survey_clip60")
    required = [
        "telemetry.parquet",
        "track.geojson",
        "samples.parquet",
        "problems.parquet",
        "summary.json",
        "map.html",
        "problems_table.html",
        "index.html",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise AssertionError(f"Missing survey outputs in {run_dir}: {missing}")

    telemetry = pd.read_parquet(run_dir / "telemetry.parquet")
    samples = pd.read_parquet(run_dir / "samples.parquet")
    problems = pd.read_parquet(run_dir / "problems.parquet")
    if telemetry.empty:
        raise AssertionError("telemetry.parquet is empty")
    if samples.empty:
        raise AssertionError("samples.parquet is empty")
    for col in ["t", "lat", "lon", "speed_kmh", "quality_grade", "surface_type", "issues"]:
        if col not in samples.columns:
            raise AssertionError(f"samples.parquet missing column {col}")
    if "problem_image" not in problems.columns:
        raise AssertionError("problems.parquet missing problem_image column")

    geojson = json.loads((run_dir / "track.geojson").read_text(encoding="utf-8"))
    if geojson.get("type") != "FeatureCollection" or not geojson.get("features"):
        raise AssertionError("track.geojson is not a non-empty FeatureCollection")

    map_html = (run_dir / "map.html").read_text(encoding="utf-8")
    table_html = (run_dir / "problems_table.html").read_text(encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    route_notice = str(summary.get("route_notice") or "")
    expected_candidates = [
        route_notice,
        route_notice.replace(" - ", " — "),
        str(summary.get("gps_source", {}).get("type") or ""),
        "Route is IMU-estimated",
        "GPS",
    ]
    if not any(candidate and candidate in map_html for candidate in expected_candidates) or "L.map" not in map_html:
        raise AssertionError("map.html missing Leaflet map or GPS notice")
    if not any(candidate and candidate in table_html for candidate in expected_candidates) or "problem-table" not in table_html:
        raise AssertionError("problems_table.html missing sortable table or GPS notice")
    print(
        f"survey smoke ok: samples={len(samples)} problems={len(problems)} "
        f"telemetry_rows={len(telemetry)} run_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
