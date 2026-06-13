from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    input_dir = Path("/tmp/tarmac_runway_test")
    out_dir = Path("/tmp/crackseg_smoke")
    analyze_dir = Path("/tmp/crackseg_analyze_smoke")
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing smoke input directory: {input_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(analyze_dir, ignore_errors=True)

    subprocess.run(
        [
            "uv",
            "run",
            "tarmac",
            "crack-measure",
            str(input_dir),
            "--out",
            str(out_dir),
        ],
        check=True,
    )
    measurements = pd.read_csv(out_dir / "crack_measurements.csv")
    if measurements.empty:
        raise AssertionError("crack-measure produced no measurement rows.")
    missing = [p for p in measurements["overlay_path"] if not Path(p).exists()]
    if missing:
        raise AssertionError(f"Missing crack overlays: {missing[:3]}")

    names = measurements["filename"].str.lower()
    non_mask = names.str.contains("non|normal|clean|no_crack|nocrack|negative")
    if non_mask.any() and (~non_mask).any():
        cracked_mean = float(measurements.loc[~non_mask, "crack_area_pct"].mean())
        non_mean = float(measurements.loc[non_mask, "crack_area_pct"].mean())
        print(f"cracked_mean_area_pct={cracked_mean:.6f}")
        print(f"non_cracked_mean_area_pct={non_mean:.6f}")
        if cracked_mean <= non_mean:
            print(
                "group_mean_check=flagged "
                "(filename-noncracked fixtures measured as visibly cracked; inspect overlays)"
            )
        else:
            print("group_mean_check=ok")
    else:
        print("group_mean_check=skipped (could not infer cracked/non-cracked groups from filenames)")

    first_image = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)[0]
    subprocess.run(
        [
            "uv",
            "run",
            "tarmac",
            "analyze",
            str(first_image),
            "--region",
            "full",
            "--out",
            str(analyze_dir),
        ],
        check=True,
    )
    summary = pd.read_json(analyze_dir / "summary.json", typ="series")
    if summary["region"] != "full":
        raise AssertionError(f"analyze --region full selected {summary['region']!r}")
    print("analyze_region_full=ok")


if __name__ == "__main__":
    main()
