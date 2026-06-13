from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    manifest_path = Path("data/processed/crack_manifest.parquet")
    head_path = Path("models/crack_head.pt")
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tarmac_crack_run")
    tiles_path = run_dir / "tiles.parquet"
    report_path = run_dir / "report.html"

    assert manifest_path.exists(), f"missing {manifest_path}"
    manifest = pd.read_parquet(manifest_path)
    counts = manifest["has_crack"].value_counts().to_dict()
    assert set(counts) == {0, 1}, f"manifest is not binary/balanced enough: {counts}"
    ratio = counts[1] / max(counts[0], 1)
    assert 0.9 <= ratio <= 1.2, f"manifest class balance ratio out of range: {ratio:.3f}"

    assert head_path.exists(), f"missing {head_path}"
    assert tiles_path.exists(), f"missing {tiles_path}"
    tiles = pd.read_parquet(tiles_path)
    for column in ("tile_crack_prob", "tile_crack"):
        assert column in tiles.columns, f"missing analyze crack column {column}"
    assert tiles["tile_crack_prob"].notna().all(), "tile_crack_prob contains nulls"

    assert report_path.exists(), f"missing {report_path}"
    html = report_path.read_text()
    assert "cracked-sections" in html, "report is missing cracked-sections panel"

    print(
        "smoke_crack ok: "
        f"manifest_rows={len(manifest)} pos={counts[1]} neg={counts[0]} "
        f"tiles={len(tiles)} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
