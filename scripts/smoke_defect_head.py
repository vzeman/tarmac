from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXPECTED_LABELS = {"crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"}
CHECKPOINT_PATH = Path("models/defect_head.pt")
METRICS_PATH = Path("reports/defect_metrics.json")
RUN_DIR = Path("runs/defect_smoke")


def main() -> None:
    assert CHECKPOINT_PATH.exists(), f"Missing {CHECKPOINT_PATH}; run `uv run tarmac train-defect`."
    assert METRICS_PATH.exists(), f"Missing {METRICS_PATH}; run `uv run tarmac evaluate-defect`."
    metrics = json.loads(METRICS_PATH.read_text())
    observed = set(metrics.get("label_vocab", []))
    assert EXPECTED_LABELS <= observed, f"Missing labels in metrics vocab: {sorted(EXPECTED_LABELS - observed)}"
    for split in ("val", "test"):
        per_label = metrics.get(split, {}).get("per_label", {})
        for label in EXPECTED_LABELS:
            assert label in per_label, f"Missing {split} metric for {label}"
            assert "ap" in per_label[label], f"Missing AP for {split}/{label}"

    tiles_path = RUN_DIR / "tiles.parquet"
    results_path = RUN_DIR / "results.parquet"
    assert tiles_path.exists(), f"Missing analyze tiles output: {tiles_path}"
    assert results_path.exists(), f"Missing analyze results output: {results_path}"
    tiles = pd.read_parquet(tiles_path)
    results = pd.read_parquet(results_path)
    for label in EXPECTED_LABELS:
        assert f"tile_defect_{label}_prob" in tiles.columns, f"Missing tile probability for {label}"
        assert f"tile_defect_{label}" in tiles.columns, f"Missing tile flag for {label}"
        assert f"defect_{label}_ratio" in results.columns, f"Missing frame ratio for {label}"
        assert f"frame_has_defect_{label}" in results.columns, f"Missing frame flag for {label}"
    assert "structural_defects" in results.columns, "Missing structural_defects frame column"
    print(
        f"defect head smoke ok: labels={sorted(EXPECTED_LABELS)}, "
        f"tiles={len(tiles)}, frames={len(results)}, run={RUN_DIR}"
    )


if __name__ == "__main__":
    main()

