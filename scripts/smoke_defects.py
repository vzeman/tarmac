from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_LABELS = {"crack", "spalling", "efflorescence", "exposed_rebar", "corrosion", "none"}
MANIFEST_PATH = Path("data/processed/defect_manifest.parquet")


def main() -> None:
    assert MANIFEST_PATH.exists(), f"Missing {MANIFEST_PATH}; run `uv run tarmac prepare-defects`."
    frame = pd.read_parquet(MANIFEST_PATH)
    required = {
        "image_path",
        "source_dataset",
        "domain",
        "structure_material",
        "labels",
        "has_crack",
        "split",
    }
    missing = required - set(frame.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert len(set(frame["domain"])) > 1, "Expected multiple domains in defect manifest."

    observed: set[str] = set()
    for labels in frame["labels"]:
        label_list = list(labels)
        assert label_list, "Empty label list found."
        observed.update(str(label) for label in label_list)
    assert observed <= EXPECTED_LABELS, f"Unexpected labels: {sorted(observed - EXPECTED_LABELS)}"
    assert EXPECTED_LABELS <= observed, f"Expected every vocab label at least once, missing: {sorted(EXPECTED_LABELS - observed)}"

    sample = frame.sample(n=min(50, len(frame)), random_state=42)
    missing_paths = [path for path in sample["image_path"] if not Path(path).exists()]
    assert not missing_paths, f"Sample contains missing image paths: {missing_paths[:5]}"
    print(
        f"defect_manifest smoke ok: rows={len(frame)}, domains={sorted(frame['domain'].unique())}, "
        f"labels={sorted(observed)}"
    )


if __name__ == "__main__":
    main()
