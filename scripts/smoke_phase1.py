from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    manifest_path = Path("data/processed/manifest.parquet")
    assert manifest_path.exists(), "manifest parquet does not exist"

    manifest = pd.read_parquet(manifest_path)
    assert len(manifest) > 8000, f"expected >8000 rows, found {len(manifest)}"
    assert set(manifest["quality"].unique()).issubset({1, 2, 3, 4, 5})

    sample = manifest.sample(n=min(50, len(manifest)), random_state=42)
    missing = [path for path in sample["image_path"] if not Path(path).exists()]
    assert not missing, f"sample contains missing image paths: {missing[:5]}"
    print(f"Phase 1 smoke test passed: {len(manifest)} manifest rows.")


if __name__ == "__main__":
    main()
