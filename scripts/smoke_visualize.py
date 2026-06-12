from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from tarmac.inference.analyze import load_reference_embeddings
from tarmac.report.umap_html import reference_scatter_html

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path("/tmp/tarmac_viz_test")
TEST_HTML = (ROOT / "reports" / "visualize_tarmac_viz_test.html").resolve()
BEST_HTML = (ROOT / "reports" / "umap_scatter_best.html").resolve()
REGEN_TARGETS = {
    ROOT / "reports" / "umap_scatter.html": ROOT / "data/processed/embeddings.parquet",
    ROOT / "reports" / "umap_scatter_best.html": ROOT / "data/processed/embeddings_dinov3_finetuned.parquet",
    ROOT / "reports" / "umap_scatter_dinov3_finetuned.html": ROOT / "data/processed/embeddings_dinov3_finetuned.parquet",
    ROOT / "reports" / "umap_scatter_dinov3_frozen.html": ROOT / "data/processed/embeddings_dinov3_frozen.parquet",
    ROOT / "reports" / "umap_scatter_dinov2_finetuned.html": ROOT / "data/processed/embeddings_dinov2_finetuned.parquet",
}


def main() -> None:
    manifest = pd.read_parquet(ROOT / "data/processed/manifest.parquet")
    selected = select_test_images(manifest)
    test_filenames = prepare_images(selected)

    run(["tarmac", "visualize", str(TEST_DIR), "--out", str(TEST_HTML), "--batch-size", "16"])
    assert TEST_HTML.exists(), TEST_HTML
    html = TEST_HTML.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "base64," in html
    assert 'id="img-dialog"' in html
    assert "customdata" in html
    for filename in test_filenames:
        assert filename in html, filename

    regenerate_reference_reports()
    best_html = BEST_HTML.read_text(encoding="utf-8")
    assert BEST_HTML.stat().st_size > 1_000_000, BEST_HTML.stat().st_size
    assert "plotly" in best_html.lower()
    assert 'id="img-dialog"' in best_html

    print(
        json.dumps(
            {
                "test_visualization_html": str(TEST_HTML),
                "regenerated_umap_scatter_best_html": str(BEST_HTML),
                "test_images": str(TEST_DIR),
                "image_count": len(test_filenames),
            },
            indent=2,
        )
    )


def select_test_images(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = manifest.sort_values(["quality", "surface_type", "image_path"])
    for quality in sorted(grouped["quality"].unique()):
        rows.extend(grouped[grouped["quality"] == quality].groupby("surface_type", sort=True).head(1).to_dict("records"))
    selected = pd.DataFrame(rows).drop_duplicates("image_path")
    if len(selected) < 20:
        remainder = manifest[~manifest["image_path"].isin(selected["image_path"])]
        remainder = remainder.sort_values(["surface_type", "quality", "image_path"]).head(20 - len(selected))
        selected = pd.concat([selected, remainder], ignore_index=True)
    return selected.head(20).reset_index(drop=True)


def prepare_images(selected: pd.DataFrame) -> list[str]:
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True)
    filenames: list[str] = []
    for index, row in enumerate(selected.itertuples()):
        src = ROOT / row.image_path
        subdir = TEST_DIR / f"quality_{int(row.quality)}"
        subdir.mkdir(parents=True, exist_ok=True)
        stem = f"{index:02d}_{src.stem}"
        if index % 5 == 1:
            dst = subdir / f"{stem}.png"
            with Image.open(src) as image:
                image.save(dst)
        elif index % 5 == 2:
            dst = subdir / f"{stem}.webp"
            with Image.open(src) as image:
                image.save(dst, quality=86)
        else:
            dst = subdir / f"{stem}{src.suffix.lower()}"
            shutil.copy2(src, dst)
        filenames.append(dst.name)
    return filenames


def regenerate_reference_reports() -> None:
    reducer = joblib.load(ROOT / "models/umap_reducer.pkl")
    for html_path, embeddings_path in REGEN_TARGETS.items():
        df, embeddings = load_reference_embeddings(embeddings_path)
        if embeddings_path.name == "embeddings_dinov3_finetuned.parquet" and len(reducer.embedding_) == len(df):
            projection = np.asarray(reducer.embedding_)
        else:
            projection = np.asarray(reducer.transform(embeddings))
        reference_scatter_html(
            df=df,
            projection=projection,
            path=html_path,
            title=f"Reference UMAP projection - {embeddings_path.stem}",
        )


def run(cmd: list[str]) -> None:
    subprocess.run(
        ["uv", "run", *cmd],
        cwd=ROOT,
        env={**os.environ, "UV_CACHE_DIR": ".uv-cache"},
        check=True,
    )


if __name__ == "__main__":
    main()
