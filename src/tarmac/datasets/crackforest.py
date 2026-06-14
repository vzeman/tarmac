from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

REPO_URL = "https://github.com/cuilimeng/CrackForest-dataset.git"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CrackForestResult:
    output_dir: Path
    source_dir: Path
    image_count: int
    mask_count: int
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_crackforest(output_dir: Path = Path("data/raw/crackforest")) -> CrackForestResult:
    """Clone CrackForest/CFD and normalize paired images plus binary PNG masks.

    The upstream repository currently stores 156 JPG files under ``image/`` and
    118 MATLAB ground-truth files under ``groundTruth/``. The 118 numbered
    image/ground-truth pairs are normalized into ``images/`` and ``masks/``.
    MATLAB ``groundTruth.Segmentation`` values greater than the background value
    are interpreted as crack pixels; image masks are supported as a fallback if
    a future mirror exposes masks directly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "_source"
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _clone_or_update(source_dir)
    pairs = normalize_crackforest(source_dir, images_dir, masks_dir)
    if len(pairs) != 118:
        raise RuntimeError(f"Expected 118 CrackForest image/mask pairs, found {len(pairs)}.")

    pairs_path = output_dir / "pairs.jsonl"
    with pairs_path.open("w") as handle:
        for image_path, mask_path in pairs:
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "image_relpath": str(image_path.relative_to(output_dir)),
                        "mask_relpath": str(mask_path.relative_to(output_dir)),
                        "source_dataset": "crackforest",
                    }
                )
                + "\n"
            )

    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(describe_layout(source_dir, output_dir, pairs) + "\n")
    return CrackForestResult(
        output_dir=output_dir,
        source_dir=source_dir,
        image_count=len({image for image, _ in pairs}),
        mask_count=len({mask for _, mask in pairs}),
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_crackforest_pairs(raw_dir: Path = Path("data/raw/crackforest")) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]

    images = {
        image_path.stem: image_path
        for image_path in sorted((raw_dir / "images").glob("*"))
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        mask_path.stem: mask_path
        for mask_path in sorted((raw_dir / "masks").glob("*.png"))
    }
    return sorted((images[key], masks[key]) for key in images.keys() & masks.keys())


def normalize_crackforest(source_dir: Path, images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    source_images = {
        image_path.stem: image_path
        for image_path in sorted((source_dir / "image").glob("*"))
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    }
    mask_sources = _ground_truth_sources(source_dir)
    pairs: list[tuple[Path, Path]] = []
    for stem, mask_source in tqdm(sorted(mask_sources.items()), desc="CrackForest masks", unit="mask"):
        image_source = source_images.get(stem)
        if image_source is None:
            continue
        image_target = images_dir / f"{stem}{image_source.suffix.lower()}"
        mask_target = masks_dir / f"{stem}.png"
        if not image_target.exists() or image_target.stat().st_size != image_source.stat().st_size:
            shutil.copy2(image_source, image_target)
        mask = _load_binary_mask(mask_source)
        Image.fromarray(mask, mode="L").save(mask_target)
        pairs.append((image_target, mask_target))
    return pairs


def describe_layout(source_dir: Path, output_dir: Path, pairs: list[tuple[Path, Path]]) -> str:
    source_image_count = sum(
        1 for path in (source_dir / "image").glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    mat_count = len(list((source_dir / "groundTruth").glob("*.mat")))
    return "\n".join(
        [
            "# CrackForest detected layout",
            "",
            f"Source repo: `{REPO_URL}`",
            f"Source root: `{source_dir}`",
            f"Upstream image files: {source_image_count}",
            f"Upstream MATLAB ground-truth files: {mat_count}",
            f"Normalized pairs: {len(pairs)}",
            f"Images: `{output_dir / 'images'}`",
            f"Masks: `{output_dir / 'masks'}`",
            "",
            "Mask rule: `groundTruth.Segmentation` values above the background value are written as 255 in PNG masks.",
        ]
    )


def _clone_or_update(source_dir: Path) -> None:
    if not source_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(source_dir)], check=True)
        return
    if not (source_dir / ".git").exists():
        return
    subprocess.run(["git", "-C", str(source_dir), "fetch", "--depth", "1", "origin", "master"], check=True)
    subprocess.run(["git", "-C", str(source_dir), "checkout", "--force", "FETCH_HEAD"], check=True)


def _ground_truth_sources(source_dir: Path) -> dict[str, Path]:
    ground_truth_dir = source_dir / "groundTruth"
    mat_sources = {
        path.stem: path
        for path in sorted(ground_truth_dir.glob("*.mat"))
    }
    if mat_sources:
        return mat_sources

    candidates: dict[str, Path] = {}
    for folder_name in ("groundTruth", "ground_truth", "masks", "mask", "labels", "label"):
        folder = source_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in MASK_IMAGE_EXTENSIONS:
                candidates.setdefault(_mask_stem(path), path)
    return candidates


def _load_binary_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".mat":
        mask = _load_mat_mask(path)
    else:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _load_mat_mask(path: Path) -> np.ndarray:
    mat = loadmat(path)
    if "groundTruth" in mat:
        ground_truth = mat["groundTruth"]
        if ground_truth.dtype.names and ground_truth.size:
            item = ground_truth[0, 0]
            if "Segmentation" in ground_truth.dtype.names:
                return _foreground_from_segmentation(np.asarray(item["Segmentation"]))
            if "Boundaries" in ground_truth.dtype.names:
                return np.asarray(item["Boundaries"]) > 0

    for key, value in mat.items():
        if key.startswith("__") or not isinstance(value, np.ndarray) or value.ndim < 2:
            continue
        if value.dtype.names:
            continue
        return _foreground_from_segmentation(np.asarray(value))
    raise RuntimeError(f"No usable CrackForest mask array found in {path}.")


def _foreground_from_segmentation(array: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise RuntimeError(f"Expected 2D segmentation mask, got shape {squeezed.shape}.")
    values = np.unique(squeezed)
    if len(values) <= 1:
        return squeezed > 0
    background = values.min()
    return squeezed > background


def _mask_stem(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", "_label", "-label", "_gt", "-gt", "_groundtruth"):
        stem = stem.replace(token, "")
    return stem
