from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022"
COUNTRY_ARCHIVES = {
    "japan": ("Japan", f"{BASE_URL}/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip", 1022.9),
    "india": ("India", f"{BASE_URL}/Country_Specific_Data_CRDDC2022/RDD2022_India.zip", 502.3),
    "czech": ("Czech", f"{BASE_URL}/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip", 245.2),
    "norway": ("Norway", f"{BASE_URL}/Country_Specific_Data_CRDDC2022/RDD2022_Norway.zip", 9900.0),
    "united_states": (
        "United_States",
        f"{BASE_URL}/Country_Specific_Data_CRDDC2022/RDD2022_United_States.zip",
        423.8,
    ),
    "china_motorbike": (
        "China_MotorBike",
        f"{BASE_URL}/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip",
        183.1,
    ),
    "china_drone": (
        "China_Drone",
        f"{BASE_URL}/Country_Specific_Data_CRDDC2022/RDD2022_China_Drone.zip",
        152.8,
    ),
}
CLASSES = ("D00", "D10", "D20", "D40")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALL_COUNTRIES = list(COUNTRY_ARCHIVES.keys())


@dataclass(frozen=True)
class Rdd2022Result:
    output_dir: Path
    country: str
    downloaded: bool
    image_count: int
    annotation_count: int
    class_counts: dict[str, int]
    archive_path: Path | None
    message: str


def download_rdd2022(
    output_dir: Path = Path("data/raw/rdd2022"),
    country: str = "Czech",
    max_download_mb: float = 1024.0,
) -> Rdd2022Result:
    """Download one RDD2022 country archive and normalize Pascal VOC annotations.

    Only the annotated train split is normalized into ``<country>/{images,annotations}``.
    Test images in the RDD release do not include public annotations and are not
    copied into the detection-prep source.
    """
    key = _country_key(country)
    display_country, url, size_mb = COUNTRY_ARCHIVES[key]
    country_dir = output_dir / display_country
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted" / display_country
    country_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    if size_mb > max_download_mb:
        message = _manual_instructions(display_country, url, size_mb, max_download_mb)
        LOGGER.warning("RDD2022 download skipped: %s", message.splitlines()[0])
        (country_dir / "MANUAL_DOWNLOAD.md").write_text(message + "\n")
        return Rdd2022Result(
            output_dir=country_dir,
            country=display_country,
            downloaded=False,
            image_count=0,
            annotation_count=0,
            class_counts={label: 0 for label in CLASSES},
            archive_path=None,
            message=message,
        )

    archive_path = archives_dir / f"RDD2022_{display_country}.zip"
    try:
        _stream_download(url, archive_path)
        _extract_archive(archive_path, extracted_dir)
    except Exception as exc:
        message = _manual_instructions(display_country, url, size_mb, max_download_mb, reason=str(exc))
        LOGGER.warning("RDD2022 download skipped: %s", exc)
        (country_dir / "MANUAL_DOWNLOAD.md").write_text(message + "\n")
        return Rdd2022Result(
            output_dir=country_dir,
            country=display_country,
            downloaded=False,
            image_count=0,
            annotation_count=0,
            class_counts={label: 0 for label in CLASSES},
            archive_path=None,
            message=message,
        )

    image_count, annotation_count = normalize_rdd2022(extracted_dir, country_dir)
    class_counts = count_rdd_classes(country_dir / "annotations")
    metadata = {
        "country": display_country,
        "source_url": url,
        "archive_path": str(archive_path),
        "image_count": image_count,
        "annotation_count": annotation_count,
        "class_counts": class_counts,
        "classes": list(CLASSES),
        "license": "CC BY-SA 4.0",
    }
    (country_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (country_dir / "LAYOUT.md").write_text(describe_layout(country_dir, url, image_count, annotation_count, class_counts) + "\n")
    return Rdd2022Result(
        output_dir=country_dir,
        country=display_country,
        downloaded=True,
        image_count=image_count,
        annotation_count=annotation_count,
        class_counts=class_counts,
        archive_path=archive_path,
        message=f"RDD2022 {display_country} ready: images={image_count}, annotations={annotation_count}",
    )


def normalize_rdd2022(extracted_dir: Path, country_dir: Path) -> tuple[int, int]:
    train_root = _find_train_root(extracted_dir)
    source_images = train_root / "images"
    source_annotations = train_root / "annotations" / "xmls"
    target_images = country_dir / "images"
    target_annotations = country_dir / "annotations"
    target_images.mkdir(parents=True, exist_ok=True)
    target_annotations.mkdir(parents=True, exist_ok=True)

    image_index = {
        image_path.stem: image_path
        for image_path in sorted(source_images.rglob("*"))
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    }
    xml_paths = sorted(source_annotations.glob("*.xml"))
    copied_images: set[Path] = set()
    copied_annotations = 0
    for xml_path in tqdm(xml_paths, desc="RDD2022 annotations", unit="xml"):
        image_path = image_index.get(xml_path.stem) or _image_from_xml(xml_path, image_index)
        if image_path is None:
            continue
        image_target = target_images / image_path.name
        annotation_target = target_annotations / xml_path.name
        if not image_target.exists() or image_target.stat().st_size != image_path.stat().st_size:
            shutil.copy2(image_path, image_target)
        if not annotation_target.exists() or annotation_target.stat().st_size != xml_path.stat().st_size:
            shutil.copy2(xml_path, annotation_target)
        copied_images.add(image_target)
        copied_annotations += 1
    return len(copied_images), copied_annotations


def count_rdd_classes(annotations_dir: Path) -> dict[str, int]:
    counts = {label: 0 for label in CLASSES}
    if not annotations_dir.exists():
        return counts
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        for name, _bbox in voc_objects(xml_path):
            if name in counts:
                counts[name] += 1
    return counts


def voc_objects(xml_path: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    root = ElementTree.parse(xml_path).getroot()
    objects: list[tuple[str, tuple[int, int, int, int]]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = int(float(bbox.findtext("xmin") or 0))
            ymin = int(float(bbox.findtext("ymin") or 0))
            xmax = int(float(bbox.findtext("xmax") or 0))
            ymax = int(float(bbox.findtext("ymax") or 0))
        except ValueError:
            continue
        objects.append((name, (xmin, ymin, xmax, ymax)))
    return objects


def describe_layout(
    country_dir: Path,
    source_url: str,
    image_count: int,
    annotation_count: int,
    class_counts: dict[str, int],
) -> str:
    count_lines = [f"- {label}: {class_counts.get(label, 0)} objects" for label in CLASSES]
    return "\n".join(
        [
            "# RDD2022 detected layout",
            "",
            f"Source URL: {source_url}",
            f"Root: `{country_dir}`",
            f"Images: {image_count}",
            f"Pascal VOC XML annotations: {annotation_count}",
            "Class counts:",
            *count_lines,
            "",
            "Only the annotated train split is normalized here; public test images have no XML annotations.",
        ]
    )


def _country_key(country: str) -> str:
    key = country.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "czech_republic": "czech",
        "us": "united_states",
        "usa": "united_states",
        "united_states_of_america": "united_states",
        "china": "china_drone",
    }
    key = aliases.get(key, key)
    if key not in COUNTRY_ARCHIVES:
        available = ", ".join(sorted(COUNTRY_ARCHIVES))
        raise ValueError(f"Unsupported RDD2022 country {country!r}. Available: {available}.")
    return key


def _stream_download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "tarmac/0.1"}) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        with tmp_path.open("wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    tmp_path.replace(destination)


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".{archive_path.stem}.extracted"
    if marker.exists():
        return
    with ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok\n")


def _find_train_root(extracted_dir: Path) -> Path:
    candidates = [
        path
        for path in extracted_dir.rglob("train")
        if (path / "images").exists() and (path / "annotations" / "xmls").exists()
    ]
    if not candidates:
        raise RuntimeError(f"No RDD2022 train/images plus train/annotations/xmls layout found under {extracted_dir}.")
    return sorted(candidates, key=lambda path: len(path.parts))[0]


def _image_from_xml(xml_path: Path, image_index: dict[str, Path]) -> Path | None:
    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError:
        return None
    filename = root.findtext("filename")
    if filename:
        stem = Path(filename).stem
        if stem in image_index:
            return image_index[stem]
    return None


def download_rdd2022_all(
    output_dir: Path = Path("data/raw/rdd2022"),
    max_download_mb: float = 1024.0,
) -> list[Rdd2022Result]:
    """Download all RDD2022 country subsets sequentially."""
    return [
        download_rdd2022(output_dir=output_dir, country=key, max_download_mb=max_download_mb)
        for key in ALL_COUNTRIES
    ]


def find_rdd2022_country_dirs(raw_dir: Path = Path("data/raw/rdd2022")) -> list[Path]:
    """Return all country subdirectories that have a normalized images/ folder."""
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_dir() and (path / "images").exists() and not path.name.startswith(("_", "archives"))
    )


def rdd2022_image_labels(annotations_dir: Path) -> dict[str, list[str]]:
    """Return {image_stem: [labels]} for all annotated images in a country dir."""
    crack_classes = {"D00", "D10", "D20"}
    result: dict[str, list[str]] = {}
    if not annotations_dir.exists():
        return result
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        classes_in_image = {name for name, _bbox in voc_objects(xml_path)}
        labels: list[str] = []
        if classes_in_image & crack_classes:
            labels.append("crack")
        if "D40" in classes_in_image:
            labels.append("pothole")
        if not labels:
            labels = ["none"]
        result[xml_path.stem] = labels
    return result


def _manual_instructions(
    country: str,
    url: str,
    size_mb: float,
    max_download_mb: float,
    reason: str | None = None,
) -> str:
    lines = [
        f"RDD2022 {country} download skipped.",
        "",
        f"Archive URL: {url}",
        f"Archive size listed by upstream: {size_mb:.1f} MB",
        f"Configured max automatic download size: {max_download_mb:.1f} MB",
    ]
    if reason:
        lines.append(f"Download error: {reason}")
    lines.extend(
        [
            "",
            "Manual fallback:",
            f"1. Download the archive from the URL above.",
            f"2. Place it at `data/raw/rdd2022/archives/RDD2022_{country}.zip`.",
            f"3. Re-run `UV_CACHE_DIR=.uv-cache uv run tarmac download rdd2022 --country {country}`.",
        ]
    )
    return "\n".join(lines)
