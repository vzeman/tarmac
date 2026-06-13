from __future__ import annotations

import json
import subprocess
import struct
import xml.etree.ElementTree as ElementTree
import zlib
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests
from tqdm import tqdm

ZENODO_RECORD_API = "https://zenodo.org/api/records/2620293"
PREFERRED_ARCHIVE = "CODEBRIM_classification_balanced_dataset.zip"
FALLBACK_ARCHIVES = [
    "CODEBRIM_classification_dataset.zip",
    "CODEBRIM_cropped_dataset.zip",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CODEBRIM_XML_ORDER = [
    "background",
    "crack",
    "spallation",
    "efflorescence",
    "exposed_bars",
    "corrosion_stain",
]
DEFECT_LABELS = {"crack", "spallation", "efflorescence", "exposed_bars", "corrosion_stain"}


@dataclass(frozen=True)
class CodebrimResult:
    output_dir: Path
    archive_path: Path
    image_count: int
    class_counts: dict[str, int]
    annotations_path: Path
    layout_path: Path


def download_codebrim(output_dir: Path = Path("data/raw/codebrim")) -> CodebrimResult:
    """Download CODEBRIM from Zenodo and index its multi-label annotations.

    The preferred file is ``CODEBRIM_classification_balanced_dataset.zip`` from
    Zenodo record 2620293. It contains train/validation/test crop folders and
    ``metadata/background.xml`` plus ``metadata/defects.xml``. The XML entries
    use the crop filename as the first attribute and six binary child values.
    When child tags are generic, the upstream loader order is interpreted as:
    background, efflorescence, corrosion_stain, crack, spallation, exposed_bars.
    The resulting ``annotations.parquet`` stores one image row with CODEBRIM's
    raw class names, and ``LAYOUT.md`` documents the detected folders and XML
    child order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    files = _zenodo_files()
    archive_key = _select_archive(files)
    archive_info = files[archive_key]
    archive_path = archives_dir / archive_key
    _stream_download(str(archive_info["links"]["self"]), archive_path, int(archive_info["size"]))
    _download_license(files, output_dir)
    _extract_archive(archive_path, extracted_dir)

    dataset_root = _find_dataset_root(extracted_dir)
    annotations = build_codebrim_annotations(dataset_root)
    annotations_path = output_dir / "annotations.parquet"
    annotations.to_parquet(annotations_path, index=False)
    layout = describe_layout(dataset_root, annotations, archive_key)
    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(layout + "\n")
    counts = class_counts(annotations)
    return CodebrimResult(
        output_dir=output_dir,
        archive_path=archive_path,
        image_count=len(annotations),
        class_counts=counts,
        annotations_path=annotations_path,
        layout_path=layout_path,
    )


def build_codebrim_annotations(dataset_root: Path) -> pd.DataFrame:
    xml_paths = sorted((dataset_root / "metadata").glob("*.xml"))
    if not xml_paths:
        xml_paths = sorted(dataset_root.rglob("*.xml"))
    if not xml_paths:
        raise RuntimeError(f"No CODEBRIM XML metadata found under {dataset_root}.")

    metadata = _parse_xml_metadata(xml_paths)
    images = _index_images(dataset_root)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for key, labels in sorted(metadata.items()):
        image_path = images.get(key) or images.get(Path(key).name)
        if image_path is None:
            missing.append(key)
            continue
        split = _split_from_path(image_path)
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "image_relpath": str(image_path.relative_to(dataset_root)),
                "split_original": split,
                "codebrim_labels": sorted(labels),
            }
        )
    if not rows:
        raise RuntimeError(
            f"CODEBRIM XML metadata did not match any images under {dataset_root}; "
            f"first missing keys: {missing[:5]}"
        )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return frame


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {label: 0 for label in [*sorted(DEFECT_LABELS), "background"]}
    for labels in frame["codebrim_labels"]:
        label_set = set(labels)
        if not label_set or label_set == {"background"}:
            counts["background"] += 1
        for label in DEFECT_LABELS:
            if label in label_set:
                counts[label] += 1
    return counts


def describe_layout(dataset_root: Path, annotations: pd.DataFrame, archive_key: str) -> str:
    image_dirs = sorted(
        str(path.parent.relative_to(dataset_root))
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    image_dirs = sorted(set(image_dirs))
    xml_lines: list[str] = []
    for xml_path in sorted((dataset_root / "metadata").glob("*.xml")):
        root = ElementTree.parse(xml_path).getroot()
        first = next(iter(root), None)
        tags = [child.tag for child in list(first)] if first is not None else []
        xml_lines.append(f"- `{xml_path.relative_to(dataset_root)}` child tags/order: {tags or CODEBRIM_XML_ORDER}")
    return "\n".join(
        [
            "# CODEBRIM detected layout",
            "",
            f"Archive: `{archive_key}`",
            f"Root: `{dataset_root}`",
            f"Images indexed: {len(annotations)}",
            f"Image directories: {', '.join(image_dirs[:20])}",
            "",
            "Metadata:",
            *(xml_lines or ["- XML metadata discovered outside `metadata/`; parsed with the same crop-name rule."]),
            "",
            "Label interpretation: first XML attribute is the crop filename; six binary child values map to "
            f"{CODEBRIM_XML_ORDER} when child tag names are generic.",
        ]
    )


def _parse_xml_metadata(xml_paths: list[Path]) -> dict[str, set[str]]:
    records: dict[str, set[str]] = {}
    for xml_path in xml_paths:
        root_name = xml_path.stem
        root = ElementTree.parse(xml_path).getroot()
        for defect in root:
            if not defect.attrib:
                continue
            crop_name = str(next(iter(defect.attrib.values())))
            child_labels = _child_label_names(defect)
            labels = {
                label
                for label, child in zip(child_labels, list(defect))
                if str(child.text).strip() in {"1", "1.0", "true", "True"}
            }
            if root_name.lower() == "background" and not labels:
                labels = {"background"}
            records[crop_name] = labels or {"background"}
            records[str(Path(root_name) / crop_name)] = labels or {"background"}
    return records


def _child_label_names(defect: ElementTree.Element) -> list[str]:
    tags = [_normalise_label(child.tag) for child in list(defect)]
    if set(tags) & set(CODEBRIM_XML_ORDER):
        return tags
    return CODEBRIM_XML_ORDER[: len(tags)]


def _index_images(dataset_root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(p for p in dataset_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        rel = path.relative_to(dataset_root)
        images.setdefault(path.name, path)
        images.setdefault(str(Path(path.parent.name) / path.name), path)
        if len(rel.parts) >= 2:
            images.setdefault(str(Path(rel.parts[-2]) / rel.name), path)
        stem_without_replication = path.name.split("_-_", maxsplit=1)[0] + path.suffix
        images.setdefault(stem_without_replication, path)
        images.setdefault(str(Path(path.parent.name) / stem_without_replication), path)
    return images


def _split_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts or "training" in parts:
        return "train"
    if "val" in parts or "validation" in parts:
        return "val"
    if "test" in parts or "testing" in parts:
        return "test"
    return "unknown"


def _find_dataset_root(extracted_dir: Path) -> Path:
    candidates = [path.parent for path in extracted_dir.rglob("metadata") if path.is_dir()]
    candidates.extend(path.parent for path in extracted_dir.rglob("background.xml"))
    if candidates:
        return sorted(set(candidates), key=lambda path: len(path.parts))[0]
    return extracted_dir


def _zenodo_files() -> dict[str, dict[str, object]]:
    response = requests.get(ZENODO_RECORD_API, timeout=60)
    response.raise_for_status()
    record = response.json()
    return {file_info["key"]: file_info for file_info in record["files"]}


def _select_archive(files: dict[str, dict[str, object]]) -> str:
    for key in [PREFERRED_ARCHIVE, *FALLBACK_ARCHIVES]:
        if key in files:
            return key
    available = sorted(key for key in files if key.endswith(".zip"))
    if not available:
        raise RuntimeError("Zenodo CODEBRIM record did not expose any zip archives.")
    return available[0]


def _download_license(files: dict[str, dict[str, object]], output_dir: Path) -> None:
    if "license.md" not in files:
        return
    destination = output_dir / "license.md"
    info = files["license.md"]
    _stream_download(str(info["links"]["self"]), destination, int(info["size"]))


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".{archive_path.stem}.extracted"
    if marker.exists():
        return
    try:
        with ZipFile(archive_path) as archive:
            archive.extractall(output_dir)
    except BadZipFile:
        _extract_archive_sequential(archive_path, output_dir)
    marker.write_text("ok\n")


def _extract_archive_sequential(archive_path: Path, output_dir: Path) -> None:
    """Extract ZIPs whose central-directory offsets are broken.

    The CODEBRIM balanced archive currently exposes valid local file headers but
    invalid offsets for many central-directory entries. Walking local headers in
    central-directory order avoids those offsets while still using the trusted
    compressed sizes from the central directory.
    """
    with ZipFile(archive_path) as archive, archive_path.open("rb") as handle:
        cursor = 0
        for info in archive.infolist():
            handle.seek(cursor)
            header = handle.read(30)
            if len(header) < 30 or header[:4] != b"PK\x03\x04":
                raise BadZipFile(f"Bad local header while sequentially extracting {info.filename!r} at {cursor}")
            (
                _version,
                flags,
                method,
                _mtime,
                _mdate,
                _crc,
                _compressed_size,
                _file_size,
                name_length,
                extra_length,
            ) = struct.unpack("<HHHHHIIIHH", header[4:30])
            raw_name = handle.read(name_length)
            handle.seek(extra_length, 1)
            name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
            target = _safe_zip_target(output_dir, name)
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                compressed = handle.read(info.compress_size)
                data = _decompress_zip_member(compressed, method)
                with target.open("wb") as out:
                    out.write(data)
            cursor = handle.tell()
            if flags & 0x08:
                cursor = _skip_data_descriptor(handle, cursor)


def _decompress_zip_member(data: bytes, method: int) -> bytes:
    if method == 0:
        return data
    if method == 8:
        decompressor = zlib.decompressobj(-15)
        return decompressor.decompress(data) + decompressor.flush()
    raise BadZipFile(f"Unsupported ZIP compression method {method}")


def _skip_data_descriptor(handle, cursor: int) -> int:
    handle.seek(cursor)
    signature = handle.read(4)
    if signature == b"PK\x07\x08":
        handle.seek(12, 1)
        return handle.tell()
    if signature == b"PK\x03\x04":
        return cursor
    handle.seek(8, 1)
    return handle.tell()


def _safe_zip_target(output_dir: Path, name: str) -> Path:
    target = output_dir / name
    resolved_output = output_dir.resolve()
    resolved_target = target.resolve()
    if resolved_output != resolved_target and resolved_output not in resolved_target.parents:
        raise BadZipFile(f"Refusing to extract unsafe ZIP member {name!r}")
    return target


def _stream_download(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        return
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "tarmac/0.1"}) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or expected_size or 0)
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
    except requests.HTTPError:
        subprocess.run(["curl", "-L", "--fail", "-o", str(tmp_path), url], check=True)
    tmp_path.replace(destination)


def _normalise_label(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "background": "background",
        "none": "background",
        "no_defect": "background",
        "corrosion": "corrosion_stain",
        "corrosionstain": "corrosion_stain",
        "corrosion_stains": "corrosion_stain",
        "spalling": "spallation",
        "exposed_rebar": "exposed_bars",
        "exposed_bar": "exposed_bars",
        "exposedbars": "exposed_bars",
        "rebar": "exposed_bars",
        "calcium_leaching": "efflorescence",
    }
    return aliases.get(text, text)


def annotations_json_summary(result: CodebrimResult) -> str:
    return json.dumps(
        {
            "image_count": result.image_count,
            "class_counts": result.class_counts,
            "annotations_path": str(result.annotations_path),
        },
        indent=2,
        sort_keys=True,
    )
