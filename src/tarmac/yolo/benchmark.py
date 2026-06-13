from __future__ import annotations

import time
from pathlib import Path

import torch

from tarmac.yolo.common import IMAGE_EXTENSIONS, file_size_mb, read_json, write_json


def benchmark_yolo(
    sample_dir: Path = Path("/tmp/tarmac_runway_test"),
    output_json: Path = Path("reports/yolo_benchmark.json"),
    output_md: Path = Path("reports/YOLO_MOBILE.md"),
    iterations: int = 20,
) -> dict[str, object]:
    from ultralytics import YOLO

    images = sorted(p for p in sample_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        fallback = Path("data/processed/yolo_crack_seg/images/test")
        images = sorted(p for p in fallback.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)[:12]
    if not images:
        raise RuntimeError("No benchmark images found.")

    models = [
        ("crack_seg", Path("models/yolo/crack_seg/weights/best.pt"), 512, "seg"),
        ("cls_type", Path("models/yolo/cls_type/weights/best.pt"), 224, "cls_type"),
        ("cls_quality", Path("models/yolo/cls_quality/weights/best.pt"), 224, "cls_quality"),
    ]
    rows: list[dict[str, object]] = []
    for name, weights, imgsz, task in models:
        if not weights.exists():
            rows.append({"model": name, "available": False, "reason": f"missing {weights}"})
            continue
        yolo = YOLO(str(weights))
        row: dict[str, object] = {
            "model": name,
            "available": True,
            "weights": str(weights),
            "weights_size_mb": file_size_mb(weights),
            "params": int(sum(p.numel() for p in yolo.model.parameters())),
        }
        for device in ("cpu", "mps"):
            if device == "mps" and not torch.backends.mps.is_available():
                row[f"{device}_ms"] = None
                row[f"{device}_fps"] = None
                continue
            ms = _time_predict(yolo, images, imgsz=imgsz, device=device, iterations=iterations)
            row[f"{device}_ms"] = ms
            row[f"{device}_fps"] = 1000.0 / ms if ms > 0 else 0.0
        row["exports"] = _export_sizes(task)
        row["export_errors"] = _export_errors(task)
        row.update(_metric_fields(name))
        rows.append(row)

    payload = {"sample_dir": str(sample_dir), "images": len(images), "iterations": iterations, "models": rows}
    write_json(output_json, payload)
    write_report(output_md, payload)
    return payload


def _time_predict(model: object, images: list[Path], imgsz: int, device: str, iterations: int) -> float:
    sample = [str(p) for p in images[: min(len(images), 8)]]
    for _ in range(2):
        list(model.predict(source=sample, imgsz=imgsz, device=device, verbose=False, stream=True))
    start = time.perf_counter()
    count = 0
    for _ in range(iterations):
        for _result in model.predict(source=sample, imgsz=imgsz, device=device, verbose=False, stream=True):
            count += 1
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(count, 1)


def _export_sizes(task: str) -> dict[str, float]:
    root = Path("models/yolo/export") / task
    if not root.exists():
        return {}
    sizes: dict[str, float] = {}
    for path in root.iterdir():
        if path.suffix.lower() in {".onnx", ".mlpackage", ".tflite", ".mlmodel"} or path.is_dir():
            sizes[path.name] = file_size_mb(path) if path.is_file() else sum(p.stat().st_size for p in path.rglob("*")) / (1024 * 1024)
    return sizes


def _export_errors(task: str) -> dict[str, str]:
    metadata = Path("models/yolo/export") / task / "export_metadata.json"
    if not metadata.exists():
        return {}
    payload = read_json(metadata)
    return {str(k): str(v) for k, v in payload.get("errors", {}).items()}


def _metric_fields(name: str) -> dict[str, float]:
    path = {
        "crack_seg": Path("models/yolo/crack_seg/metrics.json"),
        "cls_type": Path("models/yolo/cls_type/metrics.json"),
        "cls_quality": Path("models/yolo/cls_quality/metrics.json"),
    }[name]
    metrics = read_json(path)
    return {k: float(metrics[k]) for k in ("mask_map50", "mask_map", "top1", "off_by_one") if k in metrics}


def write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# YOLO Mobile Track",
        "",
        "DINOv3 remains the high-accuracy server-side teacher. These YOLO11 students are trained on labels, with an optional distillation hook, and exported for mobile runtimes rather than converted from DINOv3 weights.",
        "",
        "| Model | Params | Size MB | Metric | CPU ms | MPS ms | FPS CPU/MPS | Export sizes | Mobile suitability |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["models"]:
        if not row.get("available"):
            lines.append(f"| {row['model']} | - | - | missing | - | - | - | - | not benchmarked |")
            continue
        metric = _format_metric(row)
        exports = ", ".join(f"{k}: {v:.1f} MB" for k, v in row.get("exports", {}).items()) or "none"
        cpu_ms = row.get("cpu_ms")
        mps_ms = row.get("mps_ms")
        lines.append(
            f"| {row['model']} | {int(row['params']):,} | {float(row['weights_size_mb']):.1f} | {metric} | "
            f"{_fmt(cpu_ms)} | {_fmt(mps_ms)} | {_fps(row.get('cpu_fps'))}/{_fps(row.get('mps_fps'))} | {exports} | near-real-time target; validate on-device |"
        )
    errors = [
        f"- {row['model']}: " + ", ".join(f"{fmt}: {message}" for fmt, message in row.get("export_errors", {}).items())
        for row in payload["models"]
        if row.get("available") and row.get("export_errors")
    ]
    if errors:
        lines.extend(["", "## Export Caveats", "", *errors])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _format_metric(row: dict[str, object]) -> str:
    if "mask_map50" in row:
        return f"mask mAP50 {float(row['mask_map50']):.3f}, mAP50-95 {float(row.get('mask_map', 0.0)):.3f}"
    if row["model"] == "cls_quality":
        return f"top1 {float(row.get('top1', 0.0)):.3f}, off-by-one {float(row.get('off_by_one', 0.0)):.3f}"
    return f"top1 {float(row.get('top1', 0.0)):.3f}"


def _fmt(value: object) -> str:
    return "-" if value is None else f"{float(value):.1f}"


def _fps(value: object) -> str:
    return "-" if value is None else f"{float(value):.1f}"
