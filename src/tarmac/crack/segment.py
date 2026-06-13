from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation
from skimage.color import rgb2gray


@dataclass(frozen=True)
class CrackSegmentationResult:
    mask: np.ndarray
    heatmap: np.ndarray
    measurements: dict[str, float | int]
    overlay_path: Path | None = None


def segment_cracks(
    image: Image.Image,
    crack_head: dict[str, Any] | None,
    embedder: Any | None,
    mm_per_pixel: float | None = None,
    *,
    output_path: Path | None = None,
    prob_thresh: float = 0.3,
    min_object_px: int = 24,
    batch_size: int = 32,
) -> CrackSegmentationResult:
    """Segment crack pixels and measure crack geometry on a full image."""
    rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    heatmap = crack_probability_heatmap(
        rgb_image,
        crack_head=crack_head,
        embedder=embedder,
        batch_size=batch_size,
    )
    mask = extract_crack_mask(
        rgb,
        heatmap=heatmap,
        prob_thresh=prob_thresh,
        min_object_px=min_object_px,
    )
    measurements = measure_crack_mask(mask, mm_per_pixel=mm_per_pixel)
    overlay_path = None
    if output_path is not None:
        overlay_path = render_crack_overlay(rgb_image, mask, measurements, output_path)
    return CrackSegmentationResult(
        mask=mask,
        heatmap=heatmap,
        measurements=measurements,
        overlay_path=overlay_path,
    )


@torch.inference_mode()
def crack_probability_heatmap(
    image: Image.Image,
    crack_head: dict[str, Any] | None,
    embedder: Any | None,
    *,
    grid: int = 8,
    stride_fraction: float = 0.5,
    batch_size: int = 32,
) -> np.ndarray:
    width, height = image.size
    if crack_head is None or embedder is None:
        return np.ones((height, width), dtype=np.float32)

    tile_w = max(32, int(np.ceil(width / grid)))
    tile_h = max(32, int(np.ceil(height / grid)))
    stride_x = max(1, int(round(tile_w * stride_fraction)))
    stride_y = max(1, int(round(tile_h * stride_fraction)))
    boxes = _sliding_boxes(width, height, tile_w, tile_h, stride_x, stride_y)

    accum = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    head = crack_head["head"]
    threshold = float(crack_head.get("threshold", 0.5))
    input_size = int(getattr(embedder, "input_size", 224))

    crops: list[Image.Image] = []
    batch_boxes: list[tuple[int, int, int, int]] = []
    for box in boxes:
        crops.append(image.crop(box).resize((input_size, input_size)))
        batch_boxes.append(box)
        if len(crops) >= batch_size:
            _accumulate_probs(crops, batch_boxes, embedder, head, accum, counts)
            crops = []
            batch_boxes = []
    if crops:
        _accumulate_probs(crops, batch_boxes, embedder, head, accum, counts)

    counts[counts == 0] = 1.0
    heatmap = accum / counts
    # Keep weakly positive tiles from disappearing completely when the classifier
    # threshold is calibrated for binary tile decisions rather than segmentation.
    if threshold > 0:
        heatmap = heatmap / max(threshold, 1e-6) * 0.5
    heatmap = np.clip(filters.gaussian(heatmap, sigma=max(width, height) / 256.0), 0.0, 1.0)
    return heatmap.astype(np.float32)


def extract_crack_mask(
    rgb: np.ndarray,
    *,
    heatmap: np.ndarray,
    prob_thresh: float = 0.3,
    min_object_px: int = 24,
) -> np.ndarray:
    gray = rgb2gray(rgb)
    gray = exposure.equalize_adapthist(gray, clip_limit=0.02)
    vessel = np.maximum(
        filters.frangi(gray, sigmas=range(1, 4), black_ridges=True),
        filters.sato(gray, sigmas=range(1, 4), black_ridges=True),
    )
    blackhat = morphology.black_tophat(gray, morphology.disk(7))
    enhanced = _normalize(vessel) * 0.55 + _normalize(blackhat) * 0.45

    valid = heatmap > prob_thresh
    if not np.any(valid):
        valid = heatmap > max(0.12, float(np.quantile(heatmap, 0.70)))
    candidates = enhanced[valid]
    if candidates.size == 0:
        return np.zeros(gray.shape, dtype=bool)
    try:
        otsu = float(filters.threshold_otsu(candidates))
    except ValueError:
        otsu = float(np.mean(candidates) + np.std(candidates))
    strong = float(np.quantile(candidates, 0.975))
    ridge_thresh = max(otsu, strong)
    dark_thresh = float(np.quantile(blackhat[valid], 0.90))
    vessel_thresh = float(np.quantile(vessel[valid], 0.90))
    mask = (enhanced > ridge_thresh) & ((blackhat > dark_thresh) | (vessel > vessel_thresh)) & valid
    raw_mask = mask.astype(bool)

    mask = morphology.closing(raw_mask, morphology.disk(1))
    mask = morphology.remove_small_objects(mask, max_size=max(1, min_object_px - 1))
    mask = morphology.remove_small_holes(mask, max_size=15)
    mask = _keep_elongated_components(mask, min_area=min_object_px)
    if not np.any(mask) and int(raw_mask.sum()) >= min_object_px:
        relaxed = morphology.closing(raw_mask, morphology.disk(2))
        relaxed = morphology.remove_small_objects(relaxed, max_size=5)
        mask = _keep_elongated_components(relaxed, min_area=8)
    return mask.astype(bool)


def measure_crack_mask(mask: np.ndarray, mm_per_pixel: float | None = None) -> dict[str, float | int]:
    mask = mask.astype(bool)
    area_px = int(mask.sum())
    analyzed_area = int(mask.shape[0] * mask.shape[1])
    skeleton = morphology.skeletonize(mask)
    length_px = int(skeleton.sum())
    distance = ndi.distance_transform_edt(mask)
    widths = distance[skeleton] * 2.0
    mean_width_px = float(area_px / length_px) if length_px > 0 else 0.0
    max_width_px = float(widths.max()) if widths.size else 0.0
    labels = measure.label(mask, connectivity=2)
    n_components = int(labels.max())
    result: dict[str, float | int] = {
        "crack_area_px": area_px,
        "crack_area_pct": float(area_px / analyzed_area * 100.0) if analyzed_area else 0.0,
        "total_length_px": length_px,
        "mean_width_px": mean_width_px,
        "max_width_px": max_width_px,
        "n_components": n_components,
    }
    if mm_per_pixel is not None:
        mm = float(mm_per_pixel)
        result.update(
            {
                "crack_area_mm2": float(area_px * mm * mm),
                "total_length_mm": float(length_px * mm),
                "mean_width_mm": float(mean_width_px * mm),
                "max_width_mm": float(max_width_px * mm),
            }
        )
    return result


def render_crack_overlay(
    image: Image.Image,
    mask: np.ndarray,
    measurements: dict[str, float | int],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    red = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    red[mask.astype(bool)] = (235, 22, 35, 135)
    overlay = Image.alpha_composite(overlay, Image.fromarray(red, mode="RGBA"))

    draw = ImageDraw.Draw(overlay)
    outlines = segmentation.find_boundaries(mask.astype(bool), mode="outer")
    outline_arr = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    outline_arr[outlines] = (255, 235, 80, 230)
    overlay = Image.alpha_composite(overlay, Image.fromarray(outline_arr, mode="RGBA"))
    draw = ImageDraw.Draw(overlay)
    text = _measurement_text(measurements)
    font = ImageFont.load_default()
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    pad = 7
    draw.rounded_rectangle(
        (8, 8, bbox[2] + pad * 2 + 8, bbox[3] + pad * 2 + 8),
        radius=4,
        fill=(0, 0, 0, 170),
    )
    draw.multiline_text((8 + pad, 8 + pad), text, fill=(255, 255, 255, 255), font=font, spacing=3)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    out.save(output_path, quality=92)
    return output_path


def _accumulate_probs(
    crops: list[Image.Image],
    boxes: list[tuple[int, int, int, int]],
    embedder: Any,
    head: torch.nn.Module,
    accum: np.ndarray,
    counts: np.ndarray,
) -> None:
    pixels = embedder.processor(images=crops, return_tensors="pt")["pixel_values"]
    embeddings = embedder.embed_pixel_values(pixels).numpy().astype("float32")
    logits = head(torch.from_numpy(embeddings))
    probs = torch.sigmoid(logits).detach().cpu().numpy().astype("float32")
    for prob, (left, upper, right, lower) in zip(probs, boxes, strict=True):
        accum[upper:lower, left:right] += float(prob)
        counts[upper:lower, left:right] += 1.0


def _sliding_boxes(
    width: int,
    height: int,
    tile_w: int,
    tile_h: int,
    stride_x: int,
    stride_y: int,
) -> list[tuple[int, int, int, int]]:
    xs = _starts(width, tile_w, stride_x)
    ys = _starts(height, tile_h, stride_y)
    return [(x, y, min(width, x + tile_w), min(height, y + tile_h)) for y in ys for x in xs]


def _starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _keep_elongated_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    labels = measure.label(mask, connectivity=2)
    keep = np.zeros(mask.shape, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        aspect = max(height, width) / max(1, min(height, width))
        if aspect >= 2.0 or region.area >= min_area * 4:
            keep[labels == region.label] = True
    return keep


def _measurement_text(measurements: dict[str, float | int]) -> str:
    lines = [
        f"Area: {int(measurements['crack_area_px'])} px ({float(measurements['crack_area_pct']):.3f}%)",
        f"Length: {int(measurements['total_length_px'])} px",
        f"Mean width: {float(measurements['mean_width_px']):.2f} px",
    ]
    if "crack_area_mm2" in measurements:
        lines[0] += f" / {float(measurements['crack_area_mm2']):.1f} mm2"
        lines[1] += f" / {float(measurements['total_length_mm']):.1f} mm"
    return "\n".join(lines)
