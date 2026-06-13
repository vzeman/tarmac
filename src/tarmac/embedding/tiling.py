from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ImageInput:
    kind: str
    image: Image.Image
    box: tuple[int, int, int, int] | None = None


def make_embedding_inputs(
    image: Image.Image,
    input_size: int,
    tile_cols: int = 3,
    tile_rows: int | None = None,
    region: str = "lower_half",
) -> list[ImageInput]:
    """Return one full image plus a road/runway-region tile grid."""
    rgb = image.convert("RGB")
    inputs = [ImageInput(kind="full", image=rgb.resize((input_size, input_size)))]

    width, height = rgb.size
    for tile_index, box in enumerate(
        tile_boxes(width, height, tile_cols=tile_cols, tile_rows=tile_rows, region=region)
    ):
        tile = rgb.crop(box).resize((input_size, input_size))
        inputs.append(ImageInput(kind=f"tile_{tile_index}", image=tile, box=box))

    return inputs


def tile_boxes(
    width: int,
    height: int,
    tile_cols: int = 3,
    tile_rows: int | None = None,
    region: str = "lower_half",
) -> list[tuple[int, int, int, int]]:
    """Return tile boxes in left, upper, right, lower order."""
    region = _normalize_region(region)
    if tile_rows is None:
        tile_rows = 3 if region == "full" else 2
    if tile_cols <= 0 or tile_rows <= 0:
        raise ValueError("tile_cols and tile_rows must be positive.")

    road_top = 0 if region == "full" else height // 2
    road_height = height - road_top
    tile_width = width / tile_cols
    tile_height = road_height / tile_rows

    boxes: list[tuple[int, int, int, int]] = []
    for row in range(tile_rows):
        for col in range(tile_cols):
            left = round(col * tile_width)
            upper = round(road_top + row * tile_height)
            right = round((col + 1) * tile_width)
            lower = round(road_top + (row + 1) * tile_height)
            boxes.append((left, upper, right, lower))
    return boxes


def _normalize_region(region: str) -> str:
    if region not in {"lower_half", "full"}:
        raise ValueError(f"Unsupported tile region: {region!r}. Expected 'lower_half' or 'full'.")
    return region
