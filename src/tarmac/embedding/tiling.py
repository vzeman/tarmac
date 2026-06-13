from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ImageInput:
    kind: str
    image: Image.Image


def make_embedding_inputs(
    image: Image.Image,
    input_size: int,
    tile_cols: int = 3,
    tile_rows: int = 2,
) -> list[ImageInput]:
    """Return one full image plus a lower-half road-region tile grid."""
    rgb = image.convert("RGB")
    inputs = [ImageInput(kind="full", image=rgb.resize((input_size, input_size)))]

    width, height = rgb.size
    for tile_index, box in enumerate(tile_boxes(width, height, tile_cols=tile_cols, tile_rows=tile_rows)):
        tile = rgb.crop(box).resize((input_size, input_size))
        inputs.append(ImageInput(kind=f"tile_{tile_index}", image=tile))

    return inputs


def tile_boxes(
    width: int,
    height: int,
    tile_cols: int = 3,
    tile_rows: int = 2,
) -> list[tuple[int, int, int, int]]:
    """Return lower-half tile boxes in left, upper, right, lower order."""
    road_top = height // 2
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
