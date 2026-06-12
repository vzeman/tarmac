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
    road_top = height // 2
    road_height = height - road_top
    tile_width = width / tile_cols
    tile_height = road_height / tile_rows

    for row in range(tile_rows):
        for col in range(tile_cols):
            left = round(col * tile_width)
            upper = round(road_top + row * tile_height)
            right = round((col + 1) * tile_width)
            lower = round(road_top + (row + 1) * tile_height)
            tile = rgb.crop((left, upper, right, lower)).resize((input_size, input_size))
            inputs.append(ImageInput(kind=f"tile_{row * tile_cols + col}", image=tile))

    return inputs
