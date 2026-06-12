from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from tarmac.embedding.tiling import ImageInput, make_embedding_inputs

LOGGER = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV2_MODEL = "facebook/dinov2-base"


@dataclass(frozen=True)
class BackboneInfo:
    requested_model: str
    model_name: str
    backbone: str
    device: str
    embedding_dim: int
    input_size: int


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        processor: Any,
        input_size: int,
        include_val_tiles: bool,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.processor = processor
        self.input_size = input_size
        self.include_val_tiles = include_val_tiles

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        path = str(row["image_path"])
        with Image.open(path) as image:
            if self.include_val_tiles and row["split"] == "val":
                image_inputs = make_embedding_inputs(image, self.input_size)
            else:
                full = image.convert("RGB").resize((self.input_size, self.input_size))
                image_inputs = [ImageInput(kind="full", image=full)]

        pixel_values = self.processor(
            images=[item.image for item in image_inputs],
            return_tensors="pt",
        )["pixel_values"]

        return {
            "image_path": [path] * len(image_inputs),
            "kind": [item.kind for item in image_inputs],
            "source_dataset": [row["source_dataset"]] * len(image_inputs),
            "surface_type": [row["surface_type"]] * len(image_inputs),
            "quality": [int(row["quality"])] * len(image_inputs),
            "split": [row["split"]] * len(image_inputs),
            "pixel_values": pixel_values,
        }


def collate_embedding_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, list[Any]] = {
        "image_path": [],
        "kind": [],
        "source_dataset": [],
        "surface_type": [],
        "quality": [],
        "split": [],
    }
    pixels = []
    for item in batch:
        for key in metadata:
            metadata[key].extend(item[key])
        pixels.append(item["pixel_values"])
    metadata["pixel_values"] = torch.cat(pixels, dim=0)
    return metadata


class HFBackboneEmbedder:
    def __init__(
        self,
        model_name: str = DINOV3_MODEL,
        checkpoint_path: Path | None = None,
        *,
        allow_fallback: bool = True,
        attn_implementation: str | None = None,
        move_to_device: bool = True,
    ) -> None:
        checkpoint = _load_checkpoint_metadata(checkpoint_path)
        if checkpoint is not None and checkpoint.get("model_name"):
            model_name = str(checkpoint["model_name"])
        self.requested_model = model_name
        self.device = _preferred_device()
        self.processor, self.model, self.model_name = self._load_with_fallback(
            model_name,
            allow_fallback=allow_fallback,
            attn_implementation=attn_implementation,
        )
        if checkpoint_path is not None:
            state = checkpoint if checkpoint is not None else torch.load(checkpoint_path, map_location="cpu")
            model_state = state.get("model_state_dict", state)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(model_state, strict=True, assign=True)
            else:
                missing, unexpected = self.model.load_state_dict(model_state, strict=False)
                if unexpected:
                    raise RuntimeError(f"Unexpected checkpoint keys for backbone: {unexpected[:5]}")
                if missing:
                    LOGGER.warning("Checkpoint load left %s backbone keys at pretrained values.", len(missing))
        if move_to_device:
            self.model.to(self.device)
        self.model.eval()
        self.input_size = _processor_input_size(self.processor)
        self.backbone = "dinov3" if "dinov3" in self.model_name else "dinov2"

    @property
    def info(self) -> BackboneInfo:
        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        return BackboneInfo(
            requested_model=self.requested_model,
            model_name=self.model_name,
            backbone=self.backbone,
            device=self.device.type,
            embedding_dim=hidden_size,
            input_size=self.input_size,
        )

    def _load_with_fallback(
        self,
        model_name: str,
        *,
        allow_fallback: bool,
        attn_implementation: str | None,
    ) -> tuple[Any, Any, str]:
        candidates = [model_name]
        if allow_fallback and model_name != DINOV2_MODEL:
            candidates.append(DINOV2_MODEL)
        for candidate in candidates:
            try:
                processor = AutoImageProcessor.from_pretrained(candidate)
                kwargs = {}
                if attn_implementation:
                    kwargs["attn_implementation"] = attn_implementation
                model = AutoModel.from_pretrained(candidate, **kwargs)
                if candidate != model_name:
                    LOGGER.warning("Using fallback backbone %s.", candidate)
                return processor, model, candidate
            except Exception as exc:
                status = _exception_status(exc)
                if allow_fallback and candidate == DINOV3_MODEL and status in {401, 403}:
                    LOGGER.warning(
                        "DINOv3 download failed with HTTP %s. The model is gated; "
                        "falling back automatically to %s.",
                        status,
                        DINOV2_MODEL,
                    )
                    continue
                if allow_fallback and candidate == model_name and candidate != DINOV2_MODEL:
                    LOGGER.warning(
                        "Failed to load %s (%s). Falling back automatically to %s.",
                        candidate,
                        exc,
                        DINOV2_MODEL,
                    )
                    continue
                raise
        raise RuntimeError("Could not load any embedding backbone.")

    @torch.inference_mode()
    def embed_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        values = pixel_values.to(self.device)
        outputs = self.model(pixel_values=values)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        if torch.isnan(embeddings).any() and self.device.type == "mps":
            raise RuntimeError("MPS produced NaN embeddings; refusing silent CPU fallback.")
        if torch.isnan(embeddings).any():
            raise RuntimeError("Backbone produced NaN embeddings on CPU.")
        return embeddings.cpu()


def embed_manifest(
    manifest_path: Path,
    output_path: Path,
    faiss_index_path: Path,
    metadata_path: Path,
    model_name: str = DINOV3_MODEL,
    checkpoint_path: Path | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
) -> BackboneInfo:
    manifest = pd.read_parquet(manifest_path)
    embedder = HFBackboneEmbedder(model_name=model_name, checkpoint_path=checkpoint_path)
    info = embedder.info
    LOGGER.info("Embedding with %s on %s.", info.model_name, info.device)

    dataset = ManifestImageDataset(
        manifest=manifest,
        processor=embedder.processor,
        input_size=embedder.input_size,
        include_val_tiles=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_embedding_batch,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    full_vectors: list[np.ndarray] = []
    try:
        for batch in tqdm(loader, desc="Embedding images", unit="batch"):
            embeddings = embedder.embed_pixel_values(batch["pixel_values"]).numpy().astype("float32")
            table = _batch_to_table(batch, embeddings, info.embedding_dim)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            full_mask = np.array(batch["kind"]) == "full"
            if full_mask.any():
                full_vectors.append(embeddings[full_mask])
    finally:
        if writer is not None:
            writer.close()

    info = embedder.info
    full_matrix = np.vstack(full_vectors).astype("float32")
    index = faiss.IndexFlatIP(info.embedding_dim)
    index.add(full_matrix)
    faiss.write_index(index, str(faiss_index_path))

    metadata = {
        **info.__dict__,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "embedding_rows": int(sum(len(chunk) for chunk in full_vectors) + len(manifest[manifest["split"] == "val"]) * 6),
        "full_rows": int(len(full_matrix)),
        "tile_rows": int(len(manifest[manifest["split"] == "val"]) * 6),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return info


def _batch_to_table(batch: dict[str, Any], embeddings: np.ndarray, dim: int) -> pa.Table:
    arrays = {
        "image_path": pa.array(batch["image_path"], type=pa.string()),
        "kind": pa.array(batch["kind"], type=pa.string()),
        "source_dataset": pa.array(batch["source_dataset"], type=pa.string()),
        "surface_type": pa.array(batch["surface_type"], type=pa.string()),
        "quality": pa.array(batch["quality"], type=pa.int8()),
        "split": pa.array(batch["split"], type=pa.string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.reshape(-1), type=pa.float32()),
            dim,
        ),
    }
    return pa.table(arrays)


def _processor_input_size(processor: Any) -> int:
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        for key in ("height", "shortest_edge", "width"):
            if key in size:
                return int(size[key])
    if isinstance(size, int):
        return size
    return 224


def _preferred_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _exception_status(exc: Exception) -> int | None:
    current: BaseException | None = exc
    while current is not None:
        status = getattr(current, "response", None)
        if status is not None and getattr(status, "status_code", None) is not None:
            return int(status.status_code)
        code = getattr(current, "status_code", None)
        if code is not None and not (isinstance(code, float) and math.isnan(code)):
            return int(code)
        current = current.__cause__ or current.__context__
    return None


def _load_checkpoint_metadata(checkpoint_path: Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        return state
    return None
