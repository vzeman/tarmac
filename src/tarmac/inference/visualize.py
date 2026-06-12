from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from tarmac.embedding.embedder import HFBackboneEmbedder
from tarmac.inference.analyze import (
    load_active_artifacts,
    load_reference_embeddings,
    normalize_rows,
    predict_tile,
)
from tarmac.report.umap_html import visualize_scatter_html

ImageFile.LOAD_TRUNCATED_IMAGES = True

VISUALIZE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def visualize_directory(
    directory: Path,
    out: Path | None = None,
    *,
    k: int = 10,
    batch_size: int = 16,
    device: str = "cpu",
    reducer_path: Path = Path("models/umap_reducer.pkl"),
) -> Path:
    torch.set_num_threads(1)
    try:
        faiss.omp_set_num_threads(1)
    except AttributeError:
        pass

    directory = directory.expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Visualize expects a directory: {directory}")
    image_paths = discover_images(directory)
    if not image_paths:
        raise ValueError(f"No jpg/png/webp images found under {directory}")

    artifacts = load_active_artifacts()
    ref_df, ref_embeddings = load_reference_embeddings(artifacts.embeddings_path)
    index = faiss.read_index(str(artifacts.faiss_index_path))
    if index.ntotal != len(ref_embeddings):
        raise RuntimeError(
            f"FAISS index rows ({index.ntotal}) do not match reference embeddings ({len(ref_embeddings)})"
        )
    centroids = np.load(artifacts.centroids_path).astype("float32")
    centroids = normalize_rows(centroids)
    embedder = HFBackboneEmbedder(
        model_name=artifacts.model_name,
        checkpoint_path=artifacts.checkpoint_path,
        allow_fallback=False,
        device_name=device,
        attn_implementation="eager",
    )

    embeddings = embed_full_images(image_paths, embedder, batch_size=batch_size)
    rows: list[dict[str, Any]] = []
    for path, embedding in zip(image_paths, embeddings, strict=True):
        pred = predict_tile(
            embedding=embedding,
            reference_df=ref_df,
            index=index,
            centroids=centroids,
            k=k,
            threshold=-1.0,
        )
        rows.append(
            {
                "source_path": str(path),
                "filename": path.name,
                "relative_path": str(path.relative_to(directory)),
                "predicted_quality": int(pred["predicted_quality"]),
                "surface_type": str(pred["surface_type"]),
                "confidence": float(pred["confidence"]),
                "thumbnail_data_url": thumbnail_data_url(path),
            }
        )

    reducer = joblib.load(reducer_path)
    ref_xy = reducer.embedding_
    if len(ref_xy) != len(ref_df):
        ref_xy = reducer.transform(ref_embeddings)
    item_xy = reducer.transform(np.vstack(embeddings).astype("float32"))
    item_df = pd.DataFrame(rows)

    output = out or Path("reports") / f"visualize_{directory.name}.html"
    output = output.expanduser().resolve()
    visualize_scatter_html(
        ref_xy=np.asarray(ref_xy),
        ref_df=ref_df,
        item_xy=np.asarray(item_xy),
        item_df=item_df,
        path=output,
        title=f"Tarmac Vector Space - {directory.name}",
    )
    return output


def discover_images(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VISUALIZE_IMAGE_EXTENSIONS
    )


def embed_full_images(
    paths: list[Path],
    embedder: HFBackboneEmbedder,
    *,
    batch_size: int,
) -> list[np.ndarray]:
    records: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            records.append(image.convert("RGB").resize((embedder.input_size, embedder.input_size)))

    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(records), batch_size), desc="Embedding images", unit="batch"):
        batch = records[start : start + batch_size]
        pixels = embedder.processor(images=batch, return_tensors="pt")["pixel_values"]
        emb = embedder.embed_pixel_values(pixels).numpy().astype("float32")
        embeddings.extend(list(emb))
    return embeddings


def thumbnail_data_url(path: Path, max_size: int = 512) -> str:
    with Image.open(path) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail((max_size, max_size))
        buffer = BytesIO()
        thumb.save(buffer, format="JPEG", quality=84)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
