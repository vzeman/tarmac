from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Tarmac Label UI")

_state: dict[str, Any] = {
    "df": None,           # crack_manifest (labeled)
    "unlabeled_df": None, # manifest.parquet (road quality, no crack label)
    "defect_df": None,    # defect_manifest.parquet
    "survey_df": None,    # survey_frames_manifest.parquet (imported video frames)
    "corrections": {},    # id -> dict, e.g. {"has_crack": 1, "material": "asphalt"}
    "id_to_row": {},      # id -> (df_key, idx)
    "corrections_path": None,
    "scatter": None,
    "scatter_coords": None,   # np.ndarray (N,2) float32 — UMAP xy for similarity search
    "scatter_ids": None,      # list[str] parallel to scatter_coords
    "scatter_id_to_idx": {},  # id -> int index into scatter_coords
    "schema": None,       # loaded from data/label_schema.json
}

_SCATTER_PATH = Path("data/processed/label_scatter_2d.parquet")
_SCHEMA_PATH = Path("data/label_schema.json")


def _image_id(image_path: str) -> str:
    return hashlib.md5(image_path.encode()).hexdigest()[:12]


def _build_id_index(df: pd.DataFrame, df_key: str, index: dict[str, tuple[str, int]]) -> None:
    for idx, row in df.iterrows():
        img_id = str(row["id"])
        if img_id not in index:
            index[img_id] = (df_key, int(idx))


def _load_schema() -> None:
    if _SCHEMA_PATH.exists():
        try:
            _state["schema"] = json.loads(_SCHEMA_PATH.read_text())
        except Exception:
            _state["schema"] = None
    else:
        _state["schema"] = None


def _load_manifest(manifest_path: Path, corrections_path: Path) -> None:
    # ── crack manifest (labeled) ──────────────────────────────────────────────
    df = pd.read_parquet(manifest_path)
    df["id"] = df["image_path"].apply(_image_id)
    df["original_label"] = df["has_crack"].astype("int64")
    df["effective_label"] = df["original_label"].copy()
    _state["df"] = df

    # ── road quality manifest (unlabeled) ─────────────────────────────────────
    unlabeled_path = Path("data/processed/manifest.parquet")
    if unlabeled_path.exists():
        udf = pd.read_parquet(unlabeled_path)
        udf["id"] = udf["image_path"].apply(_image_id)
        udf["original_label"] = -1
        udf["effective_label"] = -1
        if "split" not in udf.columns:
            udf["split"] = "train"
        _state["unlabeled_df"] = udf
    else:
        _state["unlabeled_df"] = None

    # ── defect manifest ───────────────────────────────────────────────────────
    defect_path = Path("data/processed/defect_manifest.parquet")
    if defect_path.exists():
        ddf = pd.read_parquet(defect_path)
        ddf["id"] = ddf["image_path"].apply(_image_id)
        if "has_crack" in ddf.columns:
            ddf["original_label"] = ddf["has_crack"].fillna(-1).astype("int64")
        else:
            ddf["original_label"] = -1
        ddf["effective_label"] = ddf["original_label"].copy()
        if "split" not in ddf.columns:
            ddf["split"] = "train"
        _state["defect_df"] = ddf
    else:
        _state["defect_df"] = None

    # ── survey frames manifest (imported video frames) ────────────────────────
    survey_path = Path("data/processed/survey_frames_manifest.parquet")
    if survey_path.exists():
        sdf = pd.read_parquet(survey_path)
        sdf["id"] = sdf["image_path"].apply(_image_id)
        if "original_label" not in sdf.columns:
            sdf["original_label"] = -1
        sdf["effective_label"] = sdf["original_label"].copy()
        if "split" not in sdf.columns:
            sdf["split"] = "train"
        _state["survey_df"] = sdf
    else:
        _state["survey_df"] = None

    # ── build unified id index ────────────────────────────────────────────────
    id_to_row: dict[str, tuple[str, int]] = {}
    _build_id_index(df, "labeled", id_to_row)
    if _state["unlabeled_df"] is not None:
        _build_id_index(_state["unlabeled_df"], "unlabeled", id_to_row)
    if _state["defect_df"] is not None:
        _build_id_index(_state["defect_df"], "defect", id_to_row)
    if _state["survey_df"] is not None:
        _build_id_index(_state["survey_df"], "survey", id_to_row)
    _state["id_to_row"] = id_to_row

    # ── corrections (with backward-compat migration) ──────────────────────────
    _state["corrections"] = {}
    _state["corrections_path"] = corrections_path
    if corrections_path.exists():
        try:
            corr_df = pd.read_parquet(corrections_path)
            for _, row in corr_df.iterrows():
                img_id = str(row["id"])
                if "labels_json" in corr_df.columns:
                    labels: dict = json.loads(row["labels_json"])
                else:
                    # old binary format: corrected_label int
                    old_label = int(row["corrected_label"])
                    if old_label == -1:
                        continue
                    labels = {"has_crack": old_label}
                _state["corrections"][img_id] = labels
                _apply_correction_to_df(img_id, labels)
        except Exception:
            pass

    _load_schema()
    _load_scatter()


def _apply_correction_to_df(img_id: str, labels: dict) -> None:
    entry = _state["id_to_row"].get(img_id)
    if entry is None:
        return
    df_key, idx = entry
    df = _get_df(df_key)
    if df is not None and "has_crack" in labels:
        df.at[idx, "effective_label"] = int(labels["has_crack"])


def _get_df(df_key: str) -> pd.DataFrame | None:
    if df_key == "labeled":
        return _state["df"]
    if df_key == "unlabeled":
        return _state["unlabeled_df"]
    if df_key == "defect":
        return _state["defect_df"]
    if df_key == "survey":
        return _state.get("survey_df")
    return None


def _load_scatter() -> None:
    if _SCATTER_PATH.exists():
        try:
            import numpy as np
            sdf = pd.read_parquet(_SCATTER_PATH)
            records = sdf.to_dict("records")
            id_to_row: dict[str, tuple[str, int]] = _state.get("id_to_row", {})
            for r in records:
                img_id = str(r.get("id", ""))
                entry = id_to_row.get(img_id)
                r["mode"] = entry[0] if entry else "labeled"
            _state["scatter"] = records
            # Build spatial index for similarity search
            coords = np.array([[r["x"], r["y"]] for r in records], dtype="float32")
            ids = [str(r.get("id", "")) for r in records]
            _state["scatter_coords"] = coords
            _state["scatter_ids"] = ids
            _state["scatter_id_to_idx"] = {img_id: i for i, img_id in enumerate(ids)}
        except Exception:
            _state["scatter"] = None
            _state["scatter_coords"] = None
    else:
        _state["scatter"] = None
        _state["scatter_coords"] = None


def _save_corrections() -> None:
    corrections_path: Path | None = _state["corrections_path"]
    if corrections_path is None:
        return
    corrections: dict[str, dict] = _state["corrections"]
    id_to_row: dict[str, tuple[str, int]] = _state["id_to_row"]
    rows = []
    for img_id, labels in corrections.items():
        entry = id_to_row.get(img_id)
        if entry is None:
            continue
        df_key, idx = entry
        df = _get_df(df_key)
        if df is not None:
            image_path = df.at[idx, "image_path"]
            rows.append({"id": img_id, "image_path": image_path, "labels_json": json.dumps(labels)})
    corr_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["id", "image_path", "labels_json"])
    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_parquet(corrections_path, index=False)


def _df_for_mode(mode: str) -> pd.DataFrame | None:
    if mode == "unlabeled":
        return _state["unlabeled_df"]
    if mode == "defect":
        return _state["defect_df"]
    if mode == "survey":
        return _state.get("survey_df")
    return _state["df"]


def _row_to_item(row: Any, corrections: dict[str, dict]) -> dict:
    img_id = str(row["id"])
    labels = corrections.get(img_id, {})
    orig = int(row["original_label"])
    has_crack = labels.get("has_crack")
    eff = int(has_crack) if has_crack is not None else orig
    return {
        "id": img_id,
        "path": str(row["image_path"]),
        "source": str(row.get("source_dataset", "")),
        "split": str(row.get("split", "")),
        "original_label": orig,
        "corrected_label": int(has_crack) if has_crack is not None else None,
        "effective_label": eff,
        "labels": labels,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "ui.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/image")
async def get_image(path: str = Query(...)) -> FileResponse:
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return FileResponse(str(image_path), media_type=mime_type or "image/jpeg")


@app.get("/api/schema")
async def get_schema() -> dict:
    schema = _state["schema"]
    if schema is None:
        return {"available": False, "categories": []}
    return {"available": True, **schema}


class SchemaPayload(BaseModel):
    categories: list[dict]


@app.post("/api/schema")
async def save_schema(payload: SchemaPayload) -> dict:
    schema = {"categories": payload.categories}
    _SCHEMA_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    _state["schema"] = schema
    return {"ok": True}


@app.get("/api/sources")
async def get_sources(mode: str = "labeled") -> list[dict]:
    df = _df_for_mode(mode)
    if df is None:
        return []
    corrections: dict[str, dict] = _state["corrections"]
    result = []
    for source, group in df.groupby("source_dataset", observed=True):
        crack = no_crack = unknown = 0
        for _, r in group.iterrows():
            c = corrections.get(str(r["id"]), {})
            hc = c.get("has_crack")
            eff = int(hc) if hc is not None else int(r["effective_label"])
            if eff == 1:
                crack += 1
            elif eff == 0:
                no_crack += 1
            else:
                unknown += 1
        result.append({"name": str(source), "total": len(group), "crack": crack, "no_crack": no_crack, "unknown": unknown})
    return sorted(result, key=lambda x: x["total"], reverse=True)


@app.get("/api/images")
async def get_images(
    mode: str = "labeled",
    source: str = "",
    split: str = "",
    label: str = "",
    label_key: str = "",
    label_val: str = "",
    annotation: str = "",
    page: int = 1,
    per_page: int = 50,
) -> dict:
    df = _df_for_mode(mode)
    if df is None:
        return {"items": [], "total": 0, "page": 1, "per_page": per_page, "pages": 1}

    mask = pd.Series(True, index=df.index)
    if source:
        mask &= df["source_dataset"] == source
    if split:
        mask &= df["split"] == split
    if label == "crack":
        mask &= df["effective_label"] == 1
    elif label == "no_crack":
        mask &= df["effective_label"] == 0
    elif label == "unknown":
        mask &= df["effective_label"] == -1

    if annotation in ("annotated", "unannotated"):
        from tarmac.crack.annotations import load_index
        annotated_ids = set(load_index().keys())
        if annotation == "annotated":
            mask &= df["id"].isin(annotated_ids)
        else:
            mask &= ~df["id"].isin(annotated_ids)

    if label_key and label_val:
        corrections: dict[str, dict] = _state["corrections"]
        if label_val == "__unset__":
            # Images where label_key has NOT been assigned
            set_ids = {img_id for img_id, lbs in corrections.items() if label_key in lbs}
            mask &= ~df["id"].isin(set_ids)
        else:
            matching_ids = {
                img_id for img_id, lbs in corrections.items()
                if str(lbs.get(label_key, "")) == label_val
            }
            mask &= df["id"].isin(matching_ids)

    filtered = df[mask]
    total = len(filtered)
    pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    page_df = filtered.iloc[start : start + per_page]

    corrections: dict[str, dict] = _state["corrections"]
    items = [_row_to_item(row, corrections) for _, row in page_df.iterrows()]
    return {"items": items, "total": total, "page": page, "per_page": per_page, "pages": pages}


@app.get("/api/images/by-ids")
async def get_images_by_ids(ids: str = Query(...)) -> dict:
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    corrections: dict[str, dict] = _state["corrections"]
    id_to_row: dict[str, tuple[str, int]] = _state["id_to_row"]
    items = []
    for img_id in id_list:
        entry = id_to_row.get(img_id)
        if entry is None:
            continue
        df_key, idx = entry
        df = _get_df(df_key)
        if df is None:
            continue
        row = df.iloc[idx]
        items.append(_row_to_item(row, corrections))
    return {"items": items}


class LabelRequest(BaseModel):
    id: str
    labels: dict           # {key: value|None} — None removes a key; empty dict with revert_all=True clears all
    revert_all: bool = False


class BatchLabelRequest(BaseModel):
    ids: list[str]
    labels: dict


@app.post("/api/label")
async def set_label(req: LabelRequest) -> dict:
    id_to_row: dict[str, tuple[str, int]] = _state["id_to_row"]
    entry = id_to_row.get(req.id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Image not found")
    df_key, idx = entry
    df = _get_df(df_key)
    if df is None:
        raise HTTPException(status_code=500, detail="Manifest not loaded")

    original_label = int(df.at[idx, "original_label"])

    if req.revert_all:
        _state["corrections"].pop(req.id, None)
        df.at[idx, "effective_label"] = original_label
        _save_corrections()
        return {"id": req.id, "labels": {}, "original_label": original_label, "effective_label": original_label, "corrected_label": None}

    current = _state["corrections"].get(req.id, {}).copy()
    for key, value in req.labels.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value

    if current:
        _state["corrections"][req.id] = current
    else:
        _state["corrections"].pop(req.id, None)

    has_crack = current.get("has_crack")
    df.at[idx, "effective_label"] = int(has_crack) if has_crack is not None else original_label

    _save_corrections()
    eff = int(current.get("has_crack", original_label))
    return {
        "id": req.id,
        "labels": current,
        "original_label": original_label,
        "effective_label": eff,
        "corrected_label": int(has_crack) if has_crack is not None else None,
    }


@app.post("/api/labels/batch")
async def set_labels_batch(req: BatchLabelRequest) -> dict:
    id_to_row: dict[str, tuple[str, int]] = _state["id_to_row"]
    updated = 0
    for img_id in req.ids:
        entry = id_to_row.get(img_id)
        if entry is None:
            continue
        df_key, idx = entry
        df = _get_df(df_key)
        if df is None:
            continue
        current = _state["corrections"].get(img_id, {}).copy()
        for key, value in req.labels.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        if current:
            _state["corrections"][img_id] = current
        else:
            _state["corrections"].pop(img_id, None)
        if "has_crack" in req.labels:
            hc = req.labels["has_crack"]
            df.at[idx, "effective_label"] = int(hc) if hc is not None else int(df.at[idx, "original_label"])
        updated += 1
    _save_corrections()
    return {"updated": updated}


@app.get("/api/stats")
async def get_stats() -> dict:
    df = _state["df"]
    udf = _state["unlabeled_df"]
    ddf = _state["defect_df"]
    sdf = _state.get("survey_df")
    corrections = _state["corrections"]
    result: dict[str, Any] = {"corrections": len(corrections)}
    frames = [("labeled", df, "labeled"), ("unlabeled", udf, "unlabeled"), ("defect", ddf, "defect"), ("survey", sdf, "survey")]
    for _key, frame, name in frames:
        if frame is None:
            result[name] = {"total": 0, "crack": 0, "no_crack": 0, "unknown": 0}
        else:
            result[name] = {
                "total": len(frame),
                "crack": int((frame["effective_label"] == 1).sum()),
                "no_crack": int((frame["effective_label"] == 0).sum()),
                "unknown": int((frame["effective_label"] == -1).sum()),
            }
    result["total"] = result["labeled"]["total"]
    result["crack"] = result["labeled"]["crack"]
    result["no_crack"] = result["labeled"]["no_crack"]
    return result


@app.get("/api/scatter")
async def get_scatter(mode: str = "", source: str = "") -> dict:
    scatter = _state["scatter"]
    if scatter is None:
        return {"available": False, "points": []}

    rows = scatter
    survey_fallback = False
    if mode:
        filtered = [r for r in rows if r.get("mode", "labeled") == mode]
        if filtered:
            rows = filtered
        else:
            # Survey frames not yet in scatter (need to re-run build-label-scatter).
            # Show all points so the scatter isn't empty — user can still explore.
            survey_fallback = (mode == "survey")
    if source:
        rows = [r for r in rows if str(r.get("source_dataset", "")) == source]

    total = len(rows)

    corrections = _state["corrections"]
    points = []
    for row in rows:
        img_id = str(row.get("id", ""))
        corr = corrections.get(img_id, {})
        has_crack = corr.get("has_crack")
        if has_crack is None:
            has_crack = int(row.get("has_crack", -1))
        points.append({
            "id": img_id,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "has_crack": int(has_crack),
            "corrected": img_id in corrections,
            "source": str(row.get("source_dataset", "")),
            "path": str(row.get("image_path", "")),
        })
    return {"available": True, "points": points, "total": total, "survey_fallback": survey_fallback}


@app.post("/api/scatter/reload")
async def reload_scatter() -> dict:
    _load_scatter()  # re-tags mode using current id_to_row
    n = len(_state["scatter"]) if _state["scatter"] else 0
    return {"available": _state["scatter"] is not None, "count": n}


@app.get("/api/similar")
async def get_similar(id: str, n: int = 50, mode: str = "", source: str = "") -> dict:
    import numpy as np
    coords = _state.get("scatter_coords")
    ids = _state.get("scatter_ids")
    id_to_idx = _state.get("scatter_id_to_idx", {})
    if coords is None or ids is None:
        return {"items": []}

    idx = id_to_idx.get(id)
    if idx is None:
        return {"items": []}

    query = coords[idx]
    dists = np.sum((coords - query) ** 2, axis=1)
    dists[idx] = float("inf")  # exclude self

    scatter = _state["scatter"]
    if mode or source:
        for i, r in enumerate(scatter):
            if mode and r.get("mode", "labeled") != mode:
                dists[i] = float("inf")
            if source and str(r.get("source_dataset", "")) != source:
                dists[i] = float("inf")

    k = min(n, int(np.isfinite(dists).sum()))
    if k == 0:
        return {"items": []}
    nearest = np.argpartition(dists, k)[:k]
    nearest = nearest[np.argsort(dists[nearest])]

    corrections = _state["corrections"]
    items = []
    seen_ids: set[str] = set()
    for ni in nearest:
        if not np.isfinite(dists[ni]):
            continue
        r = scatter[int(ni)]
        img_id = str(r.get("id", ""))
        if img_id in seen_ids:
            continue
        seen_ids.add(img_id)
        items.append({
            "id": img_id,
            "path": str(r.get("image_path", "")),
            "source": str(r.get("source_dataset", "")),
            "dist": round(float(dists[ni]), 5),
            "labels": corrections.get(img_id, {}),
        })
    return {"items": items[:n]}


# ── Crack annotation endpoints ────────────────────────────────────────────────

class AnnotationPayload(BaseModel):
    id: str
    image_path: str
    mask_b64: str          # base64 PNG drawn at image native size (RGBA transparent = no-crack)
    source: str = "manual"
    confidence: float | None = None
    nat_w: int | None = None
    nat_h: int | None = None


@app.get("/api/crack-heatmap")
async def get_crack_heatmap(path: str = Query(...)) -> dict:
    """Run crack segmentation model on image and return heatmap as base64 PNG."""
    import asyncio
    import base64
    import io as _io
    import numpy as np
    from PIL import Image as _Image

    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        from tarmac.crack.seg_head import predict_crack_mask, DEFAULT_CHECKPOINT
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Seg head unavailable: {exc}")

    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=503,
            detail="Crack segmentation checkpoint not found. Run: uv run tarmac train-seg-head",
        )

    # Run inference in a thread pool — predict_crack_mask is synchronous and
    # CPU/GPU-intensive; calling it directly in an async handler blocks uvicorn's
    # event loop and causes the request to silently time out.
    def _infer() -> dict:
        from PIL import ImageOps as _ImageOps
        with _Image.open(image_path) as img:
            # Apply EXIF orientation so the model sees the same orientation
            # the browser shows (browsers auto-rotate; PIL does not).
            rgb = _ImageOps.exif_transpose(img).convert("RGB")
        nat_w, nat_h = rgb.size
        mask, heatmap = predict_crack_mask(rgb, checkpoint_path=DEFAULT_CHECKPOINT, device_name="auto")
        hmap_img = _Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
        buf = _io.BytesIO()
        hmap_img.save(buf, format="PNG")
        return {
            "heatmap_b64": base64.b64encode(buf.getvalue()).decode(),
            "crack_fraction": round(float(mask.sum()) / mask.size if mask.size > 0 else 0.0, 4),
            "width": nat_w,
            "height": nat_h,
        }

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _infer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {exc}")


@app.post("/api/crack-predict-and-save")
async def crack_predict_and_save(payload: dict) -> dict:
    """Run crack segmentation model on an image and auto-save the prediction as annotation."""
    import asyncio
    import base64
    import io as _io
    import numpy as np
    from PIL import Image as _Image

    img_id: str = payload.get("id", "")
    image_path_str: str = payload.get("image_path", "")
    if not img_id or not image_path_str:
        raise HTTPException(status_code=422, detail="id and image_path required")

    image_path = Path(image_path_str)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        from tarmac.crack.seg_head import predict_crack_mask, DEFAULT_CHECKPOINT
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Seg head unavailable: {exc}")

    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=503,
            detail="Crack segmentation checkpoint not found. Run: uv run tarmac train-seg-head",
        )

    def _infer_and_save() -> dict:
        from PIL import ImageOps as _ImageOps
        from tarmac.crack.annotations import save_annotation

        with _Image.open(image_path) as img:
            rgb = _ImageOps.exif_transpose(img).convert("RGB")
        nat_w, nat_h = rgb.size

        mask, _ = predict_crack_mask(rgb, checkpoint_path=DEFAULT_CHECKPOINT, device_name="auto")

        # Build RGBA mask PNG: white+opaque where crack, transparent elsewhere
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask > 0] = [255, 255, 255, 255]
        mask_img = _Image.fromarray(rgba, mode="RGBA")
        buf = _io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        save_annotation(
            img_id, image_path_str, mask_b64,
            source="ai_prediction", nat_w=nat_w, nat_h=nat_h,
        )

        crack_fraction = round(float(mask.sum()) / mask.size if mask.size > 0 else 0.0, 4)
        return {
            "ok": True,
            "mask_b64": mask_b64,
            "crack_fraction": crack_fraction,
            "nat_w": nat_w,
            "nat_h": nat_h,
        }

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _infer_and_save)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.get("/api/crack-annotation")
async def get_crack_annotation(id: str = Query(...)) -> dict:
    """Return saved annotation metadata + mask PNG as base64 for a given image id."""
    import base64
    from tarmac.crack.annotations import get_annotation

    entry = get_annotation(id)
    if entry is None:
        return {"exists": False}

    mask_path = Path(entry.get("mask_path", ""))
    mask_b64 = None
    if mask_path.exists():
        mask_b64 = base64.b64encode(mask_path.read_bytes()).decode()

    return {
        "exists": True,
        "source": entry.get("source", "manual"),
        "confidence": entry.get("confidence"),
        "mask_b64": mask_b64,
    }


@app.post("/api/crack-annotation")
async def save_crack_annotation(payload: AnnotationPayload) -> dict:
    """Persist a drawn crack mask and return the saved annotation entry."""
    from tarmac.crack.annotations import save_annotation

    try:
        entry = save_annotation(
            payload.id,
            payload.image_path,
            payload.mask_b64,
            source=payload.source,
            confidence=payload.confidence,
            nat_w=payload.nat_w,
            nat_h=payload.nat_h,
        )
        return {"ok": True, "entry": entry}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save annotation: {exc}")


@app.delete("/api/crack-annotation")
async def delete_crack_annotation(id: str = Query(...)) -> dict:
    from tarmac.crack.annotations import delete_annotation
    deleted = delete_annotation(id)
    return {"ok": deleted}


@app.get("/api/crack-review-queue")
async def get_crack_review_queue(page: int = 1, per_page: int = 20) -> dict:
    """Return crack-labeled images that don't have a manual annotation yet."""
    from tarmac.crack.annotations import load_index

    annotated_ids = set(load_index().keys())
    corrections: dict[str, dict] = _state["corrections"]
    id_to_row: dict[str, tuple[str, int]] = _state["id_to_row"]

    # Collect crack-labeled image ids without an annotation
    queue: list[str] = []
    seen: set[str] = set()

    for df_key in ("labeled", "unlabeled", "defect"):
        df = _get_df(df_key)
        if df is None:
            continue
        for _, row in df.iterrows():
            img_id = str(row["id"])
            if img_id in seen or img_id in annotated_ids:
                continue
            seen.add(img_id)
            corr = corrections.get(img_id, {})
            hc = corr.get("has_crack")
            eff = int(hc) if hc is not None else int(row.get("effective_label", -1))
            if eff == 1:
                queue.append(img_id)

    total = len(queue)
    start = (page - 1) * per_page
    page_ids = queue[start : start + per_page]

    items = []
    for img_id in page_ids:
        entry = id_to_row.get(img_id)
        if entry is None:
            continue
        df_key, idx = entry
        df = _get_df(df_key)
        if df is None:
            continue
        items.append(_row_to_item(df.iloc[idx], corrections))

    return {"items": items, "total": total, "page": page, "per_page": per_page}


@app.get("/api/crack-annotations-batch")
async def get_crack_annotations_batch(ids: str = Query(...)) -> dict:
    """Return masks for a comma-separated list of image IDs (only those with annotations)."""
    import base64
    from tarmac.crack.annotations import load_index

    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    index = load_index()
    result: dict[str, dict] = {}
    for img_id in id_list:
        entry = index.get(img_id)
        if entry is None:
            continue
        mask_path = Path(entry.get("mask_path", ""))
        if not mask_path.exists():
            continue
        result[img_id] = {
            "mask_b64": base64.b64encode(mask_path.read_bytes()).decode(),
            "nat_w": entry.get("width"),
            "nat_h": entry.get("height"),
        }
    return {"annotations": result}


@app.get("/api/crack-annotation-stats")
async def get_crack_annotation_stats() -> dict:
    from tarmac.crack.annotations import annotation_stats
    return annotation_stats()


# ── Bounding-box annotation endpoints ────────────────────────────────────────

class BboxPayload(BaseModel):
    id: str
    image_path: str
    bboxes: list[dict]
    nat_w: int | None = None
    nat_h: int | None = None


@app.get("/api/bbox-annotation")
async def get_bbox_annotation(id: str = Query(...)) -> dict:
    """Return saved bounding-box annotations for a given image id."""
    from tarmac.labeling.bbox_annotations import get_bboxes

    entry = get_bboxes(id)
    if entry is None:
        return {"exists": False}
    return {
        "exists": True,
        "bboxes": entry.get("bboxes", []),
        "width": entry.get("width"),
        "height": entry.get("height"),
    }


@app.post("/api/bbox-annotation")
async def save_bbox_annotation(payload: BboxPayload) -> dict:
    """Persist bounding-box annotations for one image."""
    from tarmac.labeling.bbox_annotations import save_bboxes

    try:
        entry = save_bboxes(
            payload.id,
            payload.image_path,
            payload.bboxes,
            nat_w=payload.nat_w,
            nat_h=payload.nat_h,
        )
        return {"ok": True, "entry": entry}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save bboxes: {exc}")


@app.delete("/api/bbox-annotation")
async def delete_bbox_annotation(id: str = Query(...)) -> dict:
    from tarmac.labeling.bbox_annotations import delete_bboxes

    deleted = delete_bboxes(id)
    return {"ok": deleted}


@app.get("/api/bbox-annotations-batch")
async def get_bbox_annotations_batch(ids: str = Query(...)) -> dict:
    """Return bbox annotations for a comma-separated list of image IDs."""
    from tarmac.labeling.bbox_annotations import load_index as _load_bbox_index

    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    index = _load_bbox_index()
    result: dict[str, dict] = {}
    for img_id in id_list:
        entry = index.get(img_id)
        if entry:
            result[img_id] = {
                "bboxes": entry.get("bboxes", []),
                "width": entry.get("width"),
                "height": entry.get("height"),
            }
    return {"annotations": result}


@app.get("/api/bbox-stats")
async def get_bbox_stats() -> dict:
    from tarmac.labeling.bbox_annotations import bbox_stats
    return bbox_stats()


# ── Survey frames endpoints ───────────────────────────────────────────────────

@app.post("/api/reload-survey-frames")
async def reload_survey_frames() -> dict:
    """Reload survey_frames_manifest.parquet from disk without restarting the server."""
    survey_path = Path("data/processed/survey_frames_manifest.parquet")
    if not survey_path.exists():
        _state["survey_df"] = None
        return {"available": False, "total": 0}
    try:
        sdf = pd.read_parquet(survey_path)
        sdf["id"] = sdf["image_path"].apply(_image_id)
        if "original_label" not in sdf.columns:
            sdf["original_label"] = -1
        sdf["effective_label"] = sdf["original_label"].copy()
        if "split" not in sdf.columns:
            sdf["split"] = "train"
        _state["survey_df"] = sdf
        # Rebuild id index entries for survey frames
        id_to_row: dict[str, tuple[str, int]] = _state["id_to_row"]
        _build_id_index(sdf, "survey", id_to_row)
        return {"available": True, "total": len(sdf)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


# ── Dataset management helpers ────────────────────────────────────────────────

_SURVEY_MANIFEST_PATH = Path("data/processed/survey_frames_manifest.parquet")
_SURVEY_SAVE_COLS = ["image_path", "id", "source_dataset", "split", "original_label"]


def _save_survey_manifest() -> None:
    sdf = _state.get("survey_df")
    _SURVEY_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if sdf is None or len(sdf) == 0:
        pd.DataFrame(columns=_SURVEY_SAVE_COLS).to_parquet(_SURVEY_MANIFEST_PATH, index=False)
    else:
        cols = [c for c in _SURVEY_SAVE_COLS if c in sdf.columns]
        sdf[cols].to_parquet(_SURVEY_MANIFEST_PATH, index=False)


def _rebuild_survey_index() -> None:
    id_to_row = _state["id_to_row"]
    stale = [k for k, v in list(id_to_row.items()) if v[0] == "survey"]
    for k in stale:
        del id_to_row[k]
    sdf = _state.get("survey_df")
    if sdf is not None and len(sdf) > 0:
        _build_id_index(sdf, "survey", id_to_row)


# ── Dataset management endpoints ──────────────────────────────────────────────

@app.delete("/api/survey-image")
async def delete_survey_image(id: str = Query(...)) -> dict:
    """Remove a single image from the survey frames manifest."""
    sdf = _state.get("survey_df")
    if sdf is None:
        raise HTTPException(status_code=404, detail="No survey data loaded")
    mask = sdf["id"] == id
    if not mask.any():
        raise HTTPException(status_code=404, detail="Image not found in survey manifest")
    _state["survey_df"] = sdf[~mask].reset_index(drop=True)
    _rebuild_survey_index()
    _save_survey_manifest()
    return {"ok": True, "removed": 1}


class DeleteSurveyImagesPayload(BaseModel):
    ids: list[str]


@app.post("/api/survey-images/delete")
async def delete_survey_images_batch(payload: DeleteSurveyImagesPayload) -> dict:
    """Remove a batch of images from the survey frames manifest."""
    sdf = _state.get("survey_df")
    if sdf is None:
        raise HTTPException(status_code=404, detail="No survey data loaded")
    id_set = set(payload.ids)
    removed = int(sdf["id"].isin(id_set).sum())
    _state["survey_df"] = sdf[~sdf["id"].isin(id_set)].reset_index(drop=True)
    _rebuild_survey_index()
    _save_survey_manifest()
    return {"ok": True, "removed": removed}


class MoveSurveyImagesPayload(BaseModel):
    ids: list[str]
    target_dataset: str


@app.post("/api/survey-image/move")
async def move_survey_images(payload: MoveSurveyImagesPayload) -> dict:
    """Move images to a different survey dataset (change source_dataset)."""
    sdf = _state.get("survey_df")
    if sdf is None:
        raise HTTPException(status_code=404, detail="No survey data loaded")
    id_set = set(payload.ids)
    moved = int(sdf["id"].isin(id_set).sum())
    sdf.loc[sdf["id"].isin(id_set), "source_dataset"] = payload.target_dataset
    _state["survey_df"] = sdf
    _save_survey_manifest()
    return {"ok": True, "moved": moved}


@app.delete("/api/survey-dataset")
async def delete_survey_dataset(name: str = Query(...)) -> dict:
    """Remove all images belonging to a survey dataset from the manifest."""
    sdf = _state.get("survey_df")
    if sdf is None:
        raise HTTPException(status_code=404, detail="No survey data loaded")
    mask = sdf["source_dataset"] == name
    removed = int(mask.sum())
    _state["survey_df"] = sdf[~mask].reset_index(drop=True)
    _rebuild_survey_index()
    _save_survey_manifest()
    return {"ok": True, "removed": removed}


class ImportSurveyFramesPayload(BaseModel):
    run_dir: str
    dataset_name: str = ""
    append: bool = True


@app.post("/api/survey-frames/import")
async def import_survey_frames_api(payload: ImportSurveyFramesPayload) -> dict:
    """Import frames from a survey run directory into the manifest."""
    import asyncio

    run_dir = Path(payload.run_dir)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {run_dir}")

    def _do_import() -> dict:
        from tarmac.datasets.survey_frames import import_survey_frames
        result = import_survey_frames(run_dir, append=payload.append)
        custom_name = payload.dataset_name.strip()
        if custom_name and custom_name != run_dir.name and _SURVEY_MANIFEST_PATH.exists():
            df = pd.read_parquet(_SURVEY_MANIFEST_PATH)
            df.loc[df["source_dataset"] == run_dir.name, "source_dataset"] = custom_name
            df.to_parquet(_SURVEY_MANIFEST_PATH, index=False)
            result["source"] = custom_name
        return result

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _do_import)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import failed: {exc}")

    # Reload in-memory state
    if _SURVEY_MANIFEST_PATH.exists():
        sdf = pd.read_parquet(_SURVEY_MANIFEST_PATH)
        sdf["id"] = sdf["image_path"].apply(_image_id)
        if "original_label" not in sdf.columns:
            sdf["original_label"] = -1
        sdf["effective_label"] = sdf["original_label"].copy()
        if "split" not in sdf.columns:
            sdf["split"] = "train"
        _state["survey_df"] = sdf
        _rebuild_survey_index()

    return {"ok": True, **result}


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_RUNS_DIRS = [Path("runs"), Path("data/raw")]


@app.get("/api/survey-run-dirs")
async def list_survey_run_dirs() -> dict:
    """Return candidate survey run directories with frame counts and sample image paths."""
    already_imported: set[str] = set()
    sdf = _state.get("survey_df")
    if sdf is not None and len(sdf) > 0:
        already_imported = set(sdf["source_dataset"].unique().tolist())

    runs: list[dict] = []
    seen: set[str] = set()
    for base in _RUNS_DIRS:
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name in seen:
                continue
            # Collect image files from frames/, problem_images/, or root
            all_images: list[Path] = []
            for sub in ("frames", "problem_images", ""):
                d = entry / sub if sub else entry
                if d.is_dir():
                    all_images.extend(
                        sorted(f for f in d.iterdir()
                               if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
                    )
            if not all_images:
                continue
            seen.add(entry.name)
            # Pick 4 evenly-spaced sample frames for preview
            n = len(all_images)
            indices = [int(i * (n - 1) / 3) for i in range(4)] if n >= 4 else list(range(n))
            samples = [str(all_images[i]) for i in indices]
            runs.append({
                "name": entry.name,
                "path": str(entry),
                "frame_count": n,
                "imported": entry.name in already_imported,
                "samples": samples,
            })

    return {"runs": runs}


def run_server(manifest_path: Path, corrections_path: Path, host: str, port: int) -> None:
    import uvicorn
    _load_manifest(manifest_path, corrections_path)
    uvicorn.run(app, host=host, port=port)
