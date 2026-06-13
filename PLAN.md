# Tarmac — Road Surface Quality Analysis: Master Plan

Goal: analyze photos/videos of road surfaces (asphalt/tarmac, concrete, paving, unpaved), embed every image (and image tiles) into a vector space, cluster by surface type + quality, classify new images via cosine similarity, and visualize how quality changes along a street. Includes a simple UI and an HTML report with an embedding scatter plot.

## Key architecture decisions

### 1. Modeling approach: Vision Transformer embeddings, not RL, not a world model
- What the project actually needs is **representation learning** (good embedding space) + **clustering/metric classification**. Reinforcement learning does not fit: there is no agent, action space, or reward loop. "RL" in the original idea maps to *self-supervised + supervised-contrastive representation learning*.
- World models (V-JEPA-style video predictors) model temporal dynamics — unnecessary and expensive for per-frame surface texture quality. **Decision: Vision Transformer.**
- Backbone: **DINOv3 ViT-B/16** (`facebook/dinov3-vitb16-pretrain-lvd1689m`, HF transformers). Trained on 1.7B images; strongest off-the-shelf texture/structure embedder, works well frozen — strong baseline before any training. Embedding = CLS token (optionally concat mean-pooled patch tokens). Note: HF weights are gated — requires accepting the DINOv3 license on the HF account and a local `HF_TOKEN`. **Fallback: DINOv2 ViT-B/14** (`facebook/dinov2-base`, Apache-2.0, ungated) — the embedder module must make the backbone configurable so both run through the identical pipeline and can be compared in Phase 2 evals.
- Fine-tuning: **Supervised Contrastive loss (SupCon)** on labeled (surface_type × quality) pairs so that quality grades form separable regions, plus a small **linear probe** head as a sanity-check classifier. Compare frozen vs fine-tuned on the same eval suite.
- Classification of new images: **kNN / cosine similarity** against cluster centroids (and optionally against the full reference set with FAISS).

### 2. Datasets (verified available, June 2026)
| Dataset | Size | Labels | Source |
|---|---|---|---|
| **StreetSurfaceVis** (primary for quality) | 9,122 street-level images | surface type (asphalt, concrete, paving stones, sett, unpaved) × quality (excellent/good/intermediate/bad/very bad) | Zenodo: https://zenodo.org/records/11449977 (CC) |
| **RSCD** (primary for scale) | 1M images, 360×240 | material (asphalt/concrete/mud/gravel) × unevenness (smooth/slight/severe) × friction | https://thu-rsxd.com/rscd/ + GitHub `ztsrxh/RSCD-Road_Surface_Classification_Dataset` (CC BY-NC). Use a stratified ~100k subset initially. |
| **RTK** | 77,547 low-res images | asphalt/paved/unpaved + quality variations | Mendeley: https://data.mendeley.com/datasets/fxy5khmhpb/1 |
| CQU-BPDD (optional, distress) | 60,056 images | 7 distress types (cracks, ravelling, repair…) + normal | GitHub `DearCaat/CQU-BPDD` (non-commercial) |
| CRACK500 / GAPs (optional, distress) | 500 / 1,969 images | pixel-wise cracks / distress classes | GitHub `fyangneil/pavement-crack-detection` |

Unified label schema: `surface_type ∈ {asphalt, concrete, paving_stones, sett, gravel, mud, unpaved}`, `quality ∈ {1..5}` (1=excellent … 5=very bad), `defects ⊂ {crack, pothole, patch, ravelling, …}` (optional), `source_dataset`.

### 3. Tiling
Full frames contain sky/cars/buildings. Pipeline embeds (a) the full image and (b) a grid of tiles from the lower image half (road region), e.g. 3×2 tiles at 224×224 effective resolution. Per-frame quality = robust aggregate (median) of road-tile predictions; tiles whose embedding is far from all road clusters are marked "non-road".

### 4. Stack
Python 3.11+, `uv` for env, PyTorch (MPS on this Mac for fine-tuning; frozen-backbone inference is cheap), timm/HF transformers, scikit-learn (k-means), hdbscan, umap-learn, FAISS (cosine kNN), ffmpeg (video→frames), Streamlit (UI), Plotly (scatter/report), SQLite or parquet for embedding store.

## Phases

### Phase 0 — Scaffold (codex)
Repo layout, `pyproject.toml` (uv), git init, config module, Makefile/CLI entry points (`tarmac <cmd>` via typer):
```
tarmac/
  data/            # raw + processed datasets (gitignored)
  src/tarmac/
    datasets/      # downloaders + unification to common schema (parquet manifest)
    embedding/     # DINOv2 embedder, tiling
    train/         # SupCon fine-tune, linear probe
    cluster/       # kmeans/hdbscan, centroid store, cosine assignment
    eval/          # metrics: accuracy, silhouette, retrieval mAP, confusion
    inference/     # photo/video pipeline
    report/        # HTML report w/ UMAP scatter + quality timeline
    ui/            # Streamlit app
  models/          # checkpoints, centroids (gitignored)
  notebooks/
```

### Phase 1 — Data acquisition (codex)
Downloaders for StreetSurfaceVis (Zenodo, automatic), RTK (Mendeley, automatic), RSCD (subset; GitHub release links — may need manual step if gated). Unify into `data/processed/manifest.parquet` with unified schema + stratified train/val/test split (70/15/15, split by location/sequence where available to avoid leakage).

### Phase 2 — Baseline embeddings + clustering (codex)
1. Embed all train images (frozen DINOv3 ViT-B/16; fallback DINOv2 ViT-B/14 if HF gating blocks; full image + tiles) → parquet/FAISS.
2. k-means (k from elbow/silhouette) and HDBSCAN on embeddings; inspect cluster ↔ (type, quality) alignment.
3. Evals on frozen embeddings: kNN classification accuracy for surface type and quality, silhouette score, UMAP 2-D scatter colored by type/quality.
4. **Gate: frozen baseline metrics reported before any training.**

### Phase 3 — Fine-tuning (codex)
SupCon fine-tune of last N blocks (MPS-friendly: ViT-B, batch 64 w/ grad accumulation, AMP) on combined labeled data; linear probe comparison. Re-run full Phase-2 eval; accept fine-tuned model only if it beats frozen baseline on val kNN quality-accuracy and silhouette.

### Phase 4 — Inference pipeline (codex)
`tarmac analyze <photo|video|dir>`: ffmpeg frame extraction (configurable fps), tiling, embedding, cluster assignment + cosine confidence, per-frame quality 1–5, GPS EXIF if present. Output: JSON + parquet of per-frame results.

### Phase 5 — Report + UI (codex)
- HTML report (Plotly): UMAP scatter of analyzed images inside the reference embedding space (reference points gray, new points colored by quality), quality-over-time/-distance line chart for video, per-cluster sample gallery, summary stats.
- Streamlit app: upload photo/video → progress → results table, quality timeline, scatter, downloadable report.

### Phase 6 — Evaluation report & iteration
Final metrics doc (`EVALUATION.md`), comparison table frozen vs fine-tuned, error analysis, next steps (more data, ViT-L, distress-specific heads).

### Phase 7 — Crack & defect detection (user-requested 2026-06-12)
1. Download + unify CQU-BPDD (60k images, 7 distress types + normal) and CRACK500/GAPs (pixel-wise crack masks) into the manifest with a `defects` multi-label column.
2. **Level 1 — defect head:** multi-label classifier (crack types, pothole, patch, ravelling) on frozen active-backbone embeddings, applied per tile at inference → coarse defect map per frame.
3. **Level 2 — defect-aware embeddings:** extend SupCon composite labels with distress labels; re-fine-tune so defect types form their own cosine-searchable clusters.
4. **Level 3 — crack heatmaps:** linear segmentation head on DINOv3 dense patch tokens trained on CRACK500 masks → crack overlay images in reports (DINOv3 excels at dense tasks with frozen backbone).
5. Inference/report/UI extended: per-tile defect probabilities, defect overlay rendering, defect filters in scatter plot.
Order: Levels 1+2 right after Phase 5 MVP; Level 3 afterwards.

### Phase 7b — Runway crack detection (user-requested 2026-06-13)
Goal: identify which sections of an airport runway are cracked. Runway pavement is the same asphalt/concrete with the same crack morphology as roads, so road/concrete crack data transfers; runway-specific imagery (top-down drone / dashcam concrete slabs) improves domain match.
Crack datasets:
| Dataset | Access | Content |
|---|---|---|
| Concrete & Pavement Crack (Mendeley `429vzbgmbx`) | keyless public API, CC BY 4.0 | 30k 227×227 images, binary crack / non-crack (concrete + pavement) |
| CRACK500, DeepCrack | GitHub, keyless | pixel-wise crack masks (asphalt + concrete) for segmentation/heatmaps |
| Roboflow `revathi-deusp/runway-crack-detection-1iq1l` | needs free `ROBOFLOW_API_KEY` | runway-specific, classes crack/mildcrack/severecrack (bbox) → convert to tile crack labels |
| ARID (Zenodo 10699570) | paper only, data not in record | reference; not auto-downloadable |
Plan:
1. Downloaders: Mendeley concrete+pavement (run now), CRACK500/DeepCrack (run now); Roboflow runway set implemented but gated on `ROBOFLOW_API_KEY` with clear instructions (do not block).
2. Build a separate **crack-detection track** (NOT mixed into quality 1–5): `has_crack` tile labels manifest. Tile cropping for bbox/mask datasets → positive tiles overlap crack regions.
3. **Crack classifier head** on active DINOv3 tile embeddings (crack probability per tile); integrate into `tarmac analyze` so every road/runway tile gets a crack flag → report highlights cracked sections + per-section crack ratio.
4. Optional crack **heatmap** (Level 3) via segmentation head on dense patch tokens (CRACK500/DeepCrack masks) for overlay showing exact crack location within a section.
5. Validate on held-out crack tiles + a runway sample; report precision/recall for crack vs non-crack.

### Phase 7c — Full-frame crack segmentation and measurement (user-requested 2026-06-13)
Goal: make top-down runway/pavement analysis cover the whole frame and move from rectangular crack tile flags to pixel-level crack geometry.
Implemented:
1. Tiling supports `region="lower_half"` and `region="full"`; full-frame mode defaults to a 3x3 grid over the entire image. `tarmac analyze --region auto` embeds a coarse full-frame 3x3 grid and chooses lower-half only when the top row is mostly non-road/sky-like; runway/top-down pavement selects full.
2. `src/tarmac/crack/segment.py` provides hybrid segmentation with no mask training requirement: sliding-window crack-head heatmap localization, dark thin-ridge extraction with `skimage` vesselness and black-hat morphology, cleanup, skeletonization, distance-transform width estimation, measurements, and red full-resolution overlays.
3. `tarmac crack-measure <image|dir>` writes `<name>_crackseg.png`, `crack_measurements.csv`, and `crack_measurements.parquet` with area, area percent, length, width, and optional metric units from `--mm-per-pixel`.
4. `tarmac analyze` writes crack geometry columns and full-resolution crackseg overlays when `--crack-segmentation` is set, or automatically when the crack head exists and the chosen region is full.
5. `tarmac report` includes a Crack geometry section with the mask overlays and measurement table.
6. Optional learned segmenter remains skipped for now: the local CRACK500/DeepCrack directories contain code mirrors but no usable masks, and Roboflow Universe does not expose a public API search endpoint for discovering arbitrary segmentation datasets. See `reports/CRACK_SEGMENTATION.md`.

### Phase 8 — YOLO mobile / real-time track (user-requested 2026-06-13)
Goal: produce small YOLO11 students for near-real-time mobile inference while keeping fine-tuned DINOv3 as the high-accuracy server-side teacher. This is **not** a weight conversion from DINOv3: YOLO models are trained directly on labels, with an optional DINOv3 teacher-to-student distillation hook for future pseudo-label expansion.

Implemented:
1. `tarmac download crackairport` downloads the Mendeley CrackAirport v1 dataset (`3v5r2fxf89`, CC BY 4.0) via the public API with zip fallback. The observed archive layout is `CrackAirport/train_images` plus `CrackAirport/train_masks`; the current public archive resolves to 2251 image/mask pairs.
2. `tarmac yolo-prep-seg` converts CrackAirport masks into Ultralytics YOLO segmentation format using OpenCV contours, class `0 = crack`, and a deterministic 70/15/15 split with seed 42.
3. `tarmac yolo-train-seg` trains YOLO11n-seg or YOLO11s-seg on Apple MPS only. CPU fallback is rejected explicitly so training either uses MPS or fails loudly.
4. `tarmac yolo-prep-cls` crops lower-half 3x2 StreetSurfaceVis road tiles from the processed manifest and builds two ImageFolder classification datasets: surface type and quality grade.
5. `tarmac yolo-train-cls --target type|quality` trains YOLO11n-cls students on MPS only. Quality reports both top-1 and off-by-one accuracy; `--distill` is present as an explicit future hook for DINOv3 pseudo-labeling.
6. `tarmac yolo-export --task seg|cls_type|cls_quality` exports best weights to mobile-oriented formats through Ultralytics. ONNX is produced for Android/edge deployment through ONNX Runtime Mobile. CoreML export is enabled with `coremltools`, but the current Ultralytics/CoreML conversion fails with `only 0-dimensional arrays can be converted to Python scalars`; TensorFlow/TFLite is intentionally not attempted on this Mac because the toolchain is heavy and fragile here.
7. `tarmac yolo-detect <image|dir>` runs the YOLO segmentation model, renders red crack overlays, and reuses the existing crack geometry measurement code so outputs are comparable with the DINOv3/classical crack pipeline.
8. `tarmac yolo-benchmark` writes `reports/YOLO_MOBILE.md` and `reports/yolo_benchmark.json` with parameters, file sizes, CPU/MPS latency, and metric headlines.

Final full-training results on 2026-06-13, MPS-only with seed 42:
- Crack segmentation: YOLO11n-seg trained for 200 epochs at 512 px and is the selected default with held-out mask mAP50 `0.1853`, mAP50-95 `0.0389`; YOLO11s-seg also trained for 200 epochs and tested lower at mask mAP50 `0.1536`, mAP50-95 `0.0293`.
- Surface type classification: YOLO11n-cls trained for 100 epochs at 224 px and reached test top-1 `0.8436` vs DINOv3 `0.954`.
- Quality classification: YOLO11n-cls reached test top-1 `0.5064`, off-by-one `0.9390`; because off-by-one was below `0.95`, YOLO11s-cls quality was also trained and kept as the better model with test top-1 `0.5026`, off-by-one `0.9459` vs DINOv3 off-by-one `0.999`.
- Export/latency: ONNX exports are available (`seg` `11.0 MB`, `type` `5.9 MB`, `quality` `20.8 MB`). Benchmark on `/tmp/tarmac_runway_test` gives crack segmentation `18.6 ms` CPU / `5.5 ms` MPS, type `3.7 ms` CPU / `1.5 ms` MPS, and quality `5.3 ms` CPU / `1.5 ms` MPS.
- Runway smoke inference: `tarmac yolo-detect /tmp/tarmac_runway_test --out runs/yolo_detect_final --device mps` correctly handled `10/12` images (`9/10` cracked detected, `1/2` non-cracked rejected).

### Phase 9 — Multi-domain structural-condition defect head (user-requested 2026-06-14)
Goal: add a multi-label structural defect track on the frozen active fine-tuned DINOv3 backbone, covering crack, spalling, efflorescence, exposed rebar, and corrosion across pavement/runway/bridge/building/generic-concrete imagery.

Implemented:
1. `tarmac embed-defects` builds `data/processed/defect_embeddings.parquet` from `data/processed/defect_manifest.parquet` using the active DINOv3 checkpoint on Apple MPS only. The cache uses full-image 224 px inputs and L2-normalized CLS embeddings. To control cost and preserve rare classes, it includes all 32,626 rows carrying any target defect plus a seed-42 stratified sample of 20,000 pure-`none` negatives, for 52,626 embedded rows.
2. `tarmac train-defect` trains a 768 -> 512 -> 5 dropout MLP with BCEWithLogits and per-class positive weights. It checkpoints every epoch under `models/checkpoints/defect/epoch_N.pt`, supports `--resume`, keeps the best validation macro-AP model at `models/defect_head.pt`, and saves validation F1-max thresholds in `models/defect_head.json`.
3. `tarmac evaluate-defect` writes `reports/DEFECT_DETECTION.md` and `reports/defect_metrics.json` with per-label precision/recall/F1/AP, per-domain macro/micro metrics, and a multi-label confusion-style summary for validation and test.
4. `tarmac analyze` loads `models/defect_head.pt` when present and adds `tile_defect_<label>_prob` / `tile_defect_<label>` to `tiles.parquet`, plus per-frame `defect_<label>_ratio`, `frame_has_defect_<label>`, `structural_defects`, and `frame_has_structural_defect` to `results.parquet`. The existing crack head remains intact and additive. `tarmac report` includes a Structural defects panel listing detected defect types per frame.
5. `scripts/smoke_defect_head.py` asserts the defect checkpoint exists, metrics contain AP for all five labels, and analyze output has the new tile/frame columns.

Final MPS run, seed 42:
- Best validation macro-AP: `0.9654` at epoch 39 of 40.
- Test per-label AP / F1: crack `0.9868 / 0.9393`, spalling `0.9660 / 0.8953`, efflorescence `0.9675 / 0.9325`, exposed rebar `0.9863 / 0.9593`, corrosion `0.8982 / 0.8513`.
- Test per-domain macro-F1: bridge `0.8975`, pavement `0.8996`, building `0.8768`, runway `0.8990`, concrete-generic `0.9986`.
- Caveat: bridge is the only domain with all five non-crack structural labels in the current manifest; building, pavement, runway, and concrete-generic domain metrics are effectively crack-transfer checks. Corrosion is the weakest label because its visual signal overlaps staining/efflorescence and has fewer positives.

## Management protocol
- Each phase executed by **local codex CLI** (`codex exec`), one detailed task prompt per phase; Claude reviews diffs/outputs, runs smoke tests, iterates with codex on failures.
- Definition of done per phase: code runs end-to-end via documented command, produces artifact (manifest/embeddings/metrics/report), reviewed by Claude.
