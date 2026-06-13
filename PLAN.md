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

## Management protocol
- Each phase executed by **local codex CLI** (`codex exec`), one detailed task prompt per phase; Claude reviews diffs/outputs, runs smoke tests, iterates with codex on failures.
- Definition of done per phase: code runs end-to-end via documented command, produces artifact (manifest/embeddings/metrics/report), reviewed by Claude.
