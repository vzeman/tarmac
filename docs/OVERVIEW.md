# Tarmac — Project Overview (what we built)

A system for **assessing road & structure surface condition from imagery**: detect surface type, grade quality, find and measure cracks and structural defects, and map where problems are along a route. It has two components that connect via a GPS-tagged video sidecar:

- **A. Tarmac analysis pipeline** (Python) — the models + analysis + reporting.
- **B. RoadSurvey Recorder** (Flutter, iOS/Android) — the field capture app that records geo-tagged video for the pipeline.

---

## A. Analysis pipeline (`src/tarmac/`)

Built in phases (all on GitHub, datasets/weights gitignored):

| Capability | What it does | Headline result |
|---|---|---|
| **Surface type + quality** | DINOv3 ViT-B/16 fine-tuned (SupCon) → cosine-kNN over a FAISS reference set | type acc **0.954**, quality off-by-one **0.999** |
| **Quality embedding space** | UMAP visualization; fine-tuning separates quality grades cleanly | macro-F1 0.47→0.66 |
| **Crack detection (tile)** | binary crack head on tile embeddings | flags cracked sections per frame |
| **Crack segmentation + measurement** | **DINOv3 dense seg head** (the high-accuracy segmenter) traces exact crack pixels → area/length/width | test **IoU 0.52 / Dice 0.68** |
| **Multi-domain structural defects** | multi-label head: crack, spalling, efflorescence, exposed rebar, corrosion, across pavement/bridge/building/runway (surface-gated to avoid phantom defects) | per-label AP 0.90–0.99 (within-dataset) |
| **Condition assessment** | `tarmac assess` → PCI-proxy condition grade + repair priority (none/monitor/plan_repair/urgent), standards-grounded, with honest "visual proxy" disclaimers | — |
| **Mobile/real-time** | YOLO11 students (seg + type/quality cls) exported to ONNX; ~47–270 FPS CPU | DINOv3 stays the server/teacher model |
| **Road survey** | `tarmac survey <video>` → stream frames, locate via GPS, keep only problem frames, map + problems table | crack false-positives cut 494→82 via seg-head confirmation |
| **GPS-source auto-detection** | detects: (1) embedded per-frame GPS incl. **drone** DJI `.SRT` / GoPro GPMF, (2) video + **sidecar** `.track.json`/`.gpx`, (3) none → IMU dead-reckoning | type printed in `summary.json` |

**Modeling rationale:** representation learning + metric classification (not RL, not a world model). DINOv3 frozen backbone + task heads; cosine-kNN/FAISS for classification.

**Datasets** (see `reports/STRUCTURAL_DATASETS.md` for the full catalog + licenses): StreetSurfaceVis, RTK, RSCD, CrackAirport (CC BY 4.0), Concrete&Pavement crack, CrackForest, RDD2022, runway-Roboflow, **CODEBRIM** (note: `other-nc` — the non-crack defect head is non-commercial; see `reports/DATA_LICENSES.md`).

**Research foundation:** `reports/SURFACE_QUALITY_RESEARCH.md` — cited methodology + norms (FHWA LTPP, ASTM D6433 PCI, FHWA SNBI 0–9, AASHTO element states), and which attributes are image-assessable vs lab-only (binder content / density / water-damage progression are **lab-only**; raveling/aggregate-loss are the visual proxies).

**Key commands:** `tarmac analyze | assess | crack-measure | survey | report | visualize | train-* | evaluate-*`.

---

## B. RoadSurvey Recorder (`recording-app/`)

Flutter app (iOS 16+/Android 10+) that captures geo-referenced road video for the pipeline — solving the gap where stock phone video stores only a single start GPS point, not a continuous track. Full spec: `recording-app/SPEC.md`.

- **Continuous HEVC video + per-frame GPS/IMU/timestamp sidecar** (`.track.json` + `.gpx`) — disk-optimal vs storing images; adaptive **1 m** distance-sampling + crack-dedup done downstream in Tarmac.
- **Clever recording:** auto-pause when stationary (GPS + accelerometer fusion).
- **Landscape-first, glanceable UX:** hero record screen (full-bleed preview, big speed, REC/PAUSED/READY pill, GPS+storage-time strip, thumb-rail Start/long-press-Stop, collapsible map), pre-flight readiness, light/night themes, haptics.
- **Sessions** with swipe-to-delete; **grouped settings** incl. mount **calibration** (height/tilt/lens → written to sidecar for true crack-area scale).
- **Storage:** internal or **external USB-C SSD** (record to scratch, move to SSD on finalize), with a Start-time "external not connected — use internal?" fallback; auto-split segments (≤ configurable size).

**Status:** M1–M2 (capture + telemetry + sidecar) and the full UX redesign are built, validated (`flutter analyze` clean), and installed/tested on-device. External-storage (M4) in progress. Next: `tarmac survey --gps-sidecar` round-trip (M6).

**Build/run:** `flutter run --release -d <device>` (signing via the configured Apple Developer team).

---

## How they connect
RoadSurvey Recorder produces video + a continuous GPS sidecar → Tarmac `survey` auto-detects the sidecar (Type 2), samples 1 frame/meter along the real GPS track, runs the DINOv3 analysis, dedups cracks, and outputs an interactive map + problems table with clickable images. Drone footage (DJI/GoPro) is also supported via embedded-GPS detection (Type 1).

## Repo map
```
src/tarmac/        analysis pipeline (datasets, embedding, train, crack, defect, inference, survey, report)
recording-app/     Flutter capture app (+ SPEC.md)
reports/           metrics, research, dataset catalog, licensing, capability gallery (RESULTS.md), example images
docs/OVERVIEW.md   this file
PLAN.md            full phase plan & decisions
```
