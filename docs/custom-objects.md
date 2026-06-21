# Custom Object Recognition from Video Frames

This guide covers the full workflow for training custom object recognizers using labeled video survey frames:

1. Extract frames from a road survey video
2. Import them into the labeling UI
3. Annotate them (quality labels or bounding boxes)
4. Export the annotations
5. Fine-tune or train a new model head

---

## Workflow overview

```
survey video (.mp4)
    ↓ uv run tarmac survey <video>
runs/<video-name>/frames/*.jpg
    ↓ uv run tarmac import-frames runs/<video-name>
data/processed/survey_frames_manifest.parquet
    ↓ uv run tarmac label-ui   →  Survey tab
label with quality/crack corrections or draw bounding boxes
    ↓ uv run tarmac export-labeled-frames  (or export-bbox-annotations)
data/processed/survey_labeled_manifest.parquet  (or bbox_annotations_coco.json)
    ↓ uv run tarmac train --manifest ...  (or YOLO / Detectron2 training)
new model checkpoint
```

---

## Step 1 — Extract frames from a video

Run the survey pipeline on your dashcam video:

```bash
uv run tarmac survey my_road.mp4 --fps 2.0
# → runs/my_road/frames/*.jpg  (one frame every 0.5 s)
# → runs/my_road/samples.parquet  (per-frame quality predictions)
```

The `frames/` directory contains the raw extracted frames you will label.

---

## Step 2 — Import frames into the labeling UI

```bash
uv run tarmac import-frames runs/my_road
# → data/processed/survey_frames_manifest.parquet  (appended if it exists)
```

To import from multiple videos:
```bash
uv run tarmac import-frames runs/road_video_1
uv run tarmac import-frames runs/road_video_2
# Both sets are appended to the same manifest.
```

Options:
| Option | Default | Description |
|---|---|---|
| `--output` | `data/processed/survey_frames_manifest.parquet` | Manifest path |
| `--split` | `train` | Split assignment: `train`, `val`, or `test` |
| `--no-append` | off | Overwrite instead of appending |

---

## Step 3 — Open the labeling UI

```bash
uv run tarmac label-ui
# → http://127.0.0.1:8765
```

Click the **Survey** tab (next to Labeled / Unlabeled / Defect) to see your imported frames.

---

## Step 4a — Label quality / crack (SupCon training)

To produce training data for the road quality backbone or crack classifier:

1. In the Survey tab, browse your frames
2. Use the label bar (surface type, quality, has_crack) to annotate each frame
3. Use **Apply Same Labels** to batch-apply to similar frames
4. Labels are saved automatically to `data/processed/corrections.parquet`

Export the labeled frames as a SupCon manifest:

```bash
uv run tarmac export-labeled-frames
# → data/processed/survey_labeled_manifest.parquet
```

Use for fine-tuning:
```bash
uv run tarmac train --manifest data/processed/survey_labeled_manifest.parquet \
    --initial-checkpoint models/finetuned_backbone.pt
```

---

## Step 4b — Draw bounding boxes (new object types)

To label new object types like road markings, hydrants, or retarders:

1. Open the labeling UI in the Survey tab
2. Click any image to open the detail modal
3. Click **Edit Annotation** → switch to the **Bounding Boxes** tab
4. Type the class name in the **Class** field (or pick from the dropdown):
   - `road_marking` — painted lines, zebra crossings, arrows
   - `hydrant` — fire hydrants on the road edge
   - `retarder` — speed bumps / rumble strips
   - `pothole` — surface potholes
   - `crack` — localized surface cracks
   - or any custom name you choose
5. Click and drag on the image to draw a rectangle
6. Repeat for each object in the frame
7. Click **Save**

---

## Step 5 — Export bbox annotations

### COCO JSON format (Detectron2, MMDetection, etc.)

```bash
uv run tarmac export-bbox-annotations --format coco
# → data/processed/bbox_annotations_coco.json
```

The file follows the standard COCO object detection format:
```json
{
  "categories": [{"id": 1, "name": "road_marking"}, ...],
  "images": [{"id": 1, "file_name": "/path/to/frame.jpg", "width": 1920, "height": 1080}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}]
}
```

### YOLO TXT format (Ultralytics YOLOv8/v11)

```bash
uv run tarmac export-bbox-annotations --format yolo \
    --output data/processed/bbox_yolo/
# → data/processed/bbox_yolo/classes.txt
# → data/processed/bbox_yolo/<img_id>.txt  (one per annotated image)
```

Each `.txt` file contains one row per box: `class_id cx cy w h` (all normalized 0..1).

---

## Step 6 — Train a detection model

### Using Ultralytics YOLOv8/v11 (recommended for new object types)

```bash
# Install (separately from tarmac)
pip install ultralytics

# Create a dataset.yaml pointing to your images and labels
# Then:
yolo detect train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640
```

Example `dataset.yaml`:
```yaml
path: data/processed/bbox_yolo
train: .
nc: 3
names: ['road_marking', 'hydrant', 'retarder']
```

> **Note**: YOLO label files are named by image ID (MD5 hash). You need to copy or symlink the actual image files to match. For a complete YOLO dataset, use a small script to copy images alongside their `.txt` label files.

### Using MMDetection or Detectron2 (COCO format)

Use the COCO JSON export directly with any framework that supports the COCO detection API.

---

## Annotation storage locations

| Type | Storage | Export command |
|---|---|---|
| Quality / crack labels | `data/processed/corrections.parquet` | `export-labeled-frames` |
| Crack pixel masks | `data/processed/crack_annotations/` | `export-crack-annotations` |
| Bounding boxes | `data/processed/bbox_annotations/index.json` | `export-bbox-annotations` |
| Survey frame manifest | `data/processed/survey_frames_manifest.parquet` | (internal, for label-ui) |

---

## Tips

- **Batch labeling**: In the Survey tab, select multiple frames with Shift+click, then use the batch label bar at the bottom to apply the same quality/crack labels to all of them at once.
- **Crack mask + bbox on same image**: The annotation editor has two tabs. You can have both a pixel crack mask and bounding boxes on the same frame.
- **Iterative import**: Re-run `import-frames` after each new survey to add more frames. The manifest is appended, not overwritten.
- **Split assignment**: Use `--split val` when importing a held-out survey for evaluation.
- **Server reload without restart**: After running `import-frames`, you can reload the survey frames in a running server by visiting `POST /api/reload-survey-frames` — or just restart `label-ui`.
