# Labeling UI

Start with `uv run tarmac label-ui`. Opens `http://127.0.0.1:8765`.

Backend: FastAPI (`src/tarmac/labeling/server.py`). Frontend: self-contained HTML (`src/tarmac/labeling/ui.html`, no CDN).

## Modes

| Mode | Dataset | Images |
|------|---------|--------|
| Labeled | `crack_manifest.parquet` | ~170k, binary `has_crack` |
| Unlabeled | `manifest.parquet` | 9k road quality, no crack label |
| Defect | `defect_manifest.parquet` | ~98k |

## Label schema

Defined in `data/label_schema.json`. Three categories:

| Key | Type | Values / shortcuts |
|-----|------|--------------------|
| `has_crack` | binary | Crack (`c` / `1`), No Crack (`n` / `2`) |
| `material` | categorical | Asphalt (`a`/`1`), Concrete (`o`/`2`), Paving Stone (`p`/`3`), Other (`t`/`4`) |
| `quality` | categorical | Good (`g`/`1`), Fair (`f`/`2`), Poor (`r`/`3`) |

Edit `data/label_schema.json` to add/remove categories or change shortcuts.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `←→` / `hjkl` | Move grid focus / prev-next in modal |
| `Enter` / `Space` | Open modal for focused card |
| `Esc` | Close modal |
| `Tab` | Cycle active label field (Crack → Material → Quality) |
| `1`–`9` | Apply Nth value in active field |
| Letter shortcuts | Per-category (see schema table above) |
| `u` | Revert active field |
| `Shift+U` | Revert all labels for image |
| `F1`–`F4` | Apply saved preset |
| `Shift+Click` | Range select cards |
| `Ctrl/Cmd+Click` | Toggle select card |

## Filtering

- **Split pills**: All / Train / Val / Test
- **Label pills**: All / Crack / No Crack / Unknown — images disappear from grid immediately when they no longer match the active filter after labeling
- **Per-page selector**: 50 / 100 / 200 / 500
- **Sidebar**: click any source dataset to filter to it

## Scatter panel

Visual cluster explorer. Drag to lasso-select a group of semantically similar images, click "Load into grid" to fill the grid with that cluster.

Build the scatter first (one-time, ~2–3h for all 275k images):
```bash
uv run tarmac build-label-scatter
```
Output: `data/processed/label_scatter_2d.parquet`. Reload via the scatter panel's reload button without restarting the server.

## Presets (F1–F4)

Right-click any labeled card → "Save preset" → pick a slot (F1–F4). Pressing Fn applies that preset's full label set to the focused card / selected cards / open modal image.

## Corrections

Auto-saved to `data/processed/label_corrections.parquet` on every label action. Format: `{id, image_path, labels_json}` where `labels_json` is a JSON dict of `{key: value}` pairs.

Corrections survive server restarts and are merged back on load with backward-compat migration from the old binary format.
