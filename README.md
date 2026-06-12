# Tarmac

Road surface quality analysis using unified image manifests, vision embeddings, clustering, and later reporting/UI workflows.

## Quickstart

```bash
uv sync
uv run tarmac download streetsurfacevis
uv run tarmac prepare
uv run python scripts/smoke_phase1.py
```

The Phase 1 workflow downloads StreetSurfaceVis v1.0 1024px images plus labels into `data/raw/streetsurfacevis/`, then writes a unified manifest to `data/processed/manifest.parquet`.

Useful commands:

```bash
uv run tarmac --help
uv run tarmac download --help
uv run tarmac prepare
```
