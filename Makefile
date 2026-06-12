.PHONY: sync download-streetsurfacevis prepare smoke

sync:
	UV_CACHE_DIR=.uv-cache uv sync

download-streetsurfacevis:
	UV_CACHE_DIR=.uv-cache uv run tarmac download streetsurfacevis

prepare:
	UV_CACHE_DIR=.uv-cache uv run tarmac prepare

smoke:
	UV_CACHE_DIR=.uv-cache uv run python scripts/smoke_phase1.py
