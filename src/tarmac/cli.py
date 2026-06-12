from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tarmac.datasets.streetsurfacevis import download_streetsurfacevis
from tarmac.datasets.unify import build_manifest

app = typer.Typer(no_args_is_help=True)
download_app = typer.Typer(no_args_is_help=True)
app.add_typer(download_app, name="download")
console = Console()


@download_app.command("streetsurfacevis")
def download_streetsurfacevis_cmd(
    output_dir: Path = typer.Option(
        Path("data/raw/streetsurfacevis"),
        "--output-dir",
        "-o",
        help="Directory for the StreetSurfaceVis raw files.",
    ),
) -> None:
    """Download StreetSurfaceVis v1.0 from Zenodo."""
    result = download_streetsurfacevis(output_dir)
    console.print(
        f"StreetSurfaceVis ready: {result.image_count} images, CSV at {result.csv_path}"
    )


@app.command()
def prepare(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Raw dataset root."),
    output: Path = typer.Option(
        Path("data/processed/manifest.parquet"), help="Manifest output path."
    ),
) -> None:
    """Build the unified parquet manifest."""
    manifest = build_manifest(raw_dir=raw_dir, output_path=output)
    console.print(f"Manifest written to {manifest.path} ({manifest.row_count} rows)")
    console.print(manifest.stats.to_string(index=False))


def _stub(command: str) -> None:
    console.print(f"{command}: stub for a later phase.")


@app.command()
def embed() -> None:
    _stub("embed")


@app.command()
def cluster() -> None:
    _stub("cluster")


@app.command()
def train() -> None:
    _stub("train")


@app.command()
def evaluate() -> None:
    _stub("evaluate")


@app.command()
def analyze() -> None:
    _stub("analyze")


@app.command()
def report() -> None:
    _stub("report")


@app.command()
def ui() -> None:
    _stub("ui")


if __name__ == "__main__":
    app()
