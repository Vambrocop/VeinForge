from __future__ import annotations
from pathlib import Path
import typer
from veinforge.params import Params
from veinforge.pipeline import process_folder

app = typer.Typer(add_completion=False, help="VeinForge — leaf-vein trait quantification")


@app.command()
def run(
    folder: Path = typer.Argument(..., help="Folder of vein tile images"),
    pixel_size_um: float = typer.Option(None, help="Microns per pixel (recommended)"),
    out: Path = typer.Option(Path("results"), help="Output directory"),
    invert: bool = typer.Option(True, help="Invert so veins become bright"),
    background_radius: int = typer.Option(50, help="Rolling-ball radius px; 0 disables"),
):
    """Batch-process a folder and write CSV + SQLite + QC overlays."""
    if pixel_size_um is None:
        typer.echo("WARNING: no --pixel-size-um given; results fall back to pixel units "
                   "unless image calibration is found.")
    params = Params(pixel_size_um=pixel_size_um, invert=invert, background_radius=background_radius)
    rows = process_folder(folder, params, out)
    typer.echo(f"Processed {len(rows)} image(s) -> {out}")


@app.command()
def view(image: Path = typer.Argument(..., help="One image to inspect in napari")):
    """Open an image with its segmentation overlaid in napari."""
    from veinforge.gui import view as gui_view
    gui_view(image)


if __name__ == "__main__":
    app()
