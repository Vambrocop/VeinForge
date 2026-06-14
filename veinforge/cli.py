from __future__ import annotations
from pathlib import Path
import typer
from veinforge.params import Params
from veinforge.pipeline import process_folder

app = typer.Typer(add_completion=False, help="VeinForge — leaf-vein trait quantification")
stress_app = typer.Typer(help="Stress phenotyping from vein traits (P2-b)")
app.add_typer(stress_app, name="stress")


@app.command()
def run(
    folder: Path = typer.Argument(..., help="Folder of vein tile images"),
    pixel_size_um: float = typer.Option(None, help="Microns per pixel (recommended)"),
    out: Path = typer.Option(Path("results"), help="Output directory"),
    invert: bool = typer.Option(True, help="Invert so veins become bright"),
    background_radius: int = typer.Option(50, help="Rolling-ball radius px; 0 disables"),
    segmenter: str = typer.Option("classical", help="'classical' (no training) or 'dl' (trained)"),
    model: Path = typer.Option(None, help="DL checkpoint path when --segmenter dl"),
):
    """Batch-process a folder and write CSV + SQLite + QC overlays."""
    if pixel_size_um is None:
        typer.echo("WARNING: no --pixel-size-um given; results fall back to pixel units "
                   "unless image calibration is found.")
    from veinforge.segment import get_segmenter
    seg = get_segmenter(segmenter, model_path=model)
    params = Params(pixel_size_um=pixel_size_um, invert=invert, background_radius=background_radius)
    rows = process_folder(folder, params, out, segmenter=seg)
    typer.echo(f"Processed {len(rows)} image(s) with '{segmenter}' segmenter -> {out}")


@app.command()
def view(image: Path = typer.Argument(..., help="One image to inspect in napari")):
    """Open an image with its segmentation overlaid in napari."""
    from veinforge.gui import view as gui_view
    gui_view(image)


@stress_app.command("compare")
def stress_compare(csv: Path = typer.Argument(..., help="results.csv with a group-label column"),
                   label: str = typer.Option("treatment", help="grouping column")):
    """No-training: per-trait group means + significance (control vs stress)."""
    import pandas as pd
    from veinforge.stress import compare_groups
    typer.echo(compare_groups(pd.read_csv(csv), label_col=label).to_string(index=False))


@stress_app.command("train")
def stress_train(csv: Path = typer.Argument(...), label: str = typer.Option("treatment"),
                 out: Path = typer.Option(Path("models/stress_rf.joblib"))):
    """Train a RandomForest stress classifier on vein traits."""
    import pandas as pd
    from veinforge.stress import train_stress_classifier, save_model
    model_obj, metrics = train_stress_classifier(pd.read_csv(csv), label_col=label)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model_obj, out)
    typer.echo(f"cv_accuracy={metrics['cv_accuracy_mean']:.3f}+/-{metrics['cv_accuracy_std']:.3f} -> {out}")


@stress_app.command("predict")
def stress_predict(csv: Path = typer.Argument(...), model: Path = typer.Option(...),
                   out: Path = typer.Option(Path("predictions.csv"))):
    """Predict stress labels for new trait rows."""
    import pandas as pd
    from veinforge.stress import predict_stress, load_model
    df = pd.read_csv(csv)
    df["predicted"] = predict_stress(load_model(model), df)
    df.to_csv(out, index=False)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
