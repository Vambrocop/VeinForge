import imageio.v3 as iio
import numpy as np
import pandas as pd
from veinforge.params import Params
from veinforge.pipeline import process_folder
from veinforge.synthetic import make_vein_phantom


def _write_phantoms(folder):
    for name in ["S1_heat_r1_top.png", "S1_heat_r1_mid.png"]:
        image, _, _ = make_vein_phantom(size=384, width_px=3, noise=0.02)
        iio.imwrite(folder / name, (image * 255).astype(np.uint8))


def test_process_folder_outputs(tmp_path):
    inp = tmp_path / "imgs"; inp.mkdir(); _write_phantoms(inp)
    out = tmp_path / "out"
    rows = process_folder(inp, Params(pixel_size_um=2.0, background_radius=0,
                                      clahe_clip=0.0, invert=False), out)
    assert len(rows) == 2
    assert (out / "results.csv").exists()
    assert (out / "samples_summary.csv").exists()
    assert (out / "params.yaml").exists()
    assert (out / "veinforge.db").exists()
    assert len(list((out / "qc").glob("*.png"))) == 2
    df = pd.read_csv(out / "results.csv")
    assert {"sample_id", "position", "vein_density"}.issubset(df.columns)
    assert (df["vein_density"] > 0).all()
