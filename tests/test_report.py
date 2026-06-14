import numpy as np
import pandas as pd
from veinforge.params import Params
from veinforge.report import write_csv, write_summary, dump_params, save_overlay


def test_write_csv_and_summary(tmp_path):
    rows = [
        {"sample_id": "S1", "position": "top", "vein_density": 3.0, "areole_count": 10},
        {"sample_id": "S1", "position": "mid", "vein_density": 4.0, "areole_count": 12},
    ]
    csv = tmp_path / "results.csv"
    write_csv(rows, csv)
    assert pd.read_csv(csv).shape[0] == 2

    summ = tmp_path / "samples_summary.csv"
    write_summary(rows, summ)
    df = pd.read_csv(summ)
    assert df.loc[df.sample_id == "S1", "vein_density_mean"].iloc[0] == 3.5


def test_dump_params_and_overlay(tmp_path):
    dump_params(Params(pixel_size_um=2.0), tmp_path / "params.yaml")
    assert (tmp_path / "params.yaml").exists()

    img = np.random.rand(64, 64)
    mask = np.zeros((64, 64), bool); mask[30:34, :] = True
    skel = np.zeros((64, 64), bool); skel[32, :] = True
    eps = np.zeros((64, 64), bool); eps[32, 0] = True
    out = tmp_path / "overlay.png"
    save_overlay(img, mask, skel, eps, out)
    assert out.exists()
