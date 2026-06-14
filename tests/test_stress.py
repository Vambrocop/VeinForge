import numpy as np
import pandas as pd
from typer.testing import CliRunner
from veinforge.cli import app
from veinforge.stress import compare_groups, train_stress_classifier, predict_stress


def _synthetic(n=40, seed=0):
    rng = np.random.default_rng(seed)
    ctrl = pd.DataFrame({
        "treatment": "control",
        "vein_density": rng.normal(3.0, 0.3, n),
        "mean_vein_width_um": rng.normal(20, 2, n),
        "free_ending_density": rng.normal(5, 1, n),
        "areole_count": rng.normal(40, 5, n),
        "areole_mean_area_um2": rng.normal(8000, 800, n),
        "interveinal_distance_um": rng.normal(120, 10, n),
        "vein_area_fraction": rng.normal(0.12, 0.02, n),
    })
    heat = ctrl.copy()
    heat["treatment"] = "heat"
    heat["vein_density"] = rng.normal(4.2, 0.3, n)            # stress -> denser veins
    heat["interveinal_distance_um"] = rng.normal(95, 10, n)
    return pd.concat([ctrl, heat], ignore_index=True)


def test_compare_groups_flags_difference():
    res = compare_groups(_synthetic(), label_col="treatment")
    vd = res[res.trait == "vein_density"].iloc[0]
    assert vd["p_value"] < 0.01
    assert vd["difference"] > 0                                # heat denser than control


def test_train_and_predict_separates_groups():
    df = _synthetic()
    model, metrics = train_stress_classifier(df, label_col="treatment")
    assert metrics["cv_accuracy_mean"] > 0.8
    assert len(predict_stress(model, df.head(3))) == 3


def test_cli_stress_compare(tmp_path):
    csv = tmp_path / "r.csv"
    _synthetic(n=20).to_csv(csv, index=False)
    result = CliRunner().invoke(app, ["stress", "compare", str(csv)])
    assert result.exit_code == 0, result.output
    assert "vein_density" in result.output
