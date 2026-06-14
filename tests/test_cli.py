import imageio.v3 as iio
import numpy as np
from typer.testing import CliRunner
from veinforge.cli import app
from veinforge.synthetic import make_vein_phantom

runner = CliRunner()


def test_cli_run(tmp_path):
    inp = tmp_path / "imgs"; inp.mkdir()
    image, _, _ = make_vein_phantom(size=320, width_px=3, noise=0.02)
    iio.imwrite(inp / "S1_heat_r1_mid.png", (image * 255).astype(np.uint8))
    out = tmp_path / "out"
    result = runner.invoke(app, ["run", str(inp), "--pixel-size-um", "2.0",
                                 "--out", str(out), "--no-invert", "--background-radius", "0"])
    assert result.exit_code == 0, result.output
    assert (out / "results.csv").exists()


def test_cli_run_warns_without_calibration(tmp_path):
    inp = tmp_path / "imgs"; inp.mkdir()
    image, _, _ = make_vein_phantom(size=256, width_px=3)
    iio.imwrite(inp / "a.png", (image * 255).astype(np.uint8))
    result = runner.invoke(app, ["run", str(inp), "--out", str(tmp_path / "o"),
                                 "--no-invert", "--background-radius", "0"])
    assert result.exit_code == 0
    assert "calibration" in result.output.lower() or "pixel" in result.output.lower()
