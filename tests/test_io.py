import imageio.v3 as iio
import numpy as np
from veinforge.io import load_image, parse_metadata
from veinforge.params import Params


def test_parse_metadata_default_pattern():
    meta = parse_metadata("S012_heat_r2_mid.tif", Params().filename_pattern)
    assert meta == {"sample_id": "S012", "treatment": "heat", "replicate": "r2", "position": "mid"}


def test_parse_metadata_no_match_returns_empty():
    assert parse_metadata("random.tif", Params().filename_pattern) == {}


def test_load_image_grayscale_floats(tmp_path):
    arr = (np.random.rand(32, 48, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.png"
    iio.imwrite(p, arr)
    img, meta = load_image(p)
    assert img.shape == (32, 48)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert meta["width_px"] == 48 and meta["height_px"] == 32
