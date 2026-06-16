import numpy as np
from veinforge.gui import prepare_layers, measure_from_mask


def test_prepare_layers_structure():
    img = np.random.rand(64, 64)
    mask = np.zeros((64, 64), bool); mask[30:34, :] = True
    skel = np.zeros((64, 64), bool); skel[32, :] = True
    layers = prepare_layers(img, mask, skel)
    names = {layer["name"] for layer in layers}
    assert names == {"image", "vein mask", "skeleton"}
    assert all(layer["data"].shape == (64, 64) for layer in layers)


def test_measure_from_mask_returns_traits():
    mask = np.zeros((64, 64), bool)
    mask[30:33, :] = True; mask[:, 30:33] = True       # a simple cross of veins
    out = measure_from_mask(mask, pixel_size_um=2.0)
    assert "vein_density" in out and out["vein_density"] > 0
