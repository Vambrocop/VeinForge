import numpy as np
from veinforge.orient import separate_orientations
from veinforge.synthetic import make_vein_phantom


def test_separates_longitudinal_and_transverse():
    # 8 vertical (longitudinal) + 5 horizontal (transverse) veins
    _, mask, _ = make_vein_phantom(size=512, pixel_size_um=2.0,
                                   n_longitudinal=8, n_transverse=5, width_px=3)
    r = separate_orientations(mask, pixel_size_um=2.0)
    assert {"axis_deg", "longitudinal_density", "transverse_density",
            "longitudinal_mask", "transverse_mask"}.issubset(r)
    # more longitudinal (8) than transverse (5)
    assert r["longitudinal_density"] > r["transverse_density"]
    ratio = r["longitudinal_density"] / r["transverse_density"]
    assert 1.2 < ratio < 2.1                       # roughly 8:5
    assert r["longitudinal_mask"].sum() > 0 and r["transverse_mask"].sum() > 0


def test_axis_detected_vertical():
    _, mask, _ = make_vein_phantom(size=384, n_longitudinal=10, n_transverse=3, width_px=3)
    r = separate_orientations(mask, pixel_size_um=1.0)
    # vertical lines dominate -> axis near 90 degrees
    d = abs(r["axis_deg"] - 90.0) % 180.0
    assert min(d, 180.0 - d) < 20.0


def test_blank_mask_safe():
    r = separate_orientations(np.zeros((64, 64), bool), pixel_size_um=1.0)
    assert r["longitudinal_density"] == 0.0 and r["transverse_density"] == 0.0
