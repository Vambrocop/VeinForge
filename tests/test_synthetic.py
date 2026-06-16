import numpy as np
from veinforge.synthetic import make_vein_phantom, make_realistic_phantom


def test_phantom_shapes_and_truth():
    image, mask, truth = make_vein_phantom(
        size=512, pixel_size_um=2.0, n_longitudinal=8, n_transverse=8, width_px=3
    )
    assert image.shape == (512, 512)
    assert mask.shape == (512, 512)
    assert mask.dtype == bool
    assert image.dtype in (np.float64, np.float32)
    # veins occupy a small but nonzero fraction
    assert 0.0 < mask.mean() < 0.3
    for key in ("total_length_mm", "vein_width_um", "interveinal_distance_um",
                "n_free_endings", "image_area_mm2", "vein_density"):
        assert key in truth
    # 8 full-length vertical + 8 full-length horizontal lines, grid has no free endings
    assert truth["n_free_endings"] == 0
    assert truth["vein_width_um"] == 3 * 2.0


def test_realistic_phantom_basic():
    img, mask, truth = make_realistic_phantom(size=192, seed=1)
    assert img.shape == (192, 192) and mask.shape == (192, 192)
    assert img.dtype in (np.float64, np.float32) and mask.dtype == bool
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert 0.02 < mask.mean() < 0.4
    assert img[mask].mean() > img[~mask].mean()      # veins brighter than background


def test_realistic_phantom_reproducible():
    _, m1, _ = make_realistic_phantom(size=128, seed=3)
    _, m2, _ = make_realistic_phantom(size=128, seed=3)
    _, m3, _ = make_realistic_phantom(size=128, seed=4)
    assert (m1 == m2).all() and (m1 != m3).any()
