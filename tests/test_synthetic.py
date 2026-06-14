import numpy as np
from veinforge.synthetic import make_vein_phantom


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
