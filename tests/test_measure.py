from veinforge.measure import measure
from veinforge.synthetic import make_vein_phantom

REQUIRED_KEYS = {
    "vein_density", "mean_vein_width_um", "median_vein_width_um",
    "free_ending_count", "free_ending_density", "areole_count",
    "areole_mean_area_um2", "interveinal_distance_um", "vein_area_fraction",
    "total_vein_length_mm", "image_area_mm2",
}


def test_measure_keys_and_density_accuracy():
    _, mask, truth = make_vein_phantom(size=512, pixel_size_um=2.0, width_px=3,
                                       n_longitudinal=8, n_transverse=8)
    out = measure(mask, pixel_size_um=2.0)
    assert REQUIRED_KEYS.issubset(out.keys())
    rel = abs(out["vein_density"] - truth["vein_density"]) / truth["vein_density"]
    assert rel < 0.10                      # density within 10% on the phantom
    assert abs(out["mean_vein_width_um"] - truth["vein_width_um"]) <= 2.0
    # 8x8 grid -> 7x7 = 49 enclosed areoles
    assert out["areole_count"] == 49
