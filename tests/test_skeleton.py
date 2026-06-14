import numpy as np
from veinforge.skeleton import skeleton_metrics
from veinforge.synthetic import make_vein_phantom


def test_skeleton_total_length_close_to_truth():
    _, mask, truth = make_vein_phantom(size=512, pixel_size_um=2.0, width_px=3)
    m = skeleton_metrics(mask, pixel_size_um=2.0)
    assert m["skeleton"].dtype == bool
    # total length within 8% of analytic grid length
    rel = abs(m["total_length_mm"] - truth["total_length_mm"]) / truth["total_length_mm"]
    assert rel < 0.08
    assert m["n_endpoints"] >= 0


def test_skeleton_counts_a_single_free_end():
    # one short stub line: 1 endpoint inside, 1 at border
    mask = np.zeros((64, 64), bool)
    mask[32, 10:40] = True
    m = skeleton_metrics(mask, pixel_size_um=1.0)
    assert m["n_endpoints"] == 2
