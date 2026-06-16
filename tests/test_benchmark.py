import numpy as np
from veinforge.benchmark import iou, dice, segmentation_scores


def test_perfect_overlap():
    m = np.zeros((10, 10), bool); m[2:8, 2:8] = True
    assert iou(m, m) == 1.0 and dice(m, m) == 1.0


def test_disjoint():
    a = np.zeros((10, 10), bool); a[0:3, :] = True
    b = np.zeros((10, 10), bool); b[7:10, :] = True
    assert iou(a, b) == 0.0 and dice(a, b) == 0.0


def test_half_overlap_known_values():
    a = np.zeros((10, 10), bool); a[:, 0:4] = True     # 40 px
    b = np.zeros((10, 10), bool); b[:, 2:6] = True     # 40 px, overlap cols 2,3 = 20 px
    # IoU = 20 / (40+40-20) = 20/60 ; Dice = 2*20/(40+40) = 40/80 = 0.5
    assert abs(iou(a, b) - 20 / 60) < 1e-9
    assert abs(dice(a, b) - 0.5) < 1e-9


def test_segmentation_scores_keys():
    m = np.zeros((8, 8), bool); m[1:4, 1:4] = True
    s = segmentation_scores(m, m)
    assert s == {"iou": 1.0, "dice": 1.0}
