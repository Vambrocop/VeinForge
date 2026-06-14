import numpy as np
from veinforge.params import Params
from veinforge.segment.classical import ClassicalSegmenter
from veinforge.synthetic import make_vein_phantom


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union


def test_classical_segmenter_recovers_veins():
    image, mask, _ = make_vein_phantom(size=512, width_px=3, noise=0.02)
    seg = ClassicalSegmenter()
    pred = seg.segment(image, Params(background_radius=0, clahe_clip=0.0, invert=False,
                                     sato_sigmas=(1.0, 2.0, 3.0), min_object_px=32))
    assert pred.dtype == bool
    assert _iou(pred, mask) > 0.5
