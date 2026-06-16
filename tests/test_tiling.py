import numpy as np
from veinforge.tiling import iter_tiles, segment_large
from veinforge.params import Params
from veinforge.segment.classical import ClassicalSegmenter
from veinforge.synthetic import make_vein_phantom

_P = dict(background_radius=0, clahe_clip=0.0, invert=False)


def test_iter_tiles_covers_image():
    covered = np.zeros((100, 120), bool)
    for y0, x0, y1, x1 in iter_tiles((100, 120), tile=50, overlap=10):
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_segment_large_disabled_equals_whole():
    img, _, _ = make_vein_phantom(size=256, width_px=3, noise=0.02)
    seg = ClassicalSegmenter()
    p = Params(tile_size=0, **_P)
    assert np.array_equal(segment_large(img, seg, p), seg.segment(img, p))


def test_segment_large_tiled_similar_to_whole():
    img, _, _ = make_vein_phantom(size=256, width_px=3, noise=0.02)
    seg = ClassicalSegmenter()
    tiled = segment_large(img, seg, Params(tile_size=128, tile_overlap=32, **_P))
    whole = seg.segment(img, Params(**_P))
    iou = np.logical_and(tiled, whole).sum() / np.logical_or(tiled, whole).sum()
    assert iou > 0.6
