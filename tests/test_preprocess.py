from veinforge.params import Params
from veinforge.preprocess import preprocess
from veinforge.synthetic import make_vein_phantom


def test_preprocess_keeps_shape_and_range():
    image, mask, _ = make_vein_phantom(size=256)
    out = preprocess(image, Params(background_radius=0, clahe_clip=0.0, invert=False))
    assert out.shape == image.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    # veins (bright) should remain brighter than background on average
    assert out[mask].mean() > out[~mask].mean()


def test_preprocess_background_subtraction_runs():
    image, mask, _ = make_vein_phantom(size=128)
    out = preprocess(image, Params(background_radius=10, clahe_clip=0.0, invert=False))
    assert out.shape == image.shape and out.min() >= 0.0 and out.max() <= 1.0
    assert out[mask].mean() > out[~mask].mean()
