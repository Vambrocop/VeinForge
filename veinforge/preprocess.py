from __future__ import annotations
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte, img_as_float
from veinforge.params import Params


def preprocess(image: np.ndarray, params: Params) -> np.ndarray:
    """Return a float[0,1] image where veins are bright, illumination flattened."""
    img = img_as_float(image).astype(np.float64)

    if params.background_radius and params.background_radius > 0:
        # mean-background subtraction approximates rolling-ball flat-fielding
        bg = rank.mean(img_as_ubyte(np.clip(img, 0, 1)),
                       footprint=disk(params.background_radius)).astype(np.float64) / 255.0
        img = np.clip(img - bg + bg.mean(), 0.0, 1.0)

    if params.clahe_clip and params.clahe_clip > 0:
        img = equalize_adapthist(np.clip(img, 0, 1), clip_limit=params.clahe_clip)

    if params.invert:
        img = 1.0 - img

    return np.clip(img, 0.0, 1.0)
