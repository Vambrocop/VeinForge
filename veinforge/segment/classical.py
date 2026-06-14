from __future__ import annotations
import numpy as np
from skimage.filters import sato, threshold_otsu, apply_hysteresis_threshold
from skimage.measure import label
from skimage.morphology import closing, disk
from veinforge.params import Params


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than `min_size` px (version-stable)."""
    if not min_size or min_size <= 0:
        return mask
    lbl = label(mask, connectivity=1)
    counts = np.bincount(lbl.ravel())
    too_small = counts < min_size
    too_small[0] = False  # never drop background
    return mask & ~too_small[lbl]


class ClassicalSegmenter:
    """Multiscale Sato vesselness -> threshold -> morphological cleanup."""

    def segment(self, image: np.ndarray, params: Params) -> np.ndarray:
        vesselness = sato(image, sigmas=params.sato_sigmas, black_ridges=False)
        v = vesselness / (vesselness.max() + 1e-12)

        if params.threshold_method == "hysteresis":
            mask = apply_hysteresis_threshold(v, params.hysteresis_low, params.hysteresis_high)
        else:
            mask = v > threshold_otsu(v)
        mask = np.asarray(mask, dtype=bool)

        if params.closing_radius and params.closing_radius > 0:
            mask = closing(mask, disk(params.closing_radius)).astype(bool)
        return _remove_small(mask, params.min_object_px).astype(bool)
