"""Full-resolution tiling — segment large images without downsizing (keeps fine veins).

`veinforge run --tile-size 512` cuts a big microscope image into overlapping tiles,
segments each at native resolution, and unions the masks. This removes the 128px
downsize compromise that loses thin veins.
"""
from __future__ import annotations
import numpy as np


def iter_tiles(shape, tile: int, overlap: int):
    """Yield (y0, x0, y1, x1) boxes of size <=tile covering the image with overlap."""
    h, w = shape[:2]
    step = max(tile - overlap, 1)

    def starts(n):
        s = list(range(0, max(n - tile, 0) + 1, step)) or [0]
        if s[-1] != max(n - tile, 0):
            s.append(max(n - tile, 0))
        return s

    for y in starts(h):
        for x in starts(w):
            yield y, x, min(y + tile, h), min(x + tile, w)


def segment_large(image: np.ndarray, segmenter, params) -> np.ndarray:
    """Segment the whole image, or in full-res tiles (union) when params.tile_size>0."""
    t = getattr(params, "tile_size", 0) or 0
    if t <= 0 or (image.shape[0] <= t and image.shape[1] <= t):
        return segmenter.segment(image, params)
    overlap = getattr(params, "tile_overlap", 32)
    mask = np.zeros(image.shape[:2], dtype=bool)
    for y0, x0, y1, x1 in iter_tiles(image.shape, t, overlap):
        mask[y0:y1, x0:x1] |= segmenter.segment(image[y0:y1, x0:x1], params)
    return mask
