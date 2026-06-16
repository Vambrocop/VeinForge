"""Geometry + photometric augmentation for vein image/mask training pairs (numpy).

Pure numpy so it is testable without the DL extra; used by scripts/train_dl.py.
"""
from __future__ import annotations
import numpy as np


def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Random flips/rotations (image+mask) + brightness/contrast jitter (image only)."""
    img, m = image, mask
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]           # horizontal flip
    if rng.random() < 0.5:
        img, m = img[::-1, :], m[::-1, :]           # vertical flip
    k = int(rng.integers(0, 4))
    if k:
        img, m = np.rot90(img, k), np.rot90(m, k)
    img = np.clip(img * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0.0, 1.0)
    return np.ascontiguousarray(img, dtype=image.dtype), np.ascontiguousarray(m, dtype=mask.dtype)
