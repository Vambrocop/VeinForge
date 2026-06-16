from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from skan import Skeleton


_NEIGHBORS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def skeleton_metrics(mask: np.ndarray, pixel_size_um: float | None) -> dict:
    """Skeletonize a vein mask and return length/endpoint/branch metrics."""
    skel = skeletonize(mask)
    px_mm = (pixel_size_um or 1000.0) / 1000.0  # default keeps "mm" == "px" when uncalibrated

    total_length_mm = 0.0
    if skel.sum() >= 2:
        try:
            sk = Skeleton(skel.astype(np.uint8), spacing=px_mm)
            total_length_mm = float(np.sum(sk.path_lengths()))
        except Exception:                          # skan can choke on degenerate skeletons
            total_length_mm = float(skel.sum()) * px_mm   # fall back to pixel-count length

    nbr = ndi.convolve(skel.astype(np.uint8), _NEIGHBORS, mode="constant")
    endpoints = skel & (nbr == 1)
    branchpoints = skel & (nbr >= 3)

    return {
        "skeleton": skel.astype(bool),
        "endpoints": endpoints,
        "total_length_mm": total_length_mm,
        "n_endpoints": int(endpoints.sum()),
        "n_branchpoints": int(branchpoints.sum()),
    }
