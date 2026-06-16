"""Separate monocot veins into longitudinal vs transverse (P1.5, classical).

Grass leaves (wheat/barley) have longitudinal veins running along the blade plus
transverse (commissural) veins connecting them. We estimate the dominant vein
direction from the skeleton, then classify each vein segment as longitudinal
(near the dominant axis) or transverse (near perpendicular). No training data.
"""
from __future__ import annotations
import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton


def _path_angle(coords: np.ndarray) -> float | None:
    """Principal-axis angle (deg, [0,180)) of a path's pixel coordinates."""
    c = coords - coords.mean(axis=0)
    if len(c) < 2 or np.allclose(c, 0):
        return None
    cov = np.cov(c.T)
    _, vecs = np.linalg.eigh(cov)
    row, col = vecs[:, -1]                          # principal eigenvector (row, col)
    return float(np.degrees(np.arctan2(row, col)) % 180.0)


def _circ_dist(a: float, b: float) -> float:
    """Smallest angular distance on the [0,180) orientation circle."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def separate_orientations(mask: np.ndarray, pixel_size_um: float | None = None,
                          axis_deg: float | None = None, tol_deg: float = 35.0) -> dict:
    """Split a vein mask into longitudinal/transverse classes and their densities."""
    px_mm = (pixel_size_um or 1000.0) / 1000.0
    area_mm2 = mask.size * px_mm * px_mm
    long_mask = np.zeros(mask.shape, dtype=bool)
    trans_mask = np.zeros(mask.shape, dtype=bool)

    skel = skeletonize(mask)
    if skel.sum() < 2:
        return {"axis_deg": float("nan"), "longitudinal_density": 0.0,
                "transverse_density": 0.0, "longitudinal_mask": long_mask,
                "transverse_mask": trans_mask}

    sk = Skeleton(skel.astype(np.uint8), spacing=1)
    lengths_px = sk.path_lengths()
    angles, coords = [], []
    for i in range(sk.n_paths):
        c = sk.path_coordinates(i)
        angles.append(_path_angle(c))
        coords.append(c)

    # Dominant axis = length-weighted circular mean of doubled angles.
    if axis_deg is None:
        vec = sum(L * np.exp(2j * np.radians(a))
                  for a, L in zip(angles, lengths_px) if a is not None)
        axis_deg = float((np.degrees(np.angle(vec)) / 2.0) % 180.0) if vec != 0 else 0.0

    long_len = trans_len = 0.0
    for a, L, c in zip(angles, lengths_px, coords):
        if a is None:
            continue
        rr, cc = c[:, 0].astype(int), c[:, 1].astype(int)
        if _circ_dist(a, axis_deg) <= tol_deg:
            long_len += L
            long_mask[rr, cc] = True
        elif _circ_dist(a, axis_deg + 90.0) <= tol_deg:
            trans_len += L
            trans_mask[rr, cc] = True

    return {
        "axis_deg": axis_deg,
        "longitudinal_density": long_len * px_mm / area_mm2,
        "transverse_density": trans_len * px_mm / area_mm2,
        "longitudinal_mask": long_mask,
        "transverse_mask": trans_mask,
    }
