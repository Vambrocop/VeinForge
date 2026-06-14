from __future__ import annotations
import numpy as np


def make_vein_phantom(
    size: int = 512,
    pixel_size_um: float = 2.0,
    n_longitudinal: int = 8,
    n_transverse: int = 8,
    width_px: int = 3,
    noise: float = 0.03,
):
    """Grid of evenly spaced bright veins on a darker noisy background.

    Returns (image float in [0,1], mask bool, truth dict).
    Veins are BRIGHT (high intensity) to mimic a preprocessed/inverted image.
    """
    mask = np.zeros((size, size), dtype=bool)
    half = width_px // 2

    def positions(n):
        # evenly spaced, leaving a margin so lines span the full image
        return np.linspace(size / (n + 1), size - size / (n + 1), n).round().astype(int)

    col_pos = positions(n_longitudinal)
    row_pos = positions(n_transverse)
    for c in col_pos:
        mask[:, max(c - half, 0):c + half + 1] = True
    for r in row_pos:
        mask[max(r - half, 0):r + half + 1, :] = True

    rng = np.random.default_rng(0)
    image = 0.25 + noise * rng.standard_normal((size, size))
    image[mask] = 0.9 + noise * rng.standard_normal(int(mask.sum()))
    image = np.clip(image, 0.0, 1.0)

    # Ground truth (skeleton length = one centerline per line, full image span)
    total_len_px = (n_longitudinal + n_transverse) * size
    px_mm = pixel_size_um / 1000.0
    image_area_mm2 = (size * px_mm) ** 2
    total_length_mm = total_len_px * px_mm
    truth = {
        "total_length_mm": total_length_mm,
        "vein_width_um": width_px * pixel_size_um,
        "interveinal_distance_um": float(np.diff(col_pos).mean()) * pixel_size_um,
        "n_free_endings": 0,                       # full grid -> all endpoints touch border
        "image_area_mm2": image_area_mm2,
        "vein_density": total_length_mm / image_area_mm2,
        "pixel_size_um": pixel_size_um,
    }
    return image, mask, truth
