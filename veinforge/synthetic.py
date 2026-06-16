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


def make_realistic_phantom(size: int = 256, pixel_size_um: float = 2.0,
                           n_longitudinal: int = 7, seed: int = 0, noise: float = 0.05):
    """Monocot-style phantom for DL pretraining: wavy longitudinal veins of varying
    width + transverse connectors, with uneven illumination, staining blotches, noise
    and a slight blur. More realistic than the clean grid. Veins are BRIGHT.

    Returns (image float[0,1], mask bool, truth dict).
    """
    from skimage.filters import gaussian

    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=bool)
    rows = np.arange(size)
    xs = np.linspace(size * 0.08, size * 0.92, n_longitudinal)

    for x0 in xs:                                       # wavy longitudinal veins
        amp = rng.uniform(0.01, 0.05) * size
        freq = rng.uniform(1.0, 3.0) * 2 * np.pi / size
        phase = rng.uniform(0, 2 * np.pi)
        cols = x0 + amp * np.sin(freq * rows + phase)
        half = int(rng.integers(1, 3))                  # varying width
        for r, cc in zip(rows, cols):
            c = int(round(cc))
            mask[r, max(c - half, 0):min(c + half + 1, size)] = True

    for i in range(len(xs) - 1):                        # transverse connectors
        for _ in range(int(rng.integers(3, 8))):
            y = int(rng.integers(2, size - 2))
            c0, c1 = int(round(xs[i])), int(round(xs[i + 1]))
            mask[y:y + 1, min(c0, c1):max(c0, c1) + 1] = True

    illum = np.linspace(rng.uniform(0.2, 0.35), rng.uniform(0.3, 0.45), size)[:, None]
    img = illum + noise * rng.standard_normal((size, size))
    for _ in range(int(rng.integers(2, 6))):            # staining blotches
        yb, xb = rng.integers(0, size, 2)
        rb = int(rng.integers(size // 12, size // 5))
        yy, xx = np.ogrid[:size, :size]
        img[(yy - yb) ** 2 + (xx - xb) ** 2 < rb ** 2] += rng.uniform(-0.1, 0.1)
    img[mask] = 0.8 + noise * rng.standard_normal(int(mask.sum()))
    img = gaussian(np.clip(img, 0.0, 1.0), sigma=float(rng.uniform(0.5, 1.0)))
    img = np.clip(img, 0.0, 1.0)

    truth = {"vein_area_fraction": float(mask.mean()),
             "n_longitudinal": n_longitudinal, "pixel_size_um": pixel_size_um}
    return img, mask, truth
