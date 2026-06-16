from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label, regionprops
from veinforge.skeleton import skeleton_metrics
from veinforge.orient import separate_orientations


def _interveinal_distance_um(mask: np.ndarray, pixel_size_um: float) -> float:
    """Distance-map modal estimator (How dense can you be?, Appl. Plant Sci. 2023).

    Mode of the log distribution of background-to-nearest-vein distances; the
    full interveinal distance is ~2x that typical half-spacing.
    """
    dt = ndi.distance_transform_edt(~mask)
    vals = dt[~mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    logv = np.log(vals)
    counts, edges = np.histogram(logv, bins=64)
    mode_log = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
    return float(2.0 * np.exp(mode_log) * pixel_size_um)


def measure(mask: np.ndarray, pixel_size_um: float | None) -> dict:
    """Compute the full P1 trait set from a boolean vein mask."""
    px_um = pixel_size_um or 1.0
    px_mm = px_um / 1000.0
    h, w = mask.shape
    image_area_mm2 = (h * px_mm) * (w * px_mm)

    sk = skeleton_metrics(mask, pixel_size_um)
    skel = sk["skeleton"]

    # widths from the medial-axis distance. EDT measures center-to-nearest-
    # background-pixel, so it overshoots the true edge (which lies half a pixel
    # beyond the last foreground pixel) by 1px total: width = (2*EDT - 1).
    dt_in = ndi.distance_transform_edt(mask)
    widths_um = np.clip(2.0 * dt_in[skel] - 1.0, 0.0, None) * px_um
    mean_w = float(widths_um.mean()) if widths_um.size else 0.0
    median_w = float(np.median(widths_um)) if widths_um.size else 0.0

    # free endings: skeleton endpoints not on the image border
    eps = sk["endpoints"]
    border = np.zeros_like(eps)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    free_endings = eps & ~border
    free_ending_count = int(free_endings.sum())

    # areoles: background components fully enclosed (not touching border)
    bg_labels = label(~mask)
    areole_areas_px = []
    for r in regionprops(bg_labels):
        minr, minc, maxr, maxc = r.bbox
        if minr == 0 or minc == 0 or maxr == h or maxc == w:
            continue                       # touches border -> not an enclosed areole
        areole_areas_px.append(r.area)
    areole_count = len(areole_areas_px)
    areole_mean_area_um2 = float(np.mean(areole_areas_px) * px_um * px_um) if areole_areas_px else 0.0

    total_length_mm = sk["total_length_mm"]
    vein_density = total_length_mm / image_area_mm2 if image_area_mm2 else 0.0

    ori = separate_orientations(mask, px_um)        # monocot longitudinal vs transverse

    return {
        "vein_density": vein_density,
        "mean_vein_width_um": mean_w,
        "median_vein_width_um": median_w,
        "free_ending_count": free_ending_count,
        "free_ending_density": free_ending_count / image_area_mm2 if image_area_mm2 else 0.0,
        "areole_count": areole_count,
        "areole_mean_area_um2": areole_mean_area_um2,
        "interveinal_distance_um": _interveinal_distance_um(mask, px_um),
        "vein_area_fraction": float(mask.mean()),
        "total_vein_length_mm": total_length_mm,
        "image_area_mm2": image_area_mm2,
        "longitudinal_density": ori["longitudinal_density"],
        "transverse_density": ori["transverse_density"],
        "vein_axis_deg": ori["axis_deg"],
    }
