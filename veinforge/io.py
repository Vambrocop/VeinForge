from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from skimage.color import rgb2gray
from skimage.util import img_as_float


def parse_metadata(filename: str, pattern: str) -> dict:
    """Extract sample metadata from a filename using a named-group regex."""
    m = re.search(pattern, Path(filename).name)
    return m.groupdict() if m else {}


def pixel_size_from_meta(path: str | Path) -> float | None:
    """Read microns-per-pixel from TIFF resolution tags if present, else None."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            tags = tf.pages[0].tags
            if "XResolution" in tags and "ResolutionUnit" in tags:
                num, den = tags["XResolution"].value
                if num == 0 or den == 0:
                    return None
                px_per_unit = num / den
                if px_per_unit <= 0:
                    return None
                unit = int(tags["ResolutionUnit"].value)  # 2=inch, 3=cm
                unit_um = 25400.0 if unit == 2 else 10000.0
                return unit_um / px_per_unit
    except Exception:
        return None
    return None


def load_image(path: str | Path):
    """Load an image as float grayscale in [0,1] plus a metadata dict."""
    raw = iio.imread(path)
    if raw.ndim == 3:
        gray = rgb2gray(raw[..., :3])
    else:
        gray = img_as_float(raw)
    gray = np.clip(img_as_float(gray), 0.0, 1.0)
    meta = {
        "path": str(path),
        "height_px": int(gray.shape[0]),
        "width_px": int(gray.shape[1]),
        "pixel_size_um": pixel_size_from_meta(path),
    }
    return gray, meta
