from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage.color import gray2rgb
from skimage.util import img_as_ubyte
from veinforge.params import Params

_SUMMARY_METRICS = ["vein_density", "mean_vein_width_um", "free_ending_density",
                    "areole_count", "interveinal_distance_um"]


def write_csv(rows: list[dict], path: str | Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary(rows: list[dict], path: str | Path) -> None:
    df = pd.DataFrame(rows)
    metrics = [m for m in _SUMMARY_METRICS if m in df.columns]
    agg = df.groupby("sample_id")[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["sample_id"] + [f"{m}_{s}" for m, s in agg.columns[1:]]
    agg.to_csv(path, index=False)


def dump_params(params: Params, path: str | Path) -> None:
    params.to_yaml(path)


def save_overlay(image, mask, skeleton, endpoints, path: str | Path) -> None:
    """RGB overlay: green = mask, red = skeleton, yellow dots = endpoints."""
    rgb = img_as_ubyte(gray2rgb(np.clip(image, 0, 1))).astype(np.float64)
    rgb[mask] = 0.5 * rgb[mask] + np.array([0, 128, 0])
    rgb[skeleton] = np.array([255, 0, 0])
    rgb[endpoints] = np.array([255, 255, 0])
    iio.imwrite(path, np.clip(rgb, 0, 255).astype(np.uint8))
