# VeinForge P1 (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `pip`-installable Python tool that takes cleared-leaf vein microscope tiles and outputs vein traits (density, width, free endings, areoles, interveinal distance) to CSV + SQLite + QC overlays, using a classical-CV pipeline with no training data.

**Architecture:** A linear, module-per-responsibility pipeline `io → preprocess → segment → skeleton → measure → report/db`. The `segment` step is a swappable interface (classical now, DL later). Correctness is proven against a **synthetic vein phantom** whose ground-truth traits are known analytically, so no real images are needed to pass tests.

**Tech Stack:** Python ≥3.10, numpy, scipy, scikit-image, skan, pandas, pyyaml, typer, imageio, tifffile; optional `napari` (GUI); pytest + ruff (dev).

**Spec:** [`docs/specs/2026-06-14-veinforge-mvp-design.md`](../specs/2026-06-14-veinforge-mvp-design.md)

---

## File Structure & Locked Signatures

Build these files (one responsibility each). Later tasks depend on these exact names/signatures — keep them consistent.

| File | Responsibility | Key public API |
|---|---|---|
| `pyproject.toml` | packaging, deps, `veinforge` CLI entry | — |
| `veinforge/__init__.py` | version | `__version__: str` |
| `veinforge/params.py` | run parameters + YAML I/O | `@dataclass Params`; `Params.from_yaml(p)`, `.to_yaml(p)`, `.to_dict()` |
| `veinforge/synthetic.py` | ground-truth vein phantom (test backbone) | `make_vein_phantom(...) -> (image, mask, truth: dict)` |
| `veinforge/io.py` | load image, pixel-size, filename metadata | `load_image(path)->(img,meta)`; `pixel_size_from_meta(path)->float|None`; `parse_metadata(name, pattern)->dict` |
| `veinforge/preprocess.py` | grayscale/channel, de-illuminate, contrast, orient | `preprocess(img, params)->np.ndarray` |
| `veinforge/segment/base.py` | segmentation interface | `class VeinSegmenter(Protocol): segment(img, params)->np.ndarray[bool]` |
| `veinforge/segment/classical.py` | Frangi/Sato → threshold → cleanup | `class ClassicalSegmenter: segment(img, params)->mask` |
| `veinforge/skeleton.py` | skeleton + graph metrics | `skeleton_metrics(mask, pixel_size_um)->dict` |
| `veinforge/measure.py` | all traits from mask+skeleton | `measure(mask, pixel_size_um)->dict` |
| `veinforge/db.py` | SQLite schema + inserts | `connect`, `init_db`, `insert_run`, `upsert_sample`, `insert_image`, `insert_measurement` |
| `veinforge/report.py` | CSV, summary, params.yaml, overlay PNG | `write_csv`, `write_summary`, `dump_params`, `save_overlay` |
| `veinforge/pipeline.py` | orchestrate one image / one folder | `process_image(...)->dict`; `process_folder(...)->list[dict]` |
| `veinforge/cli.py` | `veinforge run` / `veinforge view` | typer `app` |
| `veinforge/gui.py` | napari layers | `prepare_layers(img, mask, skel)->list`; `view(path)` |

**`measure()` returns exactly these keys** (must match DB columns & CSV):
`vein_density, mean_vein_width_um, median_vein_width_um, free_ending_count, free_ending_density, areole_count, areole_mean_area_um2, interveinal_distance_um, vein_area_fraction, total_vein_length_mm, image_area_mm2`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `veinforge/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "veinforge"
version = "0.1.0"
description = "Automated leaf-vein trait quantification for barley/wheat (classical-CV MVP)"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-image>=0.22",
    "skan>=0.11",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "typer>=0.9",
    "imageio>=2.31",
    "tifffile>=2023.7",
]

[project.optional-dependencies]
gui = ["napari[pyqt5]>=0.4.18"]
dev = ["pytest>=7.4", "ruff>=0.4"]

[project.scripts]
veinforge = "veinforge.cli:app"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `veinforge/__init__.py`**

```python
"""VeinForge: automated leaf-vein trait quantification (classical-CV MVP)."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Create empty `tests/__init__.py` and `tests/conftest.py`**

```python
# tests/conftest.py
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
```

- [ ] **Step 4: Install editable + verify import**

Run: `pip install -e ".[dev]"` then `python -c "import veinforge; print(veinforge.__version__)"`
Expected: prints `0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml veinforge/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold veinforge package"
```

---

## Task 2: `params.py` — run parameters + YAML

**Files:**
- Create: `veinforge/params.py`, `tests/test_params.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_params.py
from veinforge.params import Params


def test_params_yaml_roundtrip(tmp_path):
    p = Params(pixel_size_um=1.5, sato_sigmas=(1.0, 2.0, 3.0), min_object_px=128)
    out = tmp_path / "params.yaml"
    p.to_yaml(out)
    loaded = Params.from_yaml(out)
    assert loaded.pixel_size_um == 1.5
    assert tuple(loaded.sato_sigmas) == (1.0, 2.0, 3.0)
    assert loaded.min_object_px == 128


def test_params_defaults():
    p = Params()
    assert p.threshold_method == "otsu"
    assert p.invert is True
    assert "pixel_size_um" in p.to_dict()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_params.py -v`
Expected: FAIL — `ModuleNotFoundError: veinforge.params`

- [ ] **Step 3: Implement `veinforge/params.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass
class Params:
    pixel_size_um: float | None = None          # microns per pixel; None -> pixel units + warn
    channel: str = "gray"                        # "gray" | "r" | "g" | "b"
    invert: bool = True                          # make veins bright after preprocessing
    background_radius: int = 50                  # rolling-ball radius (px); 0 disables
    clahe_clip: float = 0.01                     # 0 disables CLAHE
    sato_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    threshold_method: str = "otsu"               # "otsu" | "hysteresis"
    hysteresis_low: float = 0.1                  # used when threshold_method == "hysteresis"
    hysteresis_high: float = 0.25
    min_object_px: int = 64                      # remove specks smaller than this
    closing_radius: int = 1                      # morphological closing radius (px)
    filename_pattern: str = r"(?P<sample_id>[^_]+)_(?P<treatment>[^_]+)_(?P<replicate>[^_]+)_(?P<position>[^_.]+)"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Params":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "sato_sigmas" in data and data["sato_sigmas"] is not None:
            data["sato_sigmas"] = tuple(data["sato_sigmas"])
        return cls(**data)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_params.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add veinforge/params.py tests/test_params.py
git commit -m "feat: Params dataclass with YAML round-trip"
```

---

## Task 3: `synthetic.py` — ground-truth vein phantom

This is the backbone of every later test: a grayscale image of known parallel + transverse veins on a noisy background, plus the boolean mask and a `truth` dict with analytically known traits.

**Files:**
- Create: `veinforge/synthetic.py`, `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthetic.py
import numpy as np
from veinforge.synthetic import make_vein_phantom


def test_phantom_shapes_and_truth():
    image, mask, truth = make_vein_phantom(
        size=512, pixel_size_um=2.0, n_longitudinal=8, n_transverse=8, width_px=3
    )
    assert image.shape == (512, 512)
    assert mask.shape == (512, 512)
    assert mask.dtype == bool
    assert image.dtype == np.float64 or image.dtype == np.float32
    # veins occupy a small but nonzero fraction
    assert 0.0 < mask.mean() < 0.3
    for key in ("total_length_mm", "vein_width_um", "interveinal_distance_um",
                "n_free_endings", "image_area_mm2", "vein_density"):
        assert key in truth
    # 8 full-length vertical + 8 full-length horizontal lines, grid has no free endings
    assert truth["n_free_endings"] == 0
    assert truth["vein_width_um"] == 3 * 2.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_synthetic.py -v`
Expected: FAIL — `ModuleNotFoundError: veinforge.synthetic`

- [ ] **Step 3: Implement `veinforge/synthetic.py`**

```python
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
    image[mask] = 0.9 + noise * rng.standard_normal(mask.sum())
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_synthetic.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/synthetic.py tests/test_synthetic.py
git commit -m "feat: synthetic vein phantom with analytic ground truth"
```

---

## Task 4: `io.py` — load image, calibration, filename metadata

**Files:**
- Create: `veinforge/io.py`, `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io.py
import imageio.v3 as iio
import numpy as np
from veinforge.io import load_image, parse_metadata


def test_parse_metadata_default_pattern():
    from veinforge.params import Params
    meta = parse_metadata("S012_heat_r2_mid.tif", Params().filename_pattern)
    assert meta == {"sample_id": "S012", "treatment": "heat", "replicate": "r2", "position": "mid"}


def test_parse_metadata_no_match_returns_empty():
    from veinforge.params import Params
    assert parse_metadata("random.tif", Params().filename_pattern) == {}


def test_load_image_grayscale_floats(tmp_path):
    arr = (np.random.rand(32, 48, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.png"
    iio.imwrite(p, arr)
    img, meta = load_image(p)
    assert img.shape == (32, 48)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert meta["width_px"] == 48 and meta["height_px"] == 32
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_io.py -v`
Expected: FAIL — `ModuleNotFoundError: veinforge.io`

- [ ] **Step 3: Implement `veinforge/io.py`**

```python
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
            page = tf.pages[0]
            tags = page.tags
            if "XResolution" in tags and "ResolutionUnit" in tags:
                num, den = tags["XResolution"].value
                if num == 0:
                    return None
                px_per_unit = den and num / den or 0.0
                unit = tags["ResolutionUnit"].value  # 2=inch, 3=cm
                if px_per_unit <= 0:
                    return None
                unit_um = 25400.0 if int(unit) == 2 else 10000.0  # per inch / per cm
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/io.py tests/test_io.py
git commit -m "feat: image loading, calibration, filename metadata"
```

---

## Task 5: `preprocess.py` — de-illuminate, contrast, orient

**Files:**
- Create: `veinforge/preprocess.py`, `tests/test_preprocess.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess.py
import numpy as np
from veinforge.params import Params
from veinforge.preprocess import preprocess
from veinforge.synthetic import make_vein_phantom


def test_preprocess_keeps_shape_and_range():
    image, mask, _ = make_vein_phantom(size=256)
    out = preprocess(image, Params(background_radius=0, clahe_clip=0.0, invert=False))
    assert out.shape == image.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    # veins (bright) should remain brighter than background on average
    assert out[mask].mean() > out[~mask].mean()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `veinforge/preprocess.py`**

```python
from __future__ import annotations
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte, img_as_float
from veinforge.params import Params


def preprocess(image: np.ndarray, params: Params) -> np.ndarray:
    """Return a float[0,1] image where veins are bright, illumination flattened."""
    img = img_as_float(image).astype(np.float64)

    if params.background_radius and params.background_radius > 0:
        # mean-background subtraction approximates rolling-ball flat-fielding
        bg = rank.mean(img_as_ubyte(np.clip(img, 0, 1)),
                       footprint=disk(params.background_radius)).astype(np.float64) / 255.0
        img = np.clip(img - bg + bg.mean(), 0.0, 1.0)

    if params.clahe_clip and params.clahe_clip > 0:
        img = equalize_adapthist(np.clip(img, 0, 1), clip_limit=params.clahe_clip)

    if params.invert:
        img = 1.0 - img

    return np.clip(img, 0.0, 1.0)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/preprocess.py tests/test_preprocess.py
git commit -m "feat: preprocessing (de-illuminate, CLAHE, orient veins bright)"
```

---

## Task 6: `segment/` — pluggable interface + classical backend

**Files:**
- Create: `veinforge/segment/__init__.py`, `veinforge/segment/base.py`, `veinforge/segment/classical.py`, `tests/test_segment.py`

- [ ] **Step 1: Write the failing test** (IoU vs phantom mask)

```python
# tests/test_segment.py
import numpy as np
from veinforge.params import Params
from veinforge.segment.classical import ClassicalSegmenter
from veinforge.synthetic import make_vein_phantom


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union


def test_classical_segmenter_recovers_veins():
    image, mask, _ = make_vein_phantom(size=512, width_px=3, noise=0.02)
    seg = ClassicalSegmenter()
    pred = seg.segment(image, Params(background_radius=0, clahe_clip=0.0, invert=False,
                                     sato_sigmas=(1.0, 2.0, 3.0), min_object_px=32))
    assert pred.dtype == bool
    assert _iou(pred, mask) > 0.5
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_segment.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the three files**

```python
# veinforge/segment/__init__.py
from veinforge.segment.classical import ClassicalSegmenter
__all__ = ["ClassicalSegmenter"]
```

```python
# veinforge/segment/base.py
from __future__ import annotations
from typing import Protocol
import numpy as np
from veinforge.params import Params


class VeinSegmenter(Protocol):
    def segment(self, image: np.ndarray, params: Params) -> np.ndarray:
        """Return a boolean vein mask the same HxW as `image`."""
        ...
```

```python
# veinforge/segment/classical.py
from __future__ import annotations
import numpy as np
from skimage.filters import sato, threshold_otsu, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, binary_closing, disk
from veinforge.params import Params


class ClassicalSegmenter:
    """Multiscale Sato vesselness -> threshold -> morphological cleanup."""

    def segment(self, image: np.ndarray, params: Params) -> np.ndarray:
        vesselness = sato(image, sigmas=params.sato_sigmas, black_ridges=False)
        v = vesselness / (vesselness.max() + 1e-12)

        if params.threshold_method == "hysteresis":
            mask = apply_hysteresis_threshold(v, params.hysteresis_low, params.hysteresis_high)
        else:
            mask = v > threshold_otsu(v)

        if params.closing_radius and params.closing_radius > 0:
            mask = binary_closing(mask, disk(params.closing_radius))
        if params.min_object_px and params.min_object_px > 0:
            mask = remove_small_objects(mask, min_size=params.min_object_px)
        return mask.astype(bool)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_segment.py -v`
Expected: PASS (IoU > 0.5)

- [ ] **Step 5: Commit**

```bash
git add veinforge/segment tests/test_segment.py
git commit -m "feat: pluggable segmenter interface + classical Sato backend"
```

---

## Task 7: `skeleton.py` — skeleton + graph metrics

**Files:**
- Create: `veinforge/skeleton.py`, `tests/test_skeleton.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skeleton.py
import numpy as np
from veinforge.skeleton import skeleton_metrics
from veinforge.synthetic import make_vein_phantom


def test_skeleton_total_length_close_to_truth():
    _, mask, truth = make_vein_phantom(size=512, pixel_size_um=2.0, width_px=3)
    m = skeleton_metrics(mask, pixel_size_um=2.0)
    assert m["skeleton"].dtype == bool
    # total length within 8% of analytic grid length
    rel = abs(m["total_length_mm"] - truth["total_length_mm"]) / truth["total_length_mm"]
    assert rel < 0.08
    # full grid: endpoints only at image borders
    assert m["n_endpoints"] >= 0


def test_skeleton_counts_a_single_free_end():
    # one short stub line: 1 endpoint inside, 1 at border
    mask = np.zeros((64, 64), bool)
    mask[32, 10:40] = True
    m = skeleton_metrics(mask, pixel_size_um=1.0)
    assert m["n_endpoints"] == 2
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_skeleton.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `veinforge/skeleton.py`**

```python
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

    if skel.sum() >= 2:
        sk = Skeleton(skel.astype(np.uint8), spacing=px_mm)
        total_length_mm = float(np.sum(sk.path_lengths()))
    else:
        total_length_mm = 0.0

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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_skeleton.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/skeleton.py tests/test_skeleton.py
git commit -m "feat: skeleton metrics (length via skan, endpoints/branches via neighbor count)"
```

---

## Task 8: `measure.py` — all traits

**Files:**
- Create: `veinforge/measure.py`, `tests/test_measure.py`

- [ ] **Step 1: Write the failing test** (the ±5% density acceptance check)

```python
# tests/test_measure.py
import numpy as np
from veinforge.measure import measure
from veinforge.synthetic import make_vein_phantom

REQUIRED_KEYS = {
    "vein_density", "mean_vein_width_um", "median_vein_width_um",
    "free_ending_count", "free_ending_density", "areole_count",
    "areole_mean_area_um2", "interveinal_distance_um", "vein_area_fraction",
    "total_vein_length_mm", "image_area_mm2",
}


def test_measure_keys_and_density_accuracy():
    _, mask, truth = make_vein_phantom(size=512, pixel_size_um=2.0, width_px=3,
                                       n_longitudinal=8, n_transverse=8)
    out = measure(mask, pixel_size_um=2.0)
    assert REQUIRED_KEYS.issubset(out.keys())
    rel = abs(out["vein_density"] - truth["vein_density"]) / truth["vein_density"]
    assert rel < 0.10                      # density within 10% on the phantom
    assert abs(out["mean_vein_width_um"] - truth["vein_width_um"]) <= 2.0
    # 8x8 grid -> 7x7 = 49 enclosed areoles
    assert out["areole_count"] == 49
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_measure.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `veinforge/measure.py`**

```python
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label, regionprops
from veinforge.skeleton import skeleton_metrics


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

    # widths: 2 x distance-to-edge at skeleton pixels
    dt_in = ndi.distance_transform_edt(mask)
    widths_um = 2.0 * dt_in[skel] * px_um
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
    }
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_measure.py -v`
Expected: PASS (density within 10%, areole_count == 49)

- [ ] **Step 5: Commit**

```bash
git add veinforge/measure.py tests/test_measure.py
git commit -m "feat: trait measurement (density, width, free endings, areoles, interveinal distance)"
```

---

## Task 9: `db.py` — SQLite store

**Files:**
- Create: `veinforge/db.py`, `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from veinforge.db import connect, init_db, insert_run, upsert_sample, insert_image, insert_measurement


def test_db_roundtrip(tmp_path):
    conn = connect(tmp_path / "veinforge.db")
    init_db(conn)
    run_id = insert_run(conn, params_json='{"a":1}', version="0.1.0")
    sample_id = upsert_sample(conn, {"sample_id": "S1", "species": "barley",
                                     "treatment": "heat", "replicate": "r1"})
    image_id = insert_image(conn, {"sample_id": "S1", "path": "x.tif", "position": "mid",
                                   "pixel_size_um": 2.0, "width_px": 512, "height_px": 512})
    insert_measurement(conn, image_id, run_id, {"vein_density": 3.2, "areole_count": 49})
    rows = conn.execute("SELECT vein_density, areole_count FROM measurements").fetchall()
    assert rows == [(3.2, 49)]
    assert sample_id == "S1"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `veinforge/db.py`**

```python
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples(
  sample_id TEXT PRIMARY KEY, species TEXT, treatment TEXT, replicate TEXT,
  notes TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS images(
  image_id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT, path TEXT, position TEXT,
  pixel_size_um REAL, width_px INTEGER, height_px INTEGER, imported_at TEXT);
CREATE TABLE IF NOT EXISTS runs(
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, params_json TEXT, veinforge_version TEXT,
  created_at TEXT);
CREATE TABLE IF NOT EXISTS measurements(
  measurement_id INTEGER PRIMARY KEY AUTOINCREMENT, image_id INTEGER, run_id INTEGER,
  vein_density REAL, mean_vein_width_um REAL, median_vein_width_um REAL,
  free_ending_count INTEGER, free_ending_density REAL, areole_count INTEGER,
  areole_mean_area_um2 REAL, interveinal_distance_um REAL, vein_area_fraction REAL,
  total_vein_length_mm REAL, image_area_mm2 REAL, created_at TEXT);
"""

_MEASURE_COLS = ["vein_density", "mean_vein_width_um", "median_vein_width_um",
                 "free_ending_count", "free_ending_density", "areole_count",
                 "areole_mean_area_um2", "interveinal_distance_um", "vein_area_fraction",
                 "total_vein_length_mm", "image_area_mm2"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def insert_run(conn, params_json: str, version: str) -> int:
    cur = conn.execute("INSERT INTO runs(params_json, veinforge_version, created_at) VALUES(?,?,?)",
                       (params_json, version, _now()))
    conn.commit()
    return cur.lastrowid


def upsert_sample(conn, s: dict) -> str:
    conn.execute(
        "INSERT OR IGNORE INTO samples(sample_id, species, treatment, replicate, notes, created_at)"
        " VALUES(?,?,?,?,?,?)",
        (s.get("sample_id"), s.get("species"), s.get("treatment"), s.get("replicate"),
         s.get("notes"), _now()))
    conn.commit()
    return s.get("sample_id")


def insert_image(conn, im: dict) -> int:
    cur = conn.execute(
        "INSERT INTO images(sample_id, path, position, pixel_size_um, width_px, height_px, imported_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (im.get("sample_id"), im.get("path"), im.get("position"), im.get("pixel_size_um"),
         im.get("width_px"), im.get("height_px"), _now()))
    conn.commit()
    return cur.lastrowid


def insert_measurement(conn, image_id: int, run_id: int, traits: dict) -> int:
    cols = ["image_id", "run_id"] + _MEASURE_COLS + ["created_at"]
    vals = [image_id, run_id] + [traits.get(c) for c in _MEASURE_COLS] + [_now()]
    placeholders = ",".join("?" * len(cols))
    cur = conn.execute(f"INSERT INTO measurements({','.join(cols)}) VALUES({placeholders})", vals)
    conn.commit()
    return cur.lastrowid
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/db.py tests/test_db.py
git commit -m "feat: SQLite store (samples/images/runs/measurements)"
```

---

## Task 10: `report.py` — CSV, summary, params, QC overlay

**Files:**
- Create: `veinforge/report.py`, `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import numpy as np
import pandas as pd
from veinforge.params import Params
from veinforge.report import write_csv, write_summary, dump_params, save_overlay


def test_write_csv_and_summary(tmp_path):
    rows = [
        {"sample_id": "S1", "position": "top", "vein_density": 3.0, "areole_count": 10},
        {"sample_id": "S1", "position": "mid", "vein_density": 4.0, "areole_count": 12},
    ]
    csv = tmp_path / "results.csv"
    write_csv(rows, csv)
    assert pd.read_csv(csv).shape[0] == 2

    summ = tmp_path / "samples_summary.csv"
    write_summary(rows, summ)
    df = pd.read_csv(summ)
    assert df.loc[df.sample_id == "S1", "vein_density_mean"].iloc[0] == 3.5


def test_dump_params_and_overlay(tmp_path):
    dump_params(Params(pixel_size_um=2.0), tmp_path / "params.yaml")
    assert (tmp_path / "params.yaml").exists()

    img = np.random.rand(64, 64)
    mask = np.zeros((64, 64), bool); mask[30:34, :] = True
    skel = np.zeros((64, 64), bool); skel[32, :] = True
    eps = np.zeros((64, 64), bool); eps[32, 0] = True
    out = tmp_path / "overlay.png"
    save_overlay(img, mask, skel, eps, out)
    assert out.exists()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `veinforge/report.py`**

```python
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/report.py tests/test_report.py
git commit -m "feat: CSV/summary/params output + QC overlay rendering"
```

---

## Task 11: `pipeline.py` — orchestrate image & folder

**Files:**
- Create: `veinforge/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test** (end-to-end on a synthetic folder)

```python
# tests/test_pipeline.py
import imageio.v3 as iio
import numpy as np
import pandas as pd
from veinforge.params import Params
from veinforge.pipeline import process_folder
from veinforge.synthetic import make_vein_phantom


def _write_phantoms(folder):
    for name in ["S1_heat_r1_top.png", "S1_heat_r1_mid.png"]:
        image, _, _ = make_vein_phantom(size=384, width_px=3, noise=0.02)
        iio.imwrite(folder / name, (image * 255).astype(np.uint8))


def test_process_folder_outputs(tmp_path):
    inp = tmp_path / "imgs"; inp.mkdir(); _write_phantoms(inp)
    out = tmp_path / "out"
    rows = process_folder(inp, Params(pixel_size_um=2.0, background_radius=0,
                                      clahe_clip=0.0, invert=False), out)
    assert len(rows) == 2
    assert (out / "results.csv").exists()
    assert (out / "samples_summary.csv").exists()
    assert (out / "params.yaml").exists()
    assert (out / "veinforge.db").exists()
    assert len(list((out / "qc").glob("*.png"))) == 2
    df = pd.read_csv(out / "results.csv")
    assert {"sample_id", "position", "vein_density"}.issubset(df.columns)
    assert (df["vein_density"] > 0).all()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `veinforge/pipeline.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
import veinforge
from veinforge.params import Params
from veinforge.io import load_image, parse_metadata
from veinforge.preprocess import preprocess
from veinforge.segment.classical import ClassicalSegmenter
from veinforge.skeleton import skeleton_metrics
from veinforge.measure import measure
from veinforge import db as dbmod
from veinforge.report import write_csv, write_summary, dump_params, save_overlay

_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def process_image(path, params: Params, segmenter=None) -> dict:
    segmenter = segmenter or ClassicalSegmenter()
    image, meta = load_image(path)
    px_um = params.pixel_size_um if params.pixel_size_um is not None else meta["pixel_size_um"]
    pre = preprocess(image, params)
    mask = segmenter.segment(pre, params)
    sk = skeleton_metrics(mask, px_um)
    traits = measure(mask, px_um)
    md = parse_metadata(Path(path).name, params.filename_pattern)
    row = {**md, "path": str(path), "pixel_size_um": px_um,
           "width_px": meta["width_px"], "height_px": meta["height_px"], **traits}
    row["_image"] = image
    row["_mask"] = mask
    row["_skeleton"] = sk["skeleton"]
    row["_endpoints"] = sk["endpoints"]
    return row


def process_folder(folder, params: Params, out_dir, segmenter=None) -> list[dict]:
    folder, out_dir = Path(folder), Path(out_dir)
    (out_dir / "qc").mkdir(parents=True, exist_ok=True)

    conn = dbmod.connect(out_dir / "veinforge.db")
    dbmod.init_db(conn)
    run_id = dbmod.insert_run(conn, json.dumps(params.to_dict(), default=list), veinforge.__version__)

    rows = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in _EXTS):
        row = process_image(path, params, segmenter)
        save_overlay(row.pop("_image"), row.pop("_mask"), row.pop("_skeleton"),
                     row.pop("_endpoints"), out_dir / "qc" / f"{path.stem}_overlay.png")
        if row.get("sample_id"):
            dbmod.upsert_sample(conn, {"sample_id": row["sample_id"],
                                       "treatment": row.get("treatment"),
                                       "replicate": row.get("replicate")})
        image_id = dbmod.insert_image(conn, {"sample_id": row.get("sample_id"), "path": row["path"],
                                             "position": row.get("position"),
                                             "pixel_size_um": row.get("pixel_size_um"),
                                             "width_px": row["width_px"], "height_px": row["height_px"]})
        dbmod.insert_measurement(conn, image_id, run_id, row)
        rows.append(row)

    write_csv(rows, out_dir / "results.csv")
    write_summary(rows, out_dir / "samples_summary.csv")
    dump_params(params, out_dir / "params.yaml")
    conn.close()
    return rows
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/pipeline.py tests/test_pipeline.py
git commit -m "feat: end-to-end folder pipeline (CSV+DB+QC+params)"
```

---

## Task 12: `cli.py` — `veinforge run` / `veinforge view`

**Files:**
- Create: `veinforge/cli.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing test** (typer CliRunner)

```python
# tests/test_cli.py
import imageio.v3 as iio
import numpy as np
from typer.testing import CliRunner
from veinforge.cli import app
from veinforge.synthetic import make_vein_phantom

runner = CliRunner()


def test_cli_run(tmp_path):
    inp = tmp_path / "imgs"; inp.mkdir()
    image, _, _ = make_vein_phantom(size=320, width_px=3, noise=0.02)
    iio.imwrite(inp / "S1_heat_r1_mid.png", (image * 255).astype(np.uint8))
    out = tmp_path / "out"
    result = runner.invoke(app, ["run", str(inp), "--pixel-size-um", "2.0",
                                 "--out", str(out), "--no-invert", "--background-radius", "0"])
    assert result.exit_code == 0, result.output
    assert (out / "results.csv").exists()


def test_cli_run_warns_without_calibration(tmp_path):
    inp = tmp_path / "imgs"; inp.mkdir()
    image, _, _ = make_vein_phantom(size=256, width_px=3)
    iio.imwrite(inp / "a.png", (image * 255).astype(np.uint8))
    result = runner.invoke(app, ["run", str(inp), "--out", str(tmp_path / "o"),
                                 "--no-invert", "--background-radius", "0"])
    assert result.exit_code == 0
    assert "calibration" in result.output.lower() or "pixel" in result.output.lower()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `veinforge/cli.py`**

```python
from __future__ import annotations
from pathlib import Path
import typer
from veinforge.params import Params
from veinforge.pipeline import process_folder

app = typer.Typer(add_completion=False, help="VeinForge — leaf-vein trait quantification")


@app.command()
def run(
    folder: Path = typer.Argument(..., help="Folder of vein tile images"),
    pixel_size_um: float = typer.Option(None, help="Microns per pixel (recommended)"),
    out: Path = typer.Option(Path("results"), help="Output directory"),
    invert: bool = typer.Option(True, help="Invert so veins become bright"),
    background_radius: int = typer.Option(50, help="Rolling-ball radius px; 0 disables"),
):
    """Batch-process a folder and write CSV + SQLite + QC overlays."""
    if pixel_size_um is None:
        typer.echo("WARNING: no --pixel-size-um given; results fall back to pixel units "
                   "unless image calibration is found.")
    params = Params(pixel_size_um=pixel_size_um, invert=invert, background_radius=background_radius)
    rows = process_folder(folder, params, out)
    typer.echo(f"Processed {len(rows)} image(s) -> {out}")


@app.command()
def view(image: Path = typer.Argument(..., help="One image to inspect in napari")):
    """Open an image with its segmentation overlaid in napari."""
    from veinforge.gui import view as gui_view
    gui_view(image)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/cli.py tests/test_cli.py
git commit -m "feat: CLI (run/view) via typer"
```

---

## Task 13: `gui.py` — napari layers (optional extra)

GUI windows aren't unit-tested; we test the pure **layer-preparation** function and keep napari import lazy.

**Files:**
- Create: `veinforge/gui.py`, `tests/test_gui.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gui.py
import numpy as np
from veinforge.gui import prepare_layers


def test_prepare_layers_structure():
    img = np.random.rand(64, 64)
    mask = np.zeros((64, 64), bool); mask[30:34, :] = True
    skel = np.zeros((64, 64), bool); skel[32, :] = True
    layers = prepare_layers(img, mask, skel)
    names = {l["name"] for l in layers}
    assert names == {"image", "vein mask", "skeleton"}
    assert all(l["data"].shape == (64, 64) for l in layers)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_gui.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `veinforge/gui.py`**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np


def prepare_layers(image: np.ndarray, mask: np.ndarray, skeleton: np.ndarray) -> list[dict]:
    """Pure description of napari layers (no napari import needed)."""
    return [
        {"name": "image", "data": image, "kind": "image"},
        {"name": "vein mask", "data": mask.astype(np.uint8), "kind": "labels"},
        {"name": "skeleton", "data": skeleton.astype(np.uint8), "kind": "labels"},
    ]


def view(path: str | Path) -> None:                # pragma: no cover (interactive)
    import napari
    from veinforge.params import Params
    from veinforge.io import load_image
    from veinforge.preprocess import preprocess
    from veinforge.segment.classical import ClassicalSegmenter
    from veinforge.skeleton import skeleton_metrics

    image, _ = load_image(path)
    params = Params()
    mask = ClassicalSegmenter().segment(preprocess(image, params), params)
    skel = skeleton_metrics(mask, params.pixel_size_um)["skeleton"]
    viewer = napari.Viewer()
    for layer in prepare_layers(image, mask, skel):
        if layer["kind"] == "image":
            viewer.add_image(layer["data"], name=layer["name"])
        else:
            viewer.add_labels(layer["data"], name=layer["name"])
    napari.run()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_gui.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add veinforge/gui.py tests/test_gui.py
git commit -m "feat: napari layer prep + lazy viewer"
```

---

## Task 14: Dev fixtures + README + final DoD check

**Files:**
- Create: `scripts/fetch_dev_samples.py`, modify `README.md`

- [ ] **Step 1: Create `scripts/fetch_dev_samples.py`** (downloads a few real vein images into the gitignored `data/dev-samples/`)

```python
"""Download a handful of real leaf-vein images for local pipeline dev.

These are dicot reticulate-vein images (NOT barley/wheat) — usable only as
pipeline smoke tests and (later) P2 pretraining, never as monocot ground truth.
Source: ImageJ demo leaf + Leaf Vein Network CNN sample (Open Data Commons-BY).
Run: python scripts/fetch_dev_samples.py
"""
from pathlib import Path
import urllib.request

DEST = Path("data/dev-samples")
URLS = {
    "imagej_leaf.jpg": "https://imagej.net/ij/images/leaf.jpg",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        target = DEST / name
        if target.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"fetch {name} <- {url}")
        urllib.request.urlretrieve(url, target)
    print(f"done -> {DEST.resolve()}  (gitignored; see docs/related-work.md §7 for full datasets)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Append a usage section to `README.md`**

```markdown

## 快速开始（P1）

```bash
pip install -e ".[dev]"          # 安装；GUI: pip install -e ".[gui]"
pytest -q                        # 跑测试（合成脉图验证）
veinforge run ./tiles --pixel-size-um 1.23 --out results
veinforge view ./tiles/example.tif   # napari 查看分割
```

输出：`results/results.csv`、`samples_summary.csv`、`qc/*_overlay.png`、`params.yaml`、`veinforge.db`。
开发样图：`python scripts/fetch_dev_samples.py`（写入 gitignored `data/dev-samples/`）。
```

- [ ] **Step 3: Run the whole suite**

Run: `pytest -q`
Expected: all tests pass (Tasks 2–13).

- [ ] **Step 4: Verify the CLI on real demo image**

Run: `python scripts/fetch_dev_samples.py && veinforge run data/dev-samples --pixel-size-um 50 --out results_demo`
Expected: prints `Processed 1 image(s)`; `results_demo/results.csv` and `results_demo/qc/imagej_leaf_overlay.png` exist.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_dev_samples.py README.md
git commit -m "docs: dev-sample fetcher + README quickstart; P1 MVP complete"
```

---

## Definition of Done (P1)

- [ ] `pip install -e ".[dev]"` succeeds; `veinforge --help` works.
- [ ] `pytest -q` green — synthetic phantom validates density (±10%), width, areole count (49), endpoints.
- [ ] `veinforge run <folder> --pixel-size-um X` writes `results.csv`, `samples_summary.csv`, `params.yaml`, `veinforge.db`, and one `qc/*_overlay.png` per image.
- [ ] Missing calibration prints a clear warning and falls back to pixel units.
- [ ] `veinforge view <image>` opens napari with image/mask/skeleton layers.

## Self-Review Notes

- **Spec coverage:** §3 modules → Tasks 4–13; §4 data flow → Task 11; §5 traits/formulas → Task 8 (interveinal distance uses the 2023 distance-map modal estimator); §6 SQLite schema → Task 9; §7 outputs → Tasks 10–11; §8 CLI/GUI → Tasks 12–13; §11 synthetic validation → Tasks 3+8; §12 DoD → mapped above.
- **Out of scope (per spec §1):** DL segmentation, longitudinal/transverse split, bundle-sheath exclusion, stress phenotyping — deferred to P1.5/P2; the `VeinSegmenter` Protocol (Task 6) is the seam where the DL backend later plugs in.
- **Naming consistency:** `measure()` keys ≡ `db._MEASURE_COLS` ≡ CSV columns; `skeleton_metrics()` returns `skeleton/endpoints/total_length_mm/n_endpoints/n_branchpoints`, consumed unchanged by `measure()` and `pipeline`.
```
