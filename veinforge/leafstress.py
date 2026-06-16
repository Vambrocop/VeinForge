"""Whole-leaf RGB stress / health classification (parallel track to vein-based stress).

Lightweight & CPU-friendly: extract colour + simple chlorosis features from a leaf
photo, then a RandomForest classifies healthy vs stressed/diseased. Complements the
vein-trait stress phenotyping with an image-level signal (chlorosis/necrosis show in
colour). Works on a folder layout:  <root>/<class>/*.png|jpg  (subfolder = label).
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from skimage.color import rgb2hsv
from skimage.util import img_as_float

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def leaf_features(rgb: np.ndarray) -> np.ndarray:
    """Colour + chlorosis/necrosis features from an RGB leaf image."""
    arr = img_as_float(np.asarray(rgb))[..., :3]
    hsv = rgb2hsv(arr)
    feats: list[float] = []
    for ch in range(3):
        feats += [float(arr[..., ch].mean()), float(arr[..., ch].std())]
    for ch in range(3):
        feats += [float(hsv[..., ch].mean()), float(hsv[..., ch].std())]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    greenness = g - 0.5 * (r + b)
    feats.append(float((greenness < 0.05).mean()))                    # not-green fraction
    feats.append(float(((r > 0.4) & (g > 0.3) & (b < 0.3)).mean()))   # yellow/brown fraction
    return np.array(feats, dtype=float)


def load_folder(root):
    """Load <root>/<class>/*.img -> (X features [n, d], y labels [n])."""
    root = Path(root)
    X, y = [], []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in _EXTS:
                X.append(leaf_features(iio.imread(img_path)))
                y.append(cls_dir.name)
    if not X:
        raise SystemExit(f"no images found under {root}/<class>/")
    return np.vstack(X), np.array(y)


def train_leaf_classifier(X, y, n_estimators: int = 300, cv: int = 5, random_state: int = 0):
    """RandomForest on leaf features; returns (model, metrics with cross-val accuracy)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError as e:
        raise RuntimeError('scikit-learn not installed. Run: pip install -e ".[stress]"') from e
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    if len(set(y)) > 1:
        n_splits = max(2, min(cv, min(Counter(y).values())))
        scores = cross_val_score(model, X, y, cv=n_splits)
    else:
        scores = np.array([np.nan])
    model.fit(X, y)
    return model, {"cv_accuracy_mean": float(np.nanmean(scores)),
                   "cv_accuracy_std": float(np.nanstd(scores)),
                   "n_samples": int(len(y)), "classes": sorted(set(map(str, y)))}


def predict_leaf(model, X) -> np.ndarray:
    return model.predict(X)
