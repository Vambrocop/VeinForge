import numpy as np
from veinforge.leafstress import leaf_features, train_leaf_classifier, predict_leaf


def _leaf(color, n=48, seed=0):
    rng = np.random.default_rng(seed)
    img = np.ones((n, n, 3)) * 0.95                       # white background
    yy, xx = np.ogrid[:n, :n]
    mask = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2) < (n * 0.4) ** 2
    img[mask] = color
    return np.clip(img + 0.03 * rng.standard_normal((n, n, 3)), 0.0, 1.0)


def test_features_differ_healthy_vs_stressed():
    fh = leaf_features(_leaf([0.2, 0.6, 0.2], seed=1))    # green
    fs = leaf_features(_leaf([0.7, 0.6, 0.15], seed=1))   # yellow/brown
    assert fh.shape == fs.shape and not np.allclose(fh, fs)


def test_classifier_separates_health_classes():
    X, y = [], []
    for i in range(15):
        X.append(leaf_features(_leaf([0.2, 0.6, 0.2], seed=i))); y.append("healthy")
        X.append(leaf_features(_leaf([0.7, 0.6, 0.15], seed=i + 100))); y.append("stressed")
    model, m = train_leaf_classifier(np.vstack(X), np.array(y))
    assert m["cv_accuracy_mean"] > 0.8
    assert sorted(m["classes"]) == ["healthy", "stressed"]
    assert predict_leaf(model, np.vstack(X[:2])).shape == (2,)


def test_leaf_features_handles_grayscale():
    g = np.random.default_rng(0).random((32, 32)).astype("float32")
    f_gray = leaf_features(g)
    f_rgb = leaf_features(np.stack([g, g, g], axis=-1))
    assert f_gray.shape == f_rgb.shape and f_gray.ndim == 1
