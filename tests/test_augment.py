import numpy as np
from veinforge.augment import augment_pair


def test_augment_preserves_shape_and_binary_mask():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32)).astype("float32")
    mask = (rng.random((32, 32)) > 0.5)
    a, m = augment_pair(img, mask, np.random.default_rng(1))
    assert a.shape == (32, 32) and m.shape == (32, 32)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert set(np.unique(m)).issubset({False, True})


def test_augment_deterministic_with_seed():
    img = np.random.default_rng(0).random((16, 16)).astype("float32")
    mask = np.zeros((16, 16), bool); mask[4:8, :] = True
    a1, m1 = augment_pair(img, mask, np.random.default_rng(5))
    a2, m2 = augment_pair(img, mask, np.random.default_rng(5))
    assert np.array_equal(a1, a2) and np.array_equal(m1, m2)


def test_augment_can_change_image():
    img = np.random.default_rng(0).random((24, 24)).astype("float32")
    mask = np.zeros((24, 24), bool); mask[:, 10:14] = True
    # try several seeds; at least one must alter the image (flip/rot/jitter)
    changed = any(not np.array_equal(augment_pair(img, mask, np.random.default_rng(s))[0], img)
                  for s in range(8))
    assert changed
