import numpy as np
import pytest
from veinforge.params import Params
from veinforge.segment.dl import DLSegmenter


def test_dl_segmenter_importable_without_torch():
    # Importing and constructing must not require the heavy DL extra.
    seg = DLSegmenter(model_path=None)
    assert hasattr(seg, "segment")


def test_dl_segmenter_requires_trained_model():
    seg = DLSegmenter(model_path=None)
    with pytest.raises(RuntimeError, match="no trained model"):
        seg.segment(np.zeros((8, 8), dtype=float), Params())


def test_dl_segmenter_missing_checkpoint_path(tmp_path):
    seg = DLSegmenter(model_path=tmp_path / "does_not_exist.pt")
    with pytest.raises(RuntimeError, match="no trained model"):
        seg.segment(np.zeros((8, 8), dtype=float), Params())
