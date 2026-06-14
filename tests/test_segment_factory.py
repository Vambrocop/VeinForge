import pytest
from veinforge.segment import get_segmenter, ClassicalSegmenter
from veinforge.segment.dl import DLSegmenter


def test_get_classical_no_training():
    assert isinstance(get_segmenter("classical"), ClassicalSegmenter)


def test_get_dl_trained():
    assert isinstance(get_segmenter("dl", model_path=None), DLSegmenter)


def test_unknown_segmenter():
    with pytest.raises(ValueError, match="unknown segmenter"):
        get_segmenter("magic")
