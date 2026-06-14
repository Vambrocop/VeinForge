from veinforge.segment.classical import ClassicalSegmenter

__all__ = ["ClassicalSegmenter", "get_segmenter"]


def get_segmenter(name: str = "classical", model_path=None, **kwargs):
    """Return a segmenter by name. 'classical' needs no training; 'dl' loads a checkpoint."""
    if name == "classical":
        return ClassicalSegmenter()
    if name == "dl":
        from veinforge.segment.dl import DLSegmenter
        return DLSegmenter(model_path=model_path, **kwargs)
    raise ValueError(f"unknown segmenter '{name}' (use 'classical' or 'dl')")
