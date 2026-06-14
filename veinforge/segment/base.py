from __future__ import annotations
from typing import Protocol
import numpy as np
from veinforge.params import Params


class VeinSegmenter(Protocol):
    def segment(self, image: np.ndarray, params: Params) -> np.ndarray:
        """Return a boolean vein mask the same HxW as `image`."""
        ...
