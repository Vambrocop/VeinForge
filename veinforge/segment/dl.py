"""Deep-learning vein segmenter (P2) — drop-in replacement for ClassicalSegmenter.

Satisfies the same VeinSegmenter interface (`segment(image, params) -> bool mask`),
so the existing pipeline uses it unchanged. A trained checkpoint is produced offline
by scripts/train_dl.py; see docs/p2-roadmap.md. torch is imported lazily so importing
this module never requires the (heavy) DL extra.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from veinforge.params import Params


class DLSegmenter:
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu",
                 threshold: float = 0.5):
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.threshold = threshold
        self._model = None

    def _load(self):
        if self.model_path is None or not self.model_path.exists():
            raise RuntimeError(
                "DLSegmenter has no trained model. Train one with scripts/train_dl.py "
                "(see docs/p2-roadmap.md), then pass model_path=<checkpoint>."
            )
        try:
            import torch
            from veinforge.segment.unet import UNet
        except ImportError as e:
            raise RuntimeError(
                'PyTorch is not installed. Install the DL extra: pip install -e ".[dl]"'
            ) from e
        model = UNet()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device).eval()
        self._model = model

    def segment(self, image: np.ndarray, params: Params) -> np.ndarray:
        if self._model is None:
            self._load()
        import torch
        with torch.no_grad():
            x = torch.from_numpy(image.astype("float32"))[None, None].to(self.device)
            prob = torch.sigmoid(self._model(x))[0, 0].cpu().numpy()
        return prob > self.threshold
