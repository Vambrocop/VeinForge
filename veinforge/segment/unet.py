"""Small U-Net for binary vein segmentation (P2).

Imported lazily (only when torch is available) by both the training script
(scripts/train_dl.py) and the inference plug (veinforge/segment/dl.py), so the
two share one architecture definition.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4-level U-Net, 1 input channel (grayscale) -> 1 logit channel (vein/not)."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.e1, self.e2, self.e3 = _block(1, base), _block(base, base * 2), _block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bott = _block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(b), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)
