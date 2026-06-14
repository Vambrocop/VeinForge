"""Train a U-Net vein segmenter for VeinForge P2 (offline).

Requires the DL extra and a folder of image/mask pairs:
    pip install -e ".[dl]"
    python scripts/train_dl.py --data data/train --epochs 50 --out models/unet.pt

Data layout (PNG, same filename in both folders):
    data/train/images/*.png   # grayscale leaf tiles
    data/train/masks/*.png    # vein masks, 0 = background, >127 = vein

Recommended workflow (see docs/p2-roadmap.md):
  1) PRETRAIN on an open dicot vein set that ships hand-traced masks
     (e.g. Zenodo Bornean / LeafVeinCNN) to learn generic "what a vein looks like".
  2) FINE-TUNE on a SMALL set of your own barley/wheat tiles, warm-starting from
     the pretrained checkpoint via --init. Bootstrap the masks cheaply with
     VeinForge's classical output + manual touch-up (e.g. in Ilastik).

The saved checkpoint plugs straight into the pipeline via
veinforge.segment.dl.DLSegmenter(model_path=<checkpoint>).
"""
from __future__ import annotations
import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit('PyTorch not installed. Run: pip install -e ".[dl]"') from exc

import numpy as np
import imageio.v3 as iio
from veinforge.segment.unet import UNet


class VeinPairs(Dataset):
    """Loads (grayscale image, binary mask) tensor pairs from a folder."""

    def __init__(self, root: Path):
        self.images = sorted((Path(root) / "images").glob("*.png"))
        self.masks_dir = Path(root) / "masks"
        if not self.images:
            raise SystemExit(f"No images found under {root}/images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = iio.imread(self.images[i]).astype("float32")
        if img.ndim == 3:
            img = img.mean(-1)
        mask = iio.imread(self.masks_dir / self.images[i].name)
        if mask.ndim == 3:
            mask = mask[..., 0]
        img = torch.from_numpy(img / 255.0)[None]
        mask = torch.from_numpy((mask > 127).astype("float32"))[None]
        return img, mask


def dice_bce_loss(logits, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("models/unet.pt"))
    ap.add_argument("--init", type=Path, default=None,
                    help="warm-start checkpoint (pretrain -> fine-tune)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = UNet().to(device)
    if args.init and args.init.exists():
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"warm-started from {args.init}")

    loader = DataLoader(VeinPairs(args.data), batch_size=args.batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            loss = dice_bce_loss(model(img), mask)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total / len(loader):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}  (load with DLSegmenter(model_path=...))")


if __name__ == "__main__":
    main()
