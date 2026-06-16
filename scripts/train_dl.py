"""Train a U-Net vein segmenter for VeinForge P2 (offline).

Requires the DL extra and a folder of image/mask pairs:
    pip install -e ".[dl]"
    python scripts/train_dl.py --data data/train --val data/val --epochs 50 --out models/unet.pt

Data layout (PNG, same filename in both folders):
    <data>/images/*.png   # grayscale leaf tiles
    <data>/masks/*.png    # vein masks, 0 = background, >127 = vein

Improvements (P2 "DL rigor"):
  - data augmentation (flips/rotations + brightness jitter) on the training set
  - clDice connectivity-aware loss (BCE + Dice + clDice) so thin veins stay connected
  - a FIXED holdout via --val: report val IoU each epoch and keep the BEST checkpoint
    (so progress across runs is actually comparable)

Recommended workflow (see docs/p2-roadmap.md): pretrain on open dicot data, then
fine-tune on your barley/wheat with --init <pretrained.pt>. The checkpoint plugs
into the pipeline via veinforge.segment.dl.DLSegmenter(model_path=...).
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
from veinforge.augment import augment_pair
from veinforge.losses import dice_bce_cldice_loss


class VeinPairs(Dataset):
    """Loads (grayscale image, binary mask) tensor pairs; optional augmentation."""

    def __init__(self, root, augment: bool = False):
        self.images = sorted((Path(root) / "images").glob("*.png"))
        self.masks_dir = Path(root) / "masks"
        self.augment = augment
        if not self.images:
            raise SystemExit(f"No images found under {root}/images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = iio.imread(self.images[i]).astype("float32")
        if img.ndim == 3:
            img = img.mean(-1)
        img = img / 255.0
        mask = iio.imread(self.masks_dir / self.images[i].name)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 127).astype("float32")
        if self.augment:
            img, mask = augment_pair(img, mask, np.random.default_rng())
        return torch.from_numpy(img)[None], torch.from_numpy(mask)[None]


@torch.no_grad()
def _val_iou(model, val_dir, device) -> float:
    ds = VeinPairs(val_dir, augment=False)
    model.eval()
    ious = []
    for i in range(len(ds)):
        img, mask = ds[i]
        pred = (torch.sigmoid(model(img[None].to(device)))[0, 0] > 0.5).cpu().numpy()
        gt = mask[0].numpy() > 0.5
        ious.append(np.logical_and(pred, gt).sum() / max(np.logical_or(pred, gt).sum(), 1))
    return float(np.mean(ious)) if ious else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None, help="FIXED holdout dir for IoU tracking")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cldice", type=float, default=0.5, help="clDice connectivity weight")
    ap.add_argument("--no-augment", action="store_true", help="disable data augmentation")
    ap.add_argument("--out", type=Path, default=Path("models/unet.pt"))
    ap.add_argument("--init", type=Path, default=None, help="warm-start checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  augment={not args.no_augment}  cldice={args.cldice}")
    model = UNet().to(device)
    if args.init and args.init.exists():
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"warm-started from {args.init}")

    loader = DataLoader(VeinPairs(args.data, augment=not args.no_augment),
                        batch_size=args.batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            loss = dice_bce_cldice_loss(model(img), mask, cldice_weight=args.cldice)
            loss.backward()
            opt.step()
            total += loss.item()
        msg = f"epoch {epoch + 1}/{args.epochs}  loss={total / len(loader):.4f}"
        if args.val:
            iou = _val_iou(model, args.val, device)
            msg += f"  val_iou={iou:.3f}"
            if iou > best:
                best = iou
                torch.save(model.state_dict(), args.out)   # keep best-by-val
        print(msg)

    if not args.val or not args.out.exists():          # always leave a model file
        torch.save(model.state_dict(), args.out)
    tag = f"  (best val_iou={best:.3f})" if args.val else ""
    print(f"saved -> {args.out}{tag}  (load with DLSegmenter(model_path=...))")


if __name__ == "__main__":
    main()
