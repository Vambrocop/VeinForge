"""Extract real image/mask training tiles from the LeafVeinCNN (Bornean) data.

Each leaf folder has: *_img.png (cleared leaf, dark veins), *_seg.png (hand-traced
vein ground truth, valid only inside the ROI) and *_roi.png (the traced region).
We invert the image so veins are bright (VeinForge convention) and keep only tiles
that lie FULLY inside the ROI, pairing image <-> vein mask.

Run: python scripts/prep_leafveincnn.py --src data/zenodo/slf_full --out data/zenodo_real
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load(p, mode="L"):
    return np.asarray(Image.open(p).convert(mode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/zenodo_real"))
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--min-vein-frac", type=float, default=0.004)
    args = ap.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    n, n_leaves = 0, 0
    for fld in sorted(p for p in args.src.iterdir() if p.is_dir()):
        stem = fld.name
        try:
            img = _load(fld / f"{stem}_img.png")
            seg = _load(fld / f"{stem}_seg.png") > 127
            roi = _load(fld / f"{stem}_roi.png") > 127
        except FileNotFoundError:
            continue
        inv = 255 - img                                  # dark veins -> bright
        ys, xs = np.where(roi)
        if ys.size == 0:
            continue
        n_leaves += 1
        t, s = args.tile, args.stride
        for y in range(int(ys.min()), int(ys.max()) - t + 1, s):
            for x in range(int(xs.min()), int(xs.max()) - t + 1, s):
                if not roi[y:y + t, x:x + t].all():
                    continue                             # tile must be fully inside the ROI
                segt = seg[y:y + t, x:x + t]
                if segt.mean() < args.min_vein_frac:
                    continue                             # skip near-empty tiles
                key = f"{stem}_{y}_{x}.png"
                Image.fromarray(inv[y:y + t, x:x + t].astype("uint8")).save(args.out / "images" / key)
                Image.fromarray((segt * 255).astype("uint8")).save(args.out / "masks" / key)
                n += 1
    print(f"{n_leaves} leaves -> wrote {n} real image/mask tiles to {args.out}")


if __name__ == "__main__":
    main()
