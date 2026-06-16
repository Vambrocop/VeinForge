"""Split a tiles folder (images/ + masks/) into train/val BY LEAF (no leakage), optional resize.

Tile filenames are <leaf>_<y>_<x>.png; tiles from the same leaf go to the same split.
Run: python scripts/split_by_leaf.py --src data/leafvein --size 256 --val-frac 0.2
"""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--train", type=Path, default=Path("data/train"))
    ap.add_argument("--val", type=Path, default=Path("data/val"))
    ap.add_argument("--size", type=int, default=0, help="resize tiles to NxN; 0 = keep native")
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    imgs = sorted((args.src / "images").glob("*.png"))
    leaves: dict[str, list[Path]] = {}
    for p in imgs:
        leaves.setdefault(p.name.rsplit("_", 2)[0], []).append(p)
    ids = sorted(leaves)
    step = max(int(round(1 / max(args.val_frac, 1e-6))), 2)
    val_ids = set(ids[::step])

    def save(split_dir: Path, p: Path):
        img = Image.open(p).convert("L")
        m = Image.open(args.src / "masks" / p.name).convert("L")
        if args.size > 0:
            img = img.resize((args.size, args.size), Image.BILINEAR)
            m = m.resize((args.size, args.size), Image.NEAREST)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)
        img.save(split_dir / "images" / p.name)
        m.save(split_dir / "masks" / p.name)

    nt = nv = 0
    for lid, ps in leaves.items():
        for p in ps:
            if lid in val_ids:
                save(args.val, p); nv += 1
            else:
                save(args.train, p); nt += 1
    print(f"train {nt}, val {nv}  (val leaves {len(val_ids)}/{len(ids)})")


if __name__ == "__main__":
    main()
