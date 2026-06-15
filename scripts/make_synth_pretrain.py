"""Generate a synthetic vein image/mask training set — no downloads, no manual labels.

Used to validate the P2 DL training pipeline end-to-end before any real data exists.
Writes PNG pairs:  <out>/images/*.png  and  <out>/masks/*.png  (same filenames).
Run: python scripts/make_synth_pretrain.py --n 48 --size 128 --out data/pretrain_synth
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from veinforge.synthetic import make_vein_phantom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("data/pretrain_synth"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        img, mask, _ = make_vein_phantom(
            size=args.size,
            n_longitudinal=int(rng.integers(5, 12)),
            n_transverse=int(rng.integers(5, 12)),
            width_px=int(rng.integers(2, 5)),
            noise=float(rng.uniform(0.02, 0.06)),
        )
        name = f"phantom_{i:03d}.png"
        iio.imwrite(args.out / "images" / name, (img * 255).astype("uint8"))
        iio.imwrite(args.out / "masks" / name, (mask * 255).astype("uint8"))
    print(f"wrote {args.n} image/mask pairs -> {args.out}")


if __name__ == "__main__":
    main()
