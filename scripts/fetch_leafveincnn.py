"""Download the FULL LeafVeinCNN (Bornean) dataset from Zenodo and prep image/mask tiles.

Meant for a free Colab/Kaggle GPU (large downloads + GPU); NOT for the small local CPU.
Downloads each plot, extracts _img/_seg/_roi, preps ROI tiles via prep_leafveincnn.py,
then deletes the big zip (disk hygiene). Produces data/leafvein/{images,masks}.

Run (on Colab):  python scripts/fetch_leafveincnn.py            # all plots
                 python scripts/fetch_leafveincnn.py --plots SLF BNTa   # a subset
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO = "https://zenodo.org/api/records/4008614/files/{plot}-downsampled_images.zip/content"
ALL_PLOTS = ["SLF", "BNTa", "DAF2", "ESAa", "BNTb", "ESAb", "SER", "BSO", "BEL", "DAS1"]
_WANT = ("_img.png", "_seg.png", "_roi.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", nargs="+", default=ALL_PLOTS)
    ap.add_argument("--out", type=Path, default=Path("data/leafvein"))
    ap.add_argument("--raw", type=Path, default=Path("data/leafvein_raw"))
    args = ap.parse_args()

    for plot in args.plots:
        zp = args.raw / f"{plot}.zip"
        zp.parent.mkdir(parents=True, exist_ok=True)
        if not zp.exists():
            print(f"downloading {plot} ...", flush=True)
            urllib.request.urlretrieve(ZENODO.format(plot=plot), zp)
        with zipfile.ZipFile(zp) as zf:
            members = [n for n in zf.namelist() if n.endswith(_WANT)]
            zf.extractall(args.raw / plot, members=members)
        subprocess.run([sys.executable, "scripts/prep_leafveincnn.py",
                        "--src", str(args.raw / plot), "--out", str(args.out)], check=True)
        zp.unlink()                                   # drop the big zip after prepping
        shutil.rmtree(args.raw / plot, ignore_errors=True)   # and the extracted images
    img_dir = args.out / "images"
    n = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    print(f"done: {n} image/mask tiles -> {args.out}")


if __name__ == "__main__":
    main()
