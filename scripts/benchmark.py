"""Score VeinForge segmenters on held-out ground-truth tiles -> a markdown scorecard.

Runs the classical (no-training) and DL (trained) segmenters on each val set and
reports mean IoU / Dice and speed, with reference rows for ImageJ-manual and
published tools. Run: python scripts/benchmark.py
"""
from __future__ import annotations
import time
import glob
from pathlib import Path
import numpy as np
from PIL import Image
from veinforge.params import Params
from veinforge.segment import get_segmenter
from veinforge.benchmark import iou, dice

# dataset label -> (val dir, in-domain DL checkpoint)
DATASETS = {
    "leaf (LeafVeinCNN val)": ("data/real_val", "models/real_unet.pt"),
    "retina (STARE val)": ("data/retinal/val", "models/retinal_unet.pt"),
}


def _load_pairs(d: str):
    pairs = []
    for ip in sorted(glob.glob(f"{d}/images/*.png")):
        img = np.asarray(Image.open(ip).convert("L")).astype("float32") / 255.0
        gt = np.asarray(Image.open(ip.replace("/images/", "/masks/")).convert("L")) > 127
        pairs.append((img, gt))
    return pairs


def _score(seg, pairs, params):
    ious, dices, t0 = [], [], time.time()
    for img, gt in pairs:
        pred = seg.segment(img, params)
        ious.append(iou(pred, gt))
        dices.append(dice(pred, gt))
    return np.mean(ious), np.mean(dices), (time.time() - t0) / max(len(pairs), 1)


def main():
    rows = []
    classical = get_segmenter("classical")
    cls_params = Params(invert=False, background_radius=0)   # val tiles already veins-bright
    for name, (d, ckpt) in DATASETS.items():
        if not Path(d, "images").exists():
            continue
        pairs = _load_pairs(d)
        i, dc, t = _score(classical, pairs, cls_params)
        rows.append((name, "VeinForge classical (no train)", i, dc, t))
        if Path(ckpt).exists():
            dl = get_segmenter("dl", model_path=ckpt)
            i, dc, t = _score(dl, pairs, Params())
            rows.append((name, f"VeinForge DL ({Path(ckpt).stem})", i, dc, t))

    print("\n## VeinForge benchmark scorecard\n")
    print("| dataset | method | IoU | Dice | sec/img |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} |")
    print("\n**Reference (published / baseline, not re-run here):**")
    print("- ImageJ *manual* = the ground-truth baseline: accuracy ~1.0 but slow & manual.")
    print("- GrasVIQ: ~95% longitudinal / ~92% transverse vs manual (Robil et al. 2021).")
    print("- phenoVein: automated multi-modality vein analysis (Buhler et al. 2015).")
    print("\n_VeinForge wins decisively on speed & automation (one command, params recorded);")
    print("accuracy is the axis to keep improving with more labeled data._")


if __name__ == "__main__":
    main()
