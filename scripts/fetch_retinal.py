"""Download STARE retinal vessels (PaddleSeg mirror) -> image/mask pairs for transfer.

Fundus vessels are topologically similar to leaf veins, so a U-Net pretrained on
them is a strong warm-start for leaf-vein fine-tuning (train_dl.py --init). ~12MB,
no login. Writes data/retinal/{train,val}/{images,masks}/*.png (128px, vessels bright).

Run: python scripts/fetch_retinal.py
"""
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from PIL import Image

URL = "https://bj.bcebos.com/paddleseg/dataset/stare/stare.zip"
ROOT = Path("data/retinal")
SIZE = 128


def _prep(img_path: Path, ann_path: Path, out_dir: Path) -> None:
    green = np.asarray(Image.open(img_path).convert("RGB"))[..., 1].astype("float32")
    inv = 255.0 - green                                  # vessels (dark) -> bright
    inv = (inv - inv.min()) / (np.ptp(inv) + 1e-6) * 255.0
    im = Image.fromarray(inv.astype("uint8")).resize((SIZE, SIZE), Image.BILINEAR)
    ann = np.asarray(Image.open(ann_path).convert("L")) > 0
    m = Image.fromarray((ann * 255).astype("uint8")).resize((SIZE, SIZE), Image.NEAREST)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    im.save(out_dir / "images" / f"{img_path.stem}.png")
    m.save(out_dir / "masks" / f"{img_path.stem}.png")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    zp = ROOT / "stare.zip"
    if not zp.exists():
        print("downloading STARE (~12MB)...")
        urllib.request.urlretrieve(URL, zp)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(ROOT / "stare_raw")
    base = ROOT / "stare_raw" / "STARE"
    n = 0
    for split, out in [("training", ROOT / "train"), ("validation", ROOT / "val")]:
        for img_path in sorted((base / "images" / split).glob("*.png")):
            ann = base / "annotations" / split / f"{img_path.stem}.ah.png"
            if ann.exists():
                _prep(img_path, ann, out)
                n += 1
    print(f"prepared {n} retinal image/mask pairs -> {ROOT}/train, {ROOT}/val")


if __name__ == "__main__":
    main()
