"""Download a handful of real leaf-vein images for local pipeline dev.

These are dicot reticulate-vein images (NOT barley/wheat) — usable only as
pipeline smoke tests and (later) P2 pretraining, never as monocot ground truth.
For full annotated datasets see docs/related-work.md section 7.
Run: python scripts/fetch_dev_samples.py
"""
from pathlib import Path
import urllib.request

DEST = Path("data/dev-samples")
URLS = {
    "imagej_leaf.jpg": "https://imagej.net/ij/images/leaf.jpg",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        target = DEST / name
        if target.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"fetch {name} <- {url}")
        urllib.request.urlretrieve(url, target)
    print(f"done -> {DEST.resolve()}  (gitignored; see docs/related-work.md sec 7)")


if __name__ == "__main__":
    main()
