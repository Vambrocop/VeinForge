from __future__ import annotations
from pathlib import Path
import numpy as np


def prepare_layers(image: np.ndarray, mask: np.ndarray, skeleton: np.ndarray) -> list[dict]:
    """Pure description of napari layers (no napari import needed)."""
    return [
        {"name": "image", "data": image, "kind": "image"},
        {"name": "vein mask", "data": mask.astype(np.uint8), "kind": "labels"},
        {"name": "skeleton", "data": skeleton.astype(np.uint8), "kind": "labels"},
    ]


def view(path: str | Path) -> None:                # pragma: no cover (interactive)
    import napari
    from veinforge.params import Params
    from veinforge.io import load_image
    from veinforge.preprocess import preprocess
    from veinforge.segment.classical import ClassicalSegmenter
    from veinforge.skeleton import skeleton_metrics

    image, _ = load_image(path)
    params = Params()
    mask = ClassicalSegmenter().segment(preprocess(image, params), params)
    skel = skeleton_metrics(mask, params.pixel_size_um)["skeleton"]
    viewer = napari.Viewer()
    for layer in prepare_layers(image, mask, skel):
        if layer["kind"] == "image":
            viewer.add_image(layer["data"], name=layer["name"])
        else:
            viewer.add_labels(layer["data"], name=layer["name"])
    napari.run()


def measure_from_mask(mask: np.ndarray, pixel_size_um=None) -> dict:
    """Re-run trait measurement on a (possibly hand-corrected) boolean mask."""
    from veinforge.measure import measure
    return measure(np.asarray(mask).astype(bool), pixel_size_um)


def correct(path, out_mask) -> None:               # pragma: no cover (interactive)
    """Open napari to hand-correct the vein mask, then save it on window close."""
    import napari
    import imageio.v3 as iio
    from veinforge.params import Params
    from veinforge.io import load_image
    from veinforge.preprocess import preprocess
    from veinforge.segment.classical import ClassicalSegmenter

    image, _ = load_image(path)
    params = Params()
    mask = ClassicalSegmenter().segment(preprocess(image, params), params)
    viewer = napari.Viewer()
    viewer.add_image(image, name="image")
    lbl = viewer.add_labels(mask.astype(np.uint8), name="vein mask (paint to correct)")
    print("用画笔在 'vein mask' 图层上修正,改完关闭窗口即保存。")
    napari.run()
    iio.imwrite(out_mask, ((np.asarray(lbl.data) > 0) * 255).astype(np.uint8))
    print(f"saved corrected mask -> {out_mask}")
