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
