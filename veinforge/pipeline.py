from __future__ import annotations
import json
from pathlib import Path
import veinforge
from veinforge.params import Params
from veinforge.io import load_image, parse_metadata
from veinforge.preprocess import preprocess
from veinforge.segment.classical import ClassicalSegmenter
from veinforge.skeleton import skeleton_metrics
from veinforge.measure import measure
from veinforge import db as dbmod
from veinforge.report import write_csv, write_summary, dump_params, save_overlay

_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def process_image(path, params: Params, segmenter=None) -> dict:
    segmenter = segmenter or ClassicalSegmenter()
    image, meta = load_image(path)
    px_um = params.pixel_size_um if params.pixel_size_um is not None else meta["pixel_size_um"]
    pre = preprocess(image, params)
    mask = segmenter.segment(pre, params)
    sk = skeleton_metrics(mask, px_um)
    traits = measure(mask, px_um)
    md = parse_metadata(Path(path).name, params.filename_pattern)
    row = {**md, "path": str(path), "pixel_size_um": px_um,
           "width_px": meta["width_px"], "height_px": meta["height_px"], **traits}
    row["_image"] = image
    row["_mask"] = mask
    row["_skeleton"] = sk["skeleton"]
    row["_endpoints"] = sk["endpoints"]
    return row


def process_folder(folder, params: Params, out_dir, segmenter=None) -> list[dict]:
    folder, out_dir = Path(folder), Path(out_dir)
    (out_dir / "qc").mkdir(parents=True, exist_ok=True)

    conn = dbmod.connect(out_dir / "veinforge.db")
    dbmod.init_db(conn)
    run_id = dbmod.insert_run(conn, json.dumps(params.to_dict(), default=list),
                              veinforge.__version__)

    rows = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in _EXTS):
        row = process_image(path, params, segmenter)
        save_overlay(row.pop("_image"), row.pop("_mask"), row.pop("_skeleton"),
                     row.pop("_endpoints"), out_dir / "qc" / f"{path.stem}_overlay.png")
        if row.get("sample_id"):
            dbmod.upsert_sample(conn, {"sample_id": row["sample_id"],
                                       "treatment": row.get("treatment"),
                                       "replicate": row.get("replicate")})
        image_id = dbmod.insert_image(conn, {"sample_id": row.get("sample_id"), "path": row["path"],
                                             "position": row.get("position"),
                                             "pixel_size_um": row.get("pixel_size_um"),
                                             "width_px": row["width_px"], "height_px": row["height_px"]})
        dbmod.insert_measurement(conn, image_id, run_id, row)
        rows.append(row)

    write_csv(rows, out_dir / "results.csv")
    write_summary(rows, out_dir / "samples_summary.csv")
    dump_params(params, out_dir / "params.yaml")
    conn.close()
    return rows
