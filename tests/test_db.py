from veinforge.db import (
    connect, init_db, insert_run, upsert_sample, insert_image, insert_measurement,
)


def test_db_roundtrip(tmp_path):
    conn = connect(tmp_path / "veinforge.db")
    init_db(conn)
    run_id = insert_run(conn, params_json='{"a":1}', version="0.1.0")
    sample_id = upsert_sample(conn, {"sample_id": "S1", "species": "barley",
                                     "treatment": "heat", "replicate": "r1"})
    image_id = insert_image(conn, {"sample_id": "S1", "path": "x.tif", "position": "mid",
                                   "pixel_size_um": 2.0, "width_px": 512, "height_px": 512})
    insert_measurement(conn, image_id, run_id, {"vein_density": 3.2, "areole_count": 49})
    rows = conn.execute("SELECT vein_density, areole_count FROM measurements").fetchall()
    assert rows == [(3.2, 49)]
    assert sample_id == "S1"
    assert run_id == 1 and image_id == 1
