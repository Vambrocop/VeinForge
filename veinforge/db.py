from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples(
  sample_id TEXT PRIMARY KEY, species TEXT, treatment TEXT, replicate TEXT,
  notes TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS images(
  image_id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT, path TEXT, position TEXT,
  pixel_size_um REAL, width_px INTEGER, height_px INTEGER, imported_at TEXT);
CREATE TABLE IF NOT EXISTS runs(
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, params_json TEXT, veinforge_version TEXT,
  created_at TEXT);
CREATE TABLE IF NOT EXISTS measurements(
  measurement_id INTEGER PRIMARY KEY AUTOINCREMENT, image_id INTEGER, run_id INTEGER,
  vein_density REAL, mean_vein_width_um REAL, median_vein_width_um REAL,
  free_ending_count INTEGER, free_ending_density REAL, areole_count INTEGER,
  areole_mean_area_um2 REAL, interveinal_distance_um REAL, vein_area_fraction REAL,
  total_vein_length_mm REAL, image_area_mm2 REAL, created_at TEXT);
"""

_MEASURE_COLS = ["vein_density", "mean_vein_width_um", "median_vein_width_um",
                 "free_ending_count", "free_ending_density", "areole_count",
                 "areole_mean_area_um2", "interveinal_distance_um", "vein_area_fraction",
                 "total_vein_length_mm", "image_area_mm2"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def insert_run(conn, params_json: str, version: str) -> int:
    cur = conn.execute(
        "INSERT INTO runs(params_json, veinforge_version, created_at) VALUES(?,?,?)",
        (params_json, version, _now()))
    conn.commit()
    return cur.lastrowid


def upsert_sample(conn, s: dict) -> str:
    conn.execute(
        "INSERT OR IGNORE INTO samples(sample_id, species, treatment, replicate, notes, created_at)"
        " VALUES(?,?,?,?,?,?)",
        (s.get("sample_id"), s.get("species"), s.get("treatment"), s.get("replicate"),
         s.get("notes"), _now()))
    conn.commit()
    return s.get("sample_id")


def insert_image(conn, im: dict) -> int:
    cur = conn.execute(
        "INSERT INTO images(sample_id, path, position, pixel_size_um, width_px, height_px, imported_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (im.get("sample_id"), im.get("path"), im.get("position"), im.get("pixel_size_um"),
         im.get("width_px"), im.get("height_px"), _now()))
    conn.commit()
    return cur.lastrowid


def insert_measurement(conn, image_id: int, run_id: int, traits: dict) -> int:
    cols = ["image_id", "run_id"] + _MEASURE_COLS + ["created_at"]
    vals = [image_id, run_id] + [traits.get(c) for c in _MEASURE_COLS] + [_now()]
    placeholders = ",".join("?" * len(cols))
    cur = conn.execute(f"INSERT INTO measurements({','.join(cols)}) VALUES({placeholders})", vals)
    conn.commit()
    return cur.lastrowid
