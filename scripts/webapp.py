"""VeinForge web UI (Streamlit) — drag in leaf images, get vein traits + QC, download CSV.

For non-coders. Run:  pip install -e ".[web]"  &&  streamlit run scripts/webapp.py
"""
from __future__ import annotations
import tempfile
from pathlib import Path
import imageio.v3 as iio
import pandas as pd
import streamlit as st
from veinforge.params import Params
from veinforge.pipeline import process_image
from veinforge.report import save_overlay
from veinforge.segment import get_segmenter

st.set_page_config(page_title="VeinForge", layout="wide")
st.title("🌿 VeinForge — 叶脉性状定量")
st.caption("上传透明化脉网图 → 自动量化叶脉性状 + 质控叠加图 + 下载 CSV")

c1, c2 = st.columns(2)
px = c1.number_input("像素尺寸 µm/px (0 = 未标定)", value=2.0, min_value=0.0, step=0.1)
backend = c2.selectbox("分割后端", ["classical (不训练)", "dl (需模型)"])

model_path = None
if backend.startswith("dl"):
    mf = st.file_uploader("DL 模型 .pt", type=["pt"])
    if mf is not None:
        mp = Path(tempfile.gettempdir()) / "vf_model.pt"
        mp.write_bytes(mf.read())
        model_path = str(mp)

files = st.file_uploader("脉网图(可多选)", type=["png", "jpg", "jpeg", "tif", "tiff"],
                         accept_multiple_files=True)

if files and st.button("分析", type="primary"):
    seg = get_segmenter("dl" if backend.startswith("dl") else "classical", model_path=model_path)
    params = Params(pixel_size_um=px or None)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            p = Path(td) / f.name
            p.write_bytes(f.getbuffer())
            row = process_image(p, params, seg)
            ov = Path(td) / f"{p.stem}_ov.png"
            save_overlay(row.pop("_image"), row.pop("_mask"),
                         row.pop("_skeleton"), row.pop("_endpoints"), ov)
            row["_overlay_img"] = iio.imread(ov)        # read back before temp dir closes
            rows.append(row)

    st.subheader("质控叠加图(绿=脉 / 红=骨架 / 黄=末端)")
    cols = st.columns(min(len(rows), 3))
    for i, r in enumerate(rows):
        cols[i % len(cols)].image(r["_overlay_img"], caption=r.get("sample_id", ""),
                                  use_container_width=True)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    st.subheader("性状表")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ 下载 CSV", df.to_csv(index=False).encode("utf-8"),
                       "veinforge_results.csv", "text/csv")
