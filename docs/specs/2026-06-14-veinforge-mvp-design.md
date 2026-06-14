# VeinForge P1 (MVP) 设计规格 / Design Spec

- **日期**：2026-06-14
- **状态**：待用户 review（design phase）
- **范围**：P1（MVP）——大麦/小麦透明化叶片**平面脉网**的叶脉性状自动定量，经典 CV，零训练数据。
- **相关文档**：[`README.md`](../../README.md) · [`docs/related-work.md`](../related-work.md)

---

## 1. 目标与非目标

### 目标（P1）
把现在 ImageJ 里**手动描脉长、手数自由末端**的叶脉定量，升级为**自动、批量、参数可复现**的 Python 工具：
输入一批透明化叶片显微 tile 图 →（标定像素尺寸）→ 自动分割叶脉 → 输出**脉密度、脉宽、自由末端密度、areole 统计、脉间距**等性状（CSV + SQLite + 质控叠加图）。

### 非目标（明确不在 P1）
- ❌ 深度学习分割 → **P2**
- ❌ 双子叶 1°–7° 分级；单子叶**纵/横脉分离**与束鞘排除 → **P1.5**
- ❌ 横切面解剖、整叶照片为主的形态分析 → 远期
- ❌ 热胁迫表型分析 → **P2**
- ❌ Web 部署
- ❌ 训练任何模型（P1 无学习成分）

### 已定默认（待 review 可改）
| 决策 | 默认 | 理由 |
|---|---|---|
| GUI | **napari** | 科研图层叠加/人工校正友好 |
| 纵/横脉分离 | **不进 P1**（→P1.5）| 保持 MVP 聚焦；P1 给方向无关总密度 |
| 图像数据集存放 | 本地 `data/`（gitignore），发表上 Zenodo | 大图不入代码仓库 |

---

## 2. 输入与假设

- **图像**：透明化（番红/固绿染色或仅清化）叶片在透射光显微镜下的**矩形 tile**（参照 protocol：每图 ≥1.5 mm²，取叶片上/中/下三处）。格式 `.tif/.tiff/.png/.jpg`。
- **标定**：每张图需要**像素物理尺寸**（µm/px）。来源优先级：① 图像元数据（TIFF resolution）→ ② 命令行 `--pixel-size-um` → ③ 后续可加比例尺自动识别（非 P1）。无标定时仅输出像素单位并告警。
- **元数据**：样本号、处理(treatment)、重复(replicate)、位置(top/mid/bottom)。来源：文件名约定（如 `S012_heat_r2_mid.tif`，可配 regex）或随附 `manifest.csv`。
- **假设**：veins 在增强后比背景**更亮或更暗且呈线状**；同一批图成像条件大致一致（允许通过参数微调）。

---

## 3. 架构：可插拔流水线

模块各管一段、接口清晰。**关键缝是 `segment`**：定义统一签名 `image → 布尔脉掩膜`，P1 用经典 CV 后端，P2 的 DL 后端实现同一签名即可替换，**其余模块不动**。

```
io → preprocess → segment → skeleton → measure → report
                  (classical ↔ DL)                 ↘ db
```

| 模块 | 职责 | 关键依赖/算法 |
|---|---|---|
| `io` | 读图、读/写标定、元数据解析、保存 | tifffile, imageio |
| `preprocess` | 灰度/通道选择、匀光去背景、CLAHE、可选反相 | skimage（rolling_ball / equalize_adapthist）|
| `segment` | **血管增强 → 阈值 → 形态学清理 → 布尔掩膜**（可插拔接口）| skimage.filters.sato/frangi, threshold_otsu/hysteresis, morphology |
| `skeleton` | 骨架化 + 图分析（总脉长、端点、分支点、段）| skimage.morphology.skeletonize, **skan** |
| `measure` | 由掩膜/骨架/距离变换算全部性状 | scipy.ndimage(EDT), skimage.measure(regionprops) |
| `db` | 结果写入 SQLite | sqlite3（stdlib）|
| `report` | CSV、参数快照 yaml、质控叠加图 | pandas, pyyaml, skimage/matplotlib |
| `cli` | 批量入口 `veinforge run ...` | typer/click |
| `gui` | napari 看原图+掩膜+骨架图层（可选 extra）| napari |

### 分割接口（DL 可替换的核心契约）
```python
# segment/base.py
class VeinSegmenter(Protocol):
    def segment(self, image: np.ndarray, params: SegmentParams) -> np.ndarray:  # -> bool mask
        ...
# P1: ClassicalSegmenter   |   P2: DLSegmenter  —— 同签名
```

---

## 4. 处理流程（单张 tile）

1. **读图 + 标定**：载入，取得 µm/px；记录尺寸。
2. **预处理**：转灰度或选最佳通道（染色图中脉对比最大的通道/光密度）；rolling-ball 去不均背景；CLAHE 增对比；必要时反相使**脉为亮**。
3. **血管增强**：多尺度 `sato`（或 `frangi`）vesselness，σ 范围由预期脉宽（µm→px）推导。
4. **阈值**：先 Otsu；为保连通性提供**滞后阈值(hysteresis)**选项。→ 二值脉掩膜。
5. **形态学清理**：`remove_small_objects`（去碎点）、`binary_closing`/`area_closing`（补小缺口）。
6. **骨架化 + 图分析**：`skeletonize` → `skan.Skeleton`：**总脉长**（段长求和×像素尺寸）、端点数、分支点数。
7. **脉宽**：对掩膜做 EDT，骨架点处 `width = 2×EDT`；报均值/中位数（µm）。
8. **areole / 脉间距**：反掩膜连通域 → 去除触边者 = areoles（计数、均面积）；脉间距由背景 EDT 的局部极大≈半间距估计（×2），P1 给近似值，P1.5 精化。
9. **自由末端**：骨架端点中**不触图像边界**者计数；密度 = 数/有效面积。
10. **输出**：写 DB + CSV 一行；存 `params.yaml`；渲染质控图（原图叠掩膜+骨架+端点标记）。
11. **聚合**：同一 sample 的 top/mid/bottom 自动求均值（另存 sample 级 CSV）。

---

## 5. 性状定义（对齐 protocol）

| 性状 | 定义/公式 | 单位 |
|---|---|---|
| 脉密度 VLA | 总脉长 / 图像面积 | mm·mm⁻² (= mm⁻¹) |
| 脉宽（直径）| 骨架处 2×EDT 的均值/中位数 | µm |
| 自由末端密度 | 内部骨架端点数 / 有效面积 | mm⁻² |
| areole 数/面积 | 内部连通背景域计数 / 均面积 | 个；µm² |
| 脉间距 | 背景 EDT 极大×2 的均值（近似）| µm |
| 脉面积占比 | 掩膜面积 / 图像面积 | — |
| 总脉长、图像面积 | 中间量（便于复核）| mm；mm² |

> P1 不区分脉级/方向，自由末端密度分母用**总图像面积**（protocol 中需扣 2° 脉面积——该精化属 P1.5）。

---

## 6. 数据模型（SQLite `veinforge.db`）

```sql
samples(sample_id TEXT PK, species TEXT, treatment TEXT, replicate TEXT, notes TEXT, created_at TEXT)
images(image_id INTEGER PK, sample_id TEXT FK, path TEXT, position TEXT, pixel_size_um REAL,
       width_px INT, height_px INT, imported_at TEXT)
runs(run_id INTEGER PK, params_json TEXT, veinforge_version TEXT, created_at TEXT)
measurements(measurement_id INTEGER PK, image_id INT FK, run_id INT FK,
       vein_density REAL, mean_vein_width_um REAL, median_vein_width_um REAL,
       free_ending_count INT, free_ending_density REAL,
       areole_count INT, areole_mean_area_um2 REAL, interveinal_distance_um REAL,
       vein_area_fraction REAL, total_vein_length_mm REAL, image_area_mm2 REAL, created_at TEXT)
```
P2 的胁迫数据可直接以 `sample_id`/`treatment` JOIN。

---

## 7. 输出物

- `results.csv`：每图每次运行一行（measurements + 样本/图像元数据）。
- `samples_summary.csv`：sample 级（top/mid/bottom 均值±SD）。
- `qc/<image>_overlay.png`：原图 + 掩膜 + 骨架 + 端点标记，肉眼质控。
- `params.yaml`：本次所有参数 + 版本（可复现）。
- `veinforge.db`：累积结果库。

---

## 8. CLI 与 GUI

**CLI**
```
veinforge run <folder> --pixel-size-um 1.23 [--config params.yaml] [--manifest manifest.csv] [--out results/]
veinforge view <image>            # 打开 napari 看分割结果
```
**GUI（napari，可选 extra `pip install veinforge[gui]`）**：原图 + 掩膜 + 骨架三图层叠加、可切换；P1 仅查看/质控，**人工校正画笔留 P1.5**。

---

## 9. 仓库结构

```
veinforge/
  __init__.py        io.py        preprocess.py
  segment/__init__.py segment/base.py segment/classical.py   # P2: segment/dl.py
  skeleton.py        measure.py   db.py   report.py
  cli.py             gui.py       params.py（dataclass+yaml）
tests/               # 合成图 + 单元 + (待数据)真图回归
examples/            # 用 leaf.jpg 等做冒烟（注：仅测代码可跑，非训练）
data/                # gitignore：原图/标注/中间结果
docs/  pyproject.toml  .gitignore  README.md
```

---

## 10. 依赖

核心：`numpy scipy scikit-image skan pandas pyyaml typer tifffile imageio`（sqlite3 为标准库）。
可选 extra：`napari`（GUI）。Python ≥ 3.10。保持核心精简，GUI/DL 各自 extra。

---

## 11. 测试与验证

- **合成脉图生成器**：画已知间距/宽度的平行脉+横脉 → **真值已知**（脉密度、脉间距、脉宽、端点数）。断言管道在容差内复原（脉密度 ±5%）。
- **单元测试**：`io` 标定换算、`measure` 在手工小掩膜上的公式正确性、`db` 读写。
- **真图回归（待数据）**：少量真 tile 对照 **ImageJ 手动测量**，出相关/Bland-Altman 报告；有共享图再与 **GrasVIQ** 比。

---

## 12. P1 验收标准（Definition of Done）

1. `veinforge run <folder> --pixel-size-um X` 跑通一整个文件夹，产出 `results.csv`、写入 `veinforge.db`、每图存质控叠加图与 `params.yaml`。
2. 合成测试全过（脉密度等关键性状在容差内）。
3. `veinforge view <image>` 能在 napari 中叠加显示原图/掩膜/骨架。
4. 缺失标定时优雅告警并退回像素单位。
5. `pip install -e .` 可装；`pytest` 全绿；README 有最小使用示例。

---

## 13. 路线图

- **P1.5**：单子叶纵/横脉分离（方向场/FFT 主方向）、横脉间距、束鞘排除、自由末端分母扣主脉面积、napari 人工校正画笔。
- **P2**：DL 分割（U-Net/nnU-Net + clDice 损失，视网膜血管迁移学习；用 Ilastik 引导出的标注）+ **热胁迫表型**（脉性状向量 → 胁迫等级，RF/XGBoost）。

---

## 14. 待办/未决

- 🔲 用户核查实验室数据（数量/格式/有无标注）——决定 P2 时点，不卡 P1。
- 🔲 文件名/manifest 的元数据约定细节，落地 `io` 时定稿。
