# VeinForge

面向 **大麦 / 小麦(单子叶平行脉)** 的**叶脉性状自动定量 + 胁迫识别**工具。
经典图像处理底座 + 可插拔深度学习 + 多维胁迫表型,强调 **一键、批量、可复现、诚实可信**。

> **状态:核心可用** — 经典 CV 叶脉量化 + 两条胁迫识别轨 + DL 框架 + 多维 benchmark,**46 测试全绿**。
> **唯一关键缺口:真实小麦/大麦透明化脉图**(公开数据查证不存在,只能自己成像)。详见 [`docs/scorecard.md`](docs/scorecard.md)。

---

## 安装

```bash
git clone https://github.com/Vambrocop/VeinForge
cd VeinForge
pip install -e ".[dev]"          # 核心 + 测试
pytest -q                         # 验证安装(应 46 passed)
```

可选附加(按需):
```bash
pip install -e ".[stress]"   # 随机森林胁迫分类(scikit-learn)
pip install -e ".[dl]"       # 深度学习分割(PyTorch)
pip install -e ".[gui]"      # napari 查看器
```

---

## 怎么用

### 1) 叶脉性状定量(核心,零训练数据)
```bash
veinforge run ./tiles --pixel-size-um 1.23 --out results
```
输入一批透明化脉网图 → 输出 `results/`:
- `results.csv` — 每图性状:脉密度、脉宽、脉间距、自由末端、areole、**纵/横脉密度**、整叶形态…
- `samples_summary.csv` — 样本级均值±SD · `qc/*_overlay.png` — 质控叠加图 · `params.yaml` — 参数快照 · `veinforge.db` — SQLite 结果库

切换分割后端:`--segmenter classical`(默认,不训练)/ `--segmenter dl --model models/xxx.pt`(训练好的 DL)。

### 2) 查看分割(napari)
```bash
veinforge view ./tiles/example.tif
```

### 3) 叶脉胁迫表型(脉性状 → 胁迫)
```bash
veinforge stress compare results/results.csv --label treatment   # 不训练:看哪些脉指标随胁迫显著变化
veinforge stress train   results/results.csv --label treatment   # 随机森林分类器
veinforge stress predict new.csv --model models/stress_rf.joblib
```

### 4) 整叶 RGB 胁迫/健康分类(并行轨)
```bash
veinforge leafstress train data/leaves          # 布局:data/leaves/healthy/*.jpg, data/leaves/stressed/*.jpg …
veinforge leafstress predict leaf.jpg --model models/leaf_clf.joblib
```

### 5) 训练 DL 分割(可选,需 `[dl]`)
```bash
python scripts/make_synth_pretrain.py --style realistic --out data/pretrain_synth   # 零数据合成训练集
python scripts/fetch_retinal.py                                                       # 视网膜血管预训练料(STARE)
python scripts/train_dl.py --data data/train --val data/val --epochs 50 --out models/unet.pt
#   --init <预训练.pt> 迁移微调 · --val <固定holdout> 按最佳 IoU 存模型 · clDice+增强默认开
```
训练好的 `.pt` 直接插回流水线:`veinforge run ... --segmenter dl --model models/unet.pt`。

### 6) Benchmark 打分
```bash
python scripts/benchmark.py        # 在留出真值上算 IoU/Dice/速度,vs ImageJ/GrasVIQ 参照
```

---

## 文档(都在 [`docs/`](docs/))

| 文档 | 内容 |
|---|---|
| [spec](docs/specs/2026-06-14-veinforge-mvp-design.md) · [plan](docs/plans/2026-06-14-veinforge-p1-mvp.md) | 设计规格 · 实现计划 |
| [p2-roadmap](docs/p2-roadmap.md) | DL 分割 + 胁迫表型路线 |
| [benchmark](docs/benchmark.md) · [scorecard](docs/scorecard.md) | 多维技术对比 · 项目自评(版本追踪) |
| [training-plan](docs/training-plan.md) | GPU/数据到位后怎么优化训练 |
| [colab-gpu](docs/colab-gpu.md) | 免费云 GPU(Colab/Kaggle)训练配方 |
| [backlog](docs/backlog.md) · [ideas](docs/ideas.md) · [audit-log](docs/audit-log.md) | 缺口日志 · 点子日志 · 代码审核日志 |
| [related-work](docs/related-work.md) | 参考工具/数据集 |

---

## 现状与诚实说明

- ✅ 经典 CV 叶脉量化、叶脉胁迫表型、整叶 RGB 胁迫分类、DL 框架、benchmark、全套规划/审核文档。
- ⚠️ 准确度 IoU ≈ 0.52(经典)/ 0.66(DL),受 **CPU + 小数据**限制;速度/自动化/可复现远超手工(ImageJ)。
- 🔴 **唯一关键缺口:真实小麦/大麦图**——拿到后 `train_dl.py --init ... --val ...` 微调即可。其余胁迫轨也都"能跑,需真实带标签数据才出科学结论"。

改进方向与优先级见 [`scorecard.md`](docs/scorecard.md)(分维度打分)、[`training-plan.md`](docs/training-plan.md)、[`backlog.md`](docs/backlog.md)。
