# VeinForge P2 路线：深度学习分割 + 热胁迫表型

> 面向非专业读者写。P2 有两块：**①让 AI 学会更稳地描脉(要训练)**、**②用叶脉数字判断热胁迫(轻量 ML)**。
> P1(经典 CV)已完成并合入 main；P2 在其基础上扩展。

---

## 一句话现状

- **DL 分割要训练吗?** 要。但**不用你手动跑**，用 `scripts/train_dl.py` 一条命令。
- **有现成网络图能直接训练吗?** 没有"现成的大麦/小麦脉网+标注"。所以走**迁移学习**：
  **先用开放的双子叶数据预训练 → 再用你少量小麦/大麦图微调**。
- **胁迫表型要训练吗?** 几乎不算——是随机森林这类**小样本、无需 GPU**的轻量 ML。

---

## 数据现实(逐个核实过)

| 资源 | 能用吗 | 说明 |
|---|---|---|
| **Zenodo Bornean / LeafVeinCNN** | ✅ 开放可下、**带人工掩膜** | 双子叶；**预训练首选** |
| LVD2021(4977 图、像素脉标注) | ⚠️ 需填表申请 | 双子叶(配套番茄)；可作预训练 |
| Quantitative Plant 的 cleared-leaves | ❌ 全部 deprecated、链接失效 | 别在那找了 |
| 你实验室的大麦/小麦透明化图 | 🔜 待你提供 | **微调与验证的最终目标，无可替代** |

> 标注省力法：**先用 P1 经典 CV 自动描一版 → 在 Ilastik 里人工修几笔** → 当作训练"标准答案"，不用从零手画。

---

## 已搭好的框架(数据一到就能插上跑)

- `veinforge/segment/unet.py` — 一个小型 U-Net(细管状结构主力网络)。
- `veinforge/segment/dl.py` — `DLSegmenter`，**和经典分割同一个接口**(`segment(image, params)->掩膜`)，
  训练好后**直接替换 P1 的分割模块**，后面测量流程一行不改。
- `scripts/train_dl.py` — 训练脚本(BCE+Dice 损失、支持 `--init` 预训练权重热启动微调)。
- `pip install -e ".[dl]"` — 装 PyTorch 等(可选，平时不装，不拖累 P1)。

### DL 分割：怎么跑(数据就位后)

```bash
pip install -e ".[dl]"                      # 装 torch（建议有 GPU）

# 1) 预训练：开放双子叶脉图 + 掩膜
python scripts/train_dl.py --data data/pretrain --epochs 50 --out models/pretrain.pt

# 2) 微调：你的少量大麦/小麦图（warm-start 自预训练权重）
python scripts/train_dl.py --data data/barley --init models/pretrain.pt \
       --epochs 30 --out models/barley_unet.pt
```

用法(替换分割引擎)：
```python
from veinforge.segment.dl import DLSegmenter
from veinforge.pipeline import process_folder
from veinforge.params import Params
process_folder("tiles", Params(pixel_size_um=1.23), "results",
               segmenter=DLSegmenter(model_path="models/barley_unet.pt"))
```

数据目录：`data/<set>/images/*.png` 与 `data/<set>/masks/*.png`(同名，掩膜 >127 为脉)。

### ✅ 无需人工标注的验证（已跑通）

不依赖你的数据、也不需要任何人工标注，用**合成脉图(掩膜免费、完美)**把整条 DL 链路验证过了：

```bash
python scripts/make_synth_pretrain.py --n 48 --size 128 --out data/pretrain_synth
pip install -e ".[dl]"
python scripts/train_dl.py --data data/pretrain_synth --epochs 10 --out models/synth_unet.pt
```

结果：训练损失 **0.71 → 0.11** 稳定下降；用 `DLSegmenter` 在**没见过的新合成图**上分割，**IoU = 1.00**。
说明 `train_dl.py → 存模型 → DLSegmenter 推理` 整条链路正确、即插即用(全程零下载、零人工)。

> ⚠️ 实战注意：DL 模型要在**和推理时同样的预处理**下训练(要么都用原图、要么都过 preprocess)，否则分布不一致会掉点。本 demo 用原图(脉为亮)。

### 真实公共数据的现实（逐个核实过）

- **Zenodo Bornean / LeafVeinCNN**：带人工掩膜，但每个 zip **601MB–1.6GB**、按 plot/tree 编码，需较大下载与整理。
- **LVD2021**(4977 图、像素脉标注)：需**填表申请**，非直接下载。
- Roboflow / HuggingFace 上多是**整叶 / 病害**分割，不是脉掩膜。
- → 暂无"小巧、免登录、可直接下"的真·叶脉掩膜集。上真实数据需接受 ~600MB 下载或走申请。

---

## 热胁迫表型：怎么做(P1 输出之上)

**科学依据**：叶脉密度会随热/旱胁迫变化(常变密)，与水分输送、光合直接相关——
所以叶脉数字里藏着胁迫信号。参考：项目内 *小麦热胁迫成像综述*(JxB 2025)、
[水稻叶脉密度×水分胁迫 (Sci Rep)](https://www.nature.com/articles/srep36894)。

**步骤**：
1. **分组成像**：正常组 vs 热胁迫组的大麦/小麦各拍一批。
2. **跑 VeinForge**：得到每株性状表(就是 P1 的 `results.csv`)。
3. **看差异**：统计胁迫组 vs 对照组的脉密度等是否显著不同(本身就能出结论/出图)。
4. **训分类器**：用**随机森林 / XGBoost**(小样本友好、无需 GPU)学"叶脉数字 → 是否/多严重受胁迫"。

> 这块**比 DL 省数据**，也是本项目**真正的发表新意**——多数已有工具只"量脉"，很少把脉性状接到胁迫响应上。

### 已实现（现在就能用）

```bash
pip install -e ".[stress]"                                        # 装 scikit-learn（很轻）
veinforge stress compare results/results.csv --label treatment    # 不训练：看哪些脉性状随胁迫显著变化
veinforge stress train   results/results.csv --label treatment --out models/stress_rf.joblib
veinforge stress predict new.csv --model models/stress_rf.joblib --out preds.csv
```

> 合成 demo 实测：`compare` 正确挑出 **vein_density (p≈5e-11)** 与 **interveinal_distance (p≈9e-10)** 随胁迫显著变化；`train` 随机森林交叉验证 **≈98%** 区分对照/胁迫。（数据为合成，流程真实；真实结论待你的胁迫/对照样本。）

### 分割后端可切换（所有选项都留着）

```bash
veinforge run tiles --segmenter classical                          # 不训练（默认）
veinforge run tiles --segmenter dl --model models/barley_unet.pt   # 训练好的 DL
```

---

## 阶段小结

| 阶段 | 内容 | 需要训练 | 需要的数据 | 状态 |
|---|---|---|---|---|
| P1 | 经典 CV 量叶脉 | 否 | 无 | ✅ 完成 |
| P2-a | DL 分割(U-Net) | 是 | 开放双子叶(预训练)+你的小麦/大麦(微调) | 🧰 框架就绪，待数据 |
| P2-b | 热胁迫表型 | 轻量 | 胁迫/对照分组样本 | ✅ 已实现（compare 不训练 / RF 训练），待真实数据验证 |
