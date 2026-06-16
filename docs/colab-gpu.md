# 免费云 GPU 训练(Colab / Kaggle)

> CPU 慢 → 用**免费 GPU** 训正经 DL(原分辨率、大模型、多轮),不用买硬件。
> 把下面的 cell 依次贴进一个新的 **Google Colab** 笔记本;先 `Runtime → Change runtime type → GPU (T4)`。

```python
# 1) 确认拿到 GPU
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "没有 GPU — 去选 GPU runtime")
```

```bash
# 2) 装 VeinForge
!git clone https://github.com/Vambrocop/VeinForge
%cd VeinForge
!pip install -q -e ".[dl]"
```

```bash
# 3) 准备数据
#    A. 先用合成料验证 GPU 流程跑通:
!python scripts/make_synth_pretrain.py --style realistic --size 256 --n 200 --out data/train
!python scripts/make_synth_pretrain.py --style realistic --size 256 --n 40  --out data/val
#    B. 真实数据:把 data/train、data/val 换成你的 images/ + masks/(同名 PNG)
#       标注省力法:P1 经典 CV 自动描 → Ilastik 修几笔(见 docs/training-plan.md)
```

```bash
# 4) 训练(GPU 上敢用原分辨率 / 大模型 / 多轮 —— CPU 上做不动的)
!python scripts/train_dl.py --data data/train --val data/val \
        --epochs 100 --batch 16 --out models/unet.pt
#   迁移微调:--init <预训练.pt> · 固定 holdout 按最佳 IoU 存:--val 已开 · clDice+增强默认开
```

```python
# 5) 把模型下回本地,接回流水线
from google.colab import files
files.download("models/unet.pt")
#   本地: veinforge run ./tiles --pixel-size-um 1.23 --segmenter dl --model unet.pt --tile-size 512
```

## 备注
- **Kaggle** 同理:新建 Notebook → 右侧 Accelerator 选 GPU → 同样的命令(用 `!`)。
- GPU 解锁的关键:`--size 256/512`(原分辨率,配合本地 `--tile-size` 推理)、`--batch` 调大、`--epochs` 几百、可换更大模型(见 [training-plan.md](training-plan.md))。
- 真正提分仍靠**你的小麦/大麦标注数据**;GPU 只是让你"敢放开手脚"训。
