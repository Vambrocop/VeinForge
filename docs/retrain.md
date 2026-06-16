# 如何(再次)训练 VeinForge 的 DL 模型

> 经典 CV(`--segmenter classical`)**不用训**。这份讲怎么训 / **重训** DL 分割模型。
> 本地 CPU 或免费 [Colab GPU](colab-gpu.md) 都行;真正提分靠**更多标注数据**。

## 1) 数据布局(同名 PNG)
```
data/train/images/*.png   data/train/masks/*.png     # 训练:图 + 脉掩膜(掩膜 >127 为脉)
data/val/images/*.png     data/val/masks/*.png       # 固定 holdout(用来比较,别换)
```
- **你的真实数据**:脉图 + 掩膜。掩膜省力法:先 `veinforge run` 自动描 → `veinforge correct` 画笔修几笔 → 当真值。
- **没真数据先验证流程**:
  ```bash
  python scripts/make_synth_pretrain.py --style realistic --out data/train
  python scripts/make_synth_pretrain.py --style realistic --n 40 --out data/val
  ```

## 2) 训练(本地)
```bash
pip install -e ".[dl]"
python scripts/train_dl.py --data data/train --val data/val --epochs 50 --out models/leaf_unet.pt
```
- `--val` 固定 holdout,**按最佳 val IoU 存模型**(跨次可比)。
- clDice 连通性损失 + 数据增强**默认开**。

## 3) 训练(免费 GPU,推荐用于大/真实数据)
见 [colab-gpu.md](colab-gpu.md) —— Colab/Kaggle 一键,敢用原分辨率/大模型/多轮。

## 4) 用训练好的模型
```bash
veinforge run ./tiles --pixel-size-um 1.23 --segmenter dl --model models/leaf_unet.pt --tile-size 512
```

## 5) ⭐ "再次训练 / 越训越强"的循环(你问的)
```bash
# 1. 攒更多标注 → 加进 data/train/
# 2. 从当前最佳模型【热启动继续训】(不必从零):
python scripts/train_dl.py --data data/train --val data/val \
       --init models/leaf_unet.pt --epochs 30 --out models/leaf_unet_v2.pt
# 3. 在【同一个固定 holdout】上比 val IoU,涨了就替换 models/leaf_unet.pt
# 4. 在 docs/scorecard.md 版本历史加一行,记录涨了多少
```
> **关键**:每次都在**同一个固定 holdout**上比,数字才可比(我们早期换 val 集踩过坑,见 [backlog.md](backlog.md))。
> `--init` = 重训的核心:在已有模型基础上继续学,比从零快、也更稳。
