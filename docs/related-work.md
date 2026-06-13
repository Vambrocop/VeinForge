# VeinForge — 相关工作与参考资料 (Related Work & References)

> 本文整理 VeinForge 直接相关的**已有工具、深度学习参考、经典算法积木、前沿方向、方法学基础**，
> 并标注每一项对 VeinForge 各开发阶段的启发。
>
> ⚠️ 引用中的卷期/页码等细节请以链接原文为准（本文优先保证链接可达、不臆造书目细节）。

VeinForge 定位：面向**大麦 / 小麦（单子叶平行脉）**的叶脉性状自动定量工具——
经典图像处理底座 + 可插拔深度学习分割 +（第二阶段）热胁迫表型，强调一键安装、批量、参数可复现。

阶段标记：**P1** = 经典 CV 的 MVP；**P1.5** = 单子叶纵/横脉与脉间距；**P2** = 深度学习分割 + 热胁迫表型。

---

## 1. 直接同类工具（最该研究 / 站在其肩上）

### GrasVIQ (2021) — 与我们场景最接近
- **是什么**：专为**禾本科 / 谷类的平行脉**设计的图像分析框架，经典计算机视觉，
  从透明化叶片图自动分割并量化**脉密度、脉宽、脉间距、脉数量**。
- **为何重要**：这基本就是 VeinForge **P1（MVP）的目标**，已经有人为草类做出来了。
  我们不重造，而是**复用其思路 + 工程化 / 可复现 / 胁迫扩展**来超越它。
- **要借鉴**：平行脉分割与量化流程、对单子叶几何的处理假设。
- 链接：<https://pubmed.ncbi.nlm.nih.gov/33914380/> ·
  [ResearchGate PDF](https://www.researchgate.net/publication/351188411)

### phenoVein (2015) — 自动叶脉分割与分析，开源
- **是什么**：自动分割与分析叶脉，支持多种成像方式（显微镜、宏观摄影等），开源、文档完善。
- **要借鉴**：分割 + 骨架 + 性状测量的工程组织、跨成像模态的鲁棒性设计。
- 链接：[Plant Physiology 169(4):2359](https://academic.oup.com/plphys/article/169/4/2359/6114117) ·
  <https://pubmed.ncbi.nlm.nih.gov/26468519/> ·
  [Plant Image Analysis 收录](http://www.plant-image-analysis.org/software/phenovein)

### LEAF GUI (Price et al., 2011) — 叶脉网络提取与度量
- **是什么**：图形界面工具，提取叶脉网络并度量脉密度、夹角等。
- **要借鉴**：网络拓扑度量（节点 / 段 / 环）的定义方式。

---

## 2. 深度学习分割（P2 参考样板）

### LeafVeinCNN / Xu et al. (2021, New Phytologist)
- **是什么**：U-Net 类 CNN 自动分割叶脉网络；在 700+ 叶片、约 50 个东南亚植物科的
  人工标注真值区域上训练；提供 Windows 独立程序 `LeafVeinCNN.exe`（自带 Matlab runtime）。
- **为何重要**：VeinForge **P2 深度学习分割**的直接参考——网络结构、训练数据规模、
  评估方式（与经典 CV / 人工对照）。
- **对我们的提醒**：DL 需要**标注好的大麦 / 小麦透明化脉网图**作为训练集；
  ImageJ 自带演示图（leaf.jpg 等）**不能**用于训练（物种 / 模态 / 标注都不对）。
- 链接：[Wiley / New Phytologist](https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.16923) ·
  [软件 (Zenodo)](https://zenodo.org/records/4007731) ·
  [bioRxiv 预印本](https://www.biorxiv.org/content/10.1101/2020.07.19.206631v1.full)

---

## 3. 经典算法积木（P1 直接复用）

VeinForge 的经典分割管道 = **血管增强滤波 → 阈值 → 形态学清理 → 骨架化 → 图分析 → 距离变换量脉宽**。
这些在 ImageJ/Fiji 和 scikit-image 里都有成熟实现，互为参照：

| 步骤 | ImageJ / Fiji 参考 | scikit-image / Python 等价 |
|---|---|---|
| 匀光 / 去背景 | Subtract Background（rolling ball） | `skimage.restoration.rolling_ball` |
| 血管 / 线状增强 | Tubeness、Frangi vesselness | `skimage.filters.frangi / sato / meijering` |
| 脊线检测 | Ridge Detection 插件 | `skimage.filters.meijering` / steerable filters |
| 骨架化 | Skeletonize (2D/3D) | `skimage.morphology.skeletonize / medial_axis` |
| 骨架图分析（脉长 / 分支 / 端点） | **AnalyzeSkeleton** | `skan`（skeleton network analysis）|
| 连通域 / areole / 颗粒分析 | Analyze Particles | `skimage.measure.label / regionprops` |
| 脉宽 | 距离变换 | `scipy.ndimage.distance_transform_edt` |

- Fiji 插件文档：[AnalyzeSkeleton](https://imagej.net/plugins/analyze-skeleton/) ·
  [Tubeness](https://imagej.net/plugins/tubeness) ·
  [Ridge Detection](https://imagej.net/plugins/ridge-detection)
- ImageJ 下载与**源码**（用户指定参考）：
  [下载](https://imagej.net/ij/download.html) ·
  [源码目录](https://imagej.net/ij/download/src/)

---

## 4. 前沿方向（远期 / 选做）

### Frontiers in Plant Science (2025) — 多光谱 + 高分辨率 3D
- **是什么**：融合多光谱与高分辨率 3D 成像做叶脉分割与脉密度（VLA）测量，
  **免去透明化 / 压平这类破坏性、费时的预处理**。
- **启发**：VeinForge 长期可考虑非破坏性成像输入；但与当前透明化脉网 MVP 解耦。
- 链接：[Frontiers 2025](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1560220/full) ·
  [PMC 全文](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11931515/)

### 烟草叶脉积分梯度分割 (2024)
- 基于积分梯度的叶脉分割方法，可作分割算法对照。
- 链接：[ACM 2024](https://dl.acm.org/doi/10.1145/3688574.3688595)

---

## 5. 领域方法学基础（性状定义的出处）

- **Sack & Scoffoni (2013) 叶脉结构综述** — 项目随附 PDF `2013 Sack & Scoffoni, Vein Structure.pdf`；
  叶脉等级、密度、功能的权威综述。
- **大麦叶脉性状量化双语 protocol**（项目随附 `leaf_vein_traits_protocol_bilingual_barley_v11.pdf`）—
  定义了我们要自动化的性状：各级脉密度、总 / 主 / 细脉密度、脉直径、单位面积自由末端数，
  以及整叶面积 / 周长 / 长 / 宽、长宽比、周长²/面积。来源：PrometheusWiki “Quantifying leaf vein traits”。
- 关键支撑文献（详见上面 protocol 的参考列表）：Sack & Holbrook (2006) 叶片水力学；
  Uhl & Mosbrugger (1999) 脉密度与脉间距；Brodribb et al. (2007) 脉与光合 / 水力；
  Sack et al. (2008) 掌状脉与水力冗余对损伤的耐受性（→ **胁迫表型的理论接口**）。

---

## 6. 启发映射（来源 → 我们拿什么 → 阶段）

| 来源 | VeinForge 拿什么 | 阶段 |
|---|---|---|
| GrasVIQ | 单子叶平行脉分割 + 量化流程；差异化做工程化/可复现 | P1 / P1.5 |
| phenoVein | 分割→骨架→测量的模块组织；多模态鲁棒性 | P1 |
| ImageJ/Fiji 算法 | Frangi/Tubeness、Skeletonize、AnalyzeSkeleton、距离变换 | P1 |
| 大麦 protocol + Sack & Scoffoni | 性状定义与计算公式（密度/脉间距/自由末端/形态） | P1 |
| LeafVeinCNN (Xu 2021) | U-Net 分割、训练/评估范式 | P2 |
| Sack et al. 2008 等胁迫文献 | 脉性状 → 热胁迫响应的表型分析 | P2 |
| Frontiers 2025 多光谱+3D | 非破坏性成像的长期方向 | 远期 |

---

*维护：本文随项目推进持续补充。新增参考时，请同时更新第 6 节的启发映射表。*
