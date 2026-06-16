# VeinForge 代码审核日志 (Code Audit Log)

> 五轴审查(正确性 / 可读性 / 架构 / 安全 / 性能)全模块过一遍。发现的问题 + 修复 + 待优化都记在这。

## 审核 #1 — v0.1.0 → v0.1.1（2026-06-15）

### 总评
整体健康:模块化、可插拔 segmenter、TDD(44 测试)、**SQL 全参数化(无注入)**、无密钥入库。
找到并修复 **2 个健壮性 bug**;其余为非阻塞优化项,已记录。

### ✅ 已修(含回归测试)
1. **[正确性] 空文件夹崩溃**:`process_folder` 处理无图文件夹时,`write_summary` 对空表 `groupby("sample_id")` → KeyError。
   → 加空表守卫;回归测试 `test_process_folder_empty_no_crash`、`test_write_summary_empty_no_crash`。
2. **[健壮性] skan 退化骨架抛异常**:`skeleton.py` / `orient.py` 遇闭环等无端点骨架时,`skan.Skeleton` 可能抛错,一张坏图就让整条管线崩。
   → try/except 优雅降级(脉长退回像素计数 / 方向退回 NaN);回归测试 `test_skeleton_ring_no_crash`。

### 🔲 已记录待优化(Consider / Optional,非阻塞)
- ✅ **[性能,已修 v0.1.3]** `preprocess` 改用高斯背景(替 `rank.mean(disk)`),大图快得多。
- ✅ **[性能,已修 v0.1.3]** `orient` 复用 `measure` 已算的骨架(去掉一次 skeletonize)。
- **[健壮性]** `segment` 在全空/常值图上 `threshold_otsu` 可能报错 → 加守卫。
- **[健壮性]** `stress` 训练样本极少 / 类别极不均时 `cross_val` 可能失败 → 加守卫。
- **[边界]** `io` 的 `ResolutionUnit==1`(无单位)被当作 cm → 加判断。
- **[架构]** DB schema 无迁移(新列只对新建库生效)→ 见 [`backlog.md`](backlog.md)。

### 五轴评分(本次)
| 轴 | 分 | 备注 |
|---|---:|---|
| 正确性 | 8 | 修了 2 个 bug;边界守卫仍可加 |
| 可读性 | 9 | 命名清晰、模块聚焦 |
| 架构 | 9 | 可插拔 segmenter、清晰边界 |
| 安全 | 9 | SQL 参数化、无密钥、外部数据边界读取 |
| 性能 | 7 | 热点在 sato 多尺度 + rank.mean;CPU 下可接受 |

**裁决:通过**(明确改善了代码健康)。下次审核前,优先消化"待优化"里的性能两项。

---

*每次审核在此加一节;问题→修复→回归测试 三件套齐全才算闭环。*
