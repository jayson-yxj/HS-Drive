# M8-J5C：数值归因与最小监督集合收束

2026-10-26 UTC。基础4143011，分支 `feat/m8j5c-numerical-supervision-closure`。

**数值线取得明确结果；数据线采用退出条件B，本片段保留为Ignore回归集。真实Smoke仍NotRun。**

| 事项 | 本轮状态 |
|---|---|
| 固定六层输出的Matching/Loss/直接梯度 | FP32与实际AMP输出合同三组对照全部通过 |
| 整网FP32 | eval/train三组全部通过原容差；只认证有限合成配置 |
| 当前官方AMP | 未通过；差异定位至DCNv2半精度输入梯度累加 |
| Torch严格确定性开关 | 官方索引赋值失败，单独保留No-Go |
| 8帧正常检测监督资格 | No-Go，17个Positive诊断候选、27个未解释区域、零帧负监督资格 |
| 真实数据更新 | NotRun，未启动优化器 |

数值对照每次恢复同一参数/缓冲区、输入和全部RNG，清空梯度。FP32最大梯度差7.90e-7，满足atol1e-6/rtol1e-5；AMP最大约9.77e-4。最后一个DCNv2的输出梯度一致，输入梯度首先出现差异；去除Loss、Decoder和checkpoint后仍能独立复现。已安装CUDA源码的atomicAdd与半精度累加路径提供实现证据。并非仅凭“原版也波动”就放行新版。[数值报告](m8j5c_numerical_attribution.md)

旧214帧No-Go同时包含**规则过宽**与**实际证据不足**：检测no-object不需要逐Query证明物理空间为空，也不需要为两类可见任务认证十类遮挡后空闲。但635条未解释区域观测的距离和任务归属不能凭小尺寸或token推断；其中已确认4个缺Box案例位于3帧，不能说214帧全部标错。

固定8帧为00017、00045、00076、00107、00124、00155、00196、00213，没有换帧凑通过。六摄全对象材料及27项具体疑点已生成，真实人类审核Pending。范围外、不可见与不可靠Box均保留并展示Ignore，未删除后自动作为背景。[候选与取舍](m8j5c_supervision_decision.md)

下一步只建议一个替代数据方案：独立审核nuScenes Mini中两个官方mini_train场景的8个关键帧，先1帧合同验证，再最多7帧；本轮未下载、未承诺新数据必然完整。现片段不继续无界审查。

30项CPU检查、17项浏览器检查通过；FP32通过与AMP失败分开记录。历史A=Partial、B=No-Go、C=NotRun未改写。原四组P/I/N仍是**四帧×六层**的历史回归，不是四个Decoder层。

被阻止的3步FP32 Smoke配置、具体样本00076/00107及全部缺失前置条件见 `configs/m8j5c_smoke_proposal.json`；只读preflight退出2，没有执行更新。[复现与实施](../docs/M8J5C_IMPLEMENTATION.md)
