# M5-I Architecture Decision Record

状态：已完成受控研究；默认系统无迁移。

## 决策A：视觉架构

仅保留Mobile为独立迁移候选，先做逐camera/类别与BEV/Depth接口回归，不能直接替换默认Encoder。

输出迁移是否在两seed池化P/R/Removed均改善：False。Mobile池化三项稳定改善：True；六摄P95≤20ms：True。这些是必要证据，逐位置/相机/小目标一致性仍独立，不能用池化结果关闭泛化门禁。

A/B共享初始化/顺序/预算；C独立GT+Feature的合同已实现，但B满足预声明验证收益条件，C本轮未执行，不得声称其训练有效。Mobile ImageNetV2与ResNet ImageNetV1+CARLA谱系、通道/结构、FPN初始化不同，属于架构对照。冻结主干研究不能定位容量上限与微调潜力。

## 决策B：Depth候选

Replay保留只读研究Checkpoint，未获得升级资格；近场收益不能抵消Bias/远场支持或空间质量不达标。默认原Depth不变。

[
  {
    "seed": 101,
    "qualified": false
  },
  {
    "seed": 102,
    "qualified": false
  }
]

Bias归因来自固定候选相同射线对比，场景/天气/相机之间仍存在混杂。没有用全局补偿值掩盖问题，也未用Test拟合温度。全部48LID、几何坐标和HeightSplat保持。

## 下一阶段与退出条件

若视觉与Depth未同时取得稳健收益，不启动正式Occupancy规模化。已通过的工程接口不等于空间泛化、安全或实时控制通过。可独立继续Depth几何Shadow；视觉分支若继续研究，先解决数据域/目标分辨率/监督可靠性，再预声明有限迁移对照，避免无界增加Loss。

如果未来新Student稳定通过二维且BEV仍受限，再研究更密采样与Lift融合；如果两个模块均通过，另行申请有限BEV/Occupancy Pilot。Temporal BEV、Sparse Query与视觉控制本轮不启动。

证据：[总结](../reports/m5i_summary.md)、[资格](../reports/m5i_depth_qualification.json)、[架构对照](../reports/m5i_lightweight_backbone.json)、[验收](../reports/m5i_acceptance.json)。


## 可执行的下一阶段方向

优先为Mobile视觉分支建立独立迁移版本，继续保留原Encoder/Depth对照，先审查左右侧背景误检及BEV接口；本轮不是默认替换授权。相比继续叠加KD，它在当前相同预算下提供更强的可复现证据。小目标支持不足、左侧Precision不足仍需独立门禁。

几何研究与视觉迁移分开：P8 Oracle采样仅72.06%表面Recall，LearnedDepth附加空间扩散明显；Replay降低MAE却降低新Test远场正确空间质量。先扩展可靠Actor评价支持，再预声明采样密度/概率投影对照；不更改48LID或默认Splat。当前不启动正式Occupancy规模化，只能在后续新决策中评估有限联合Pilot。

工程接口通过、二维学习相对增益、二维绝对门禁、Depth资格、BEV泛化、GPU算时、完整Runtime及驾驶安全分别判断；不存在单项收益自动关闭全系统门禁。
