# M4.5 架构决策

技术基线延续原六摄RGB→BEV/Temporal→Queries/Occupancy→Planner→独立Safety→Controller。M4.5没有重新设计主干，也没有改变已训练模型参数结构。

| 决策 | 原因/影响 | 状态 |
|---|---|---|
| ConfidenceSplat 独立诊断类，默认委托原 HeightSplat | 旧 checkpoint 和默认行为严格兼容；不写全局开关 | 已实现/验证 |
| 保留原始 Mass 与 log1p 通道 | 单独考察特征归一化；M不等于occupancy probability | 已验证 |
| 当前不替换默认 Splat | 证明了幅度效应，但未提升概率定位；冻结头存在OOD | 决策 |
| observed occupancy 三态，语义/障碍/Drivable分离 | ground也有物理表面，Unknown不得为负；规则不同于物理可见性 | 已实现，训练质量Partial |
| 枚举精确网格面交点遍历 | 非均匀四层Z也正确；CPU可解释，避免新CUDA扩展 | 单测/实际数据验证 |
| self bounding box全射线保守过滤 | 防止近场自车误标Free；明显牺牲地面覆盖 | 临时方案，待实例/mesh self-mask |
| GT以独立sidecar保存 | 原索引未改，learned输入allowlist不变；后续loader必须仅在supervision路径加载 | 已验证隔离；未接Occupancy训练loader |
| SE2 Warp，inverse sampling，align_corners=False | 网格中心与X/Y轴明确；网格外mask不等于Free | 几何通过 |
| Memory保存hidden/pose/time/scene/valid | 时间间隔、定位跳变、场景变化重置；CARLA时间误差1e-6s容差 | 已实现/验证 |
| 拒绝相对倾斜>.05rad或高度变化>.5m | 当前2D memory不能严谨表示大坡道/跨高度；小倾斜仍是平面近似 | 范围限制 |
| OracleState/Trajectory与网络接口分离 | 真值仅建立控制底座，不能混入视觉性能结果 | 已验证 |
| 官方CARLA GlobalRoutePlanner + PurePursuit/PID | 复用成熟路由，实现低速可测闭环；新增networkx3.4.2 | 实际9场景 |
| footprint离散加密而非连续swept volume | 能覆盖车身而非仅中心点，仍有连续性/规则缺口 | 有限安全检查 |
| 配对数据全归diagnostic val | 同路线所有天气/运动变体不能跨split，历史Town05/Town10非blind | 天气16帧已采，运动配对未采 |

本轮实测与因果结论边界：

- 已证实：弱mass的均值归一化会取消概率幅度；门控抑制幅度；A/B/C质量落点不变；Warp坐标正确；简单Oracle路线与故障制动可执行。
- 有支持但不充分：当前道路头更依赖Context/位置先验、学习BEV空间一致性不足；支持来自旧干预和本轮depth shuffle，不能归因于归一化单独一项。
- 未证实：gated训练后能改善障碍定位、Temporal会修复深度偏差、Occupancy已足够训练泛化模型、低速Oracle通过意味着纯视觉安全。

旧 Joint/Far/Road 三模型训练Loss、采样权重、道路监督和选checkpoint策略不同。跨模型表格是现状比较，绝非严格单变量消融。架构选择仍需同初始化、数据、步数、Loss、种子的受控训练。
