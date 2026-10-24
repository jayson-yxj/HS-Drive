# M8-A 架构决策

**AssetNoGo；结束本轮P0，保持原架构和全部安全门禁。** 本轮授权的唯一候选是原始PETR R50-DCNv2-C5，不引入替代模型。

固定官方配置与下载入口已核实，实际完整检测checkpoint未取得。Google连接超时，百度提取码验证成功后目录返回errno=-7；后者可能涉及服务、会话或客户端兼容，未完成归因。因此能成立的结论是“本次有界资产获取失败”，不能推断“PETR不适合CARLA/4090”或“官方权重不存在”。

| 决策 | 状态 |
|---|---|
| 官方检测checkpoint实际取得 | AssetNoGo |
| 新依赖/算子/模型加载/原规格前向/24GB资源 | 全部NotRun |
| 新CARLA投影与盒合同验证 | NotRun，前置条件未满足 |
| 采用PETR、完整Adapter、训练/微调、默认模型替换 | 不批准、不自动开展 |
| 其他3D框架或Direct Head恢复 | 不自动开展 |
| 后续研究 | 仅在新的独立任务中处理官方资产来源与可达性，再决定是否重开P0 |

M5-O维持ResearchCompleted / CandidateNoGo；M7维持ResearchCompleted / OfflinePlannerResearchGo / OnlinePlannerNoGo；车辆ControlNoGo、RuntimePartial、Occupancy NotPassed、视觉SafetyNotValidated不变。默认Depth、HeightSplat、Planner、Safety、Controller、CARLA与Physics保持。

若未来取得文件，仍须隔离验证DetectionCheckpoint、BackboneInitializationWeights、OptimizerState及Metadata；不能凭文件名或大小跳过检测Head覆盖与shape检查。P0前向成功最多证明技术可行性，不授予域质量、可靠定位、Occupancy或视觉驾驶资格。

本阶段交付了可复核的失败结论和可复用资产门禁工具，没有生成虚构显存/时延成绩。详见[总结](../reports/m8a_summary.md)、[原始访问证据](../reports/m8a_execution_logs/official_attempt_01/outcome.json)及[复现文档](M8A_IMPLEMENTATION.md)。
