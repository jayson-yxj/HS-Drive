# M7-C — Robust Planning Efficiency & Road Continuation

**总体Partial：缓存安全判定等价、有限慢行覆盖改善；冷启动、连续场景和新增候选的整体预算仍NoGo。** 基线684b6e6，独立分支feat/m7c-planning-efficiency-coverage，87cd11d冻结后打开验证与新最终23/28。全部为实际服务器离线CARLA0.9.15地图+合成Actor；新车辆执行、GPU推理、训练均0。

## A：几何复用有收益，未获得整体实时资格

最终版对全部205个有效M7B历史场景、1662条含见证候选逐项对照，数量、合法性、全部拒绝原因、动态冲突、停车3m、Unknown和选择均一致；历史七条危险候选仍7/7识别。地图/车道/车身/原始双精度输入和时间均参与缓存身份，不粗化坐标，不缓存动态结果。PreparedCandidateGeometry保存并复用完整车身和guide扫掠。

| 同输入配对口径 | P50 | P95 | P99 | Max | >200ms |
|---|---:|---:|---:|---:|---:|
| 冻结原核心 |212.09|333.67|408.65|557.94|107/205|
| 新版冷缓存 |219.08|313.97|323.33|549.70|112/205|
| 相同输入预热 |62.13|110.28|111.07|111.38|0/205|
| 连续场景复用 |72.49|217.35|239.25|240.16|16/205|

单位ms。预热是完全相同输入的有利回放结果，不能代表移动Ego。一次缓存优化结束，保留总体NoGo，不自动重写Planner。[等价报告](m7c_geometry_cache_equivalence_prepared.json)、[性能](m7c_planning_latency.json)、[延迟图](m7c_figures/m7c_latency.png)。

另做不含见证/CV/GT审计预热的独立核心计时，交替两臂执行顺序，205场景再次全部等价：基线P95335.10ms/Max363.81ms，缓存P95218.11ms/Max245.75ms，19/205超过200ms。这是连续核心复用的主要证据，不能把上表受其他审计预热的stream当唯一性能依据。[独立核心计时](m7c_clean_core_latency.json)。

## B：道路延拓不能替代缺失的路线意图

ConnectedArcRoute只沿合法Waypoint后继扩展，保存Road/Lane、连接、方向、宽度与Junction；多后继无意图保持Unknown，不制造末端切线。真实地图0/1/60有唯一连接示例，6/18/19存在歧义。历史18/19只补齐数米，restart_1等部分假设增加支持，原加速参考仍不能完整覆盖。额外全车身道路审核不允许把中心路径存在当作道路支持。

[连接报告](m7c_connected_route_support.json)、[预测支持](m7c_prediction_support_audit.json)、[完整车身支持](m7c_connected_body_support.json)。验证选择17/28，无法据此宣称PredictionOutOfSupport整体修复。真实道路终止没有专门实测，终止/逆向/坏连接另有合成单测。

历史18/19的24个前车场景经完整假设车身审核，8个全部支持、16个仍不足；其中6个存在部分未来OBB越出已验证corridor，其余主要为分支或长度不足。仅延长中心路径不能关闭门禁。

## C/D：存在候选库缺口，慢行带来有限覆盖增益

开发找到6个鲁棒慢行见证后，启用唯一预声明族E_slow_follow（0.5m/s、4.5s解析过渡）。保留原假设、加减速度/Jerk、六秒、3m及完整车身，额外检查保守停车余量。C停车扩展没有新准入证据，继续NotRun。

| 集合 | Fixture有效 | 原继续行驶 | 新继续行驶 | 停车前→后 | 拒绝 | 动态场景选中 |
|---|---:|---:|---:|---:|---:|---:|
| 开发3/20 |46/46|7|13|25→19|14|24/28|
| 验证19/76 |46/46|4|7|21→18|21|17/28|
| 新最终23/28 |46/46|4|9|25→20|17|21/28|

全部接受及选中候选的独立脚本参考冲突均0。新最终5个慢行选择六秒前进7.5–10.875m，单一参考最小OBB间距约5.85–10.17m。独立补充审计34个Actor、42个接受候选，未发现道路、朝向、停车间距或参考碰撞错误。

这不是完成路线或执行成功。25个旧停车不能全称必要保守行为；候选库缺口有开发证据，道路限制、支持不足、动态冲突、3m缺证书、单参考可行但鲁棒拒绝和故障逐场景列出。不存在有限库见证不证明全局无解。SafeForOneReference、RobustWithinHypotheses、FeasibleToExecute严格区分。

[准入与见证](m7c_slow_follow_feasibility.json)、[覆盖](m7c_candidate_coverage.json)、[停车归因](m7c_stop_coverage_attribution.json)、[错误接受](m7c_dynamic_false_accepts.json)、[独立最终审计](m7c_independent_final_audit.json)、[覆盖图](m7c_figures/m7c_coverage.png)、[慢行曲线](m7c_figures/m7c_slow_follow.png)。

新增假设车身道路审核和Slow Follow有真实成本：新最终核心P50/P95/Max为160.25/509.11/624.10ms，18/46超过200ms；独立参考等研究工具计入后P95644.89ms。只报告预热成绩会隐藏这项失败。没有删除关键检查或降低精度追求达标。

## E：仅可提出限定非实时Oracle Pilot

有依据继续离线、保存快照或明确非实时研究；不直接控制车辆，不批准5Hz在线接入。下一步应围绕真实缓存命中率、复杂边界查询成本、合法分支意图与未来body支持开展受限研究，不扩大轨迹搜索或放宽假设。

新最终23/28未参与开发调参，46个场景共用23模板，仅两条同Town03道路，不是跨地图泛化。最终失败保留，未修阈值重开最终。[道路与冻结](m7c_fresh_final_roads.json)。

测试：**436 passed、8 skipped、1 xfailed；SDK专项63 passed**。六摄40packet再次核验SHA/K/T/同步，模型仍CandidateNoGo。历史资产、默认Planner/Safety/Controller/Physics和模型不变。原低速舒适性、3m实际间距、严格20Hz及视觉安全门禁保持开放。

[验收](m7c_acceptance.json)、[实现与复现](../docs/M7C_IMPLEMENTATION.md)、[架构决策](../docs/M7C_ARCHITECTURE_DECISION.md)、[Shadow回归](m7c_shadow_regression.json)。首次锚点失败、开发v1guide错误、测试依赖错误和所有旧实验结果均保留，未标成通过。
