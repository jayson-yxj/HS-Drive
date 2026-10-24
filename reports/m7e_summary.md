# M7-E — State-Anchored Trajectory Feasibility & Planning Latency Gate

**总体Partial / OfflineOnly。真实起点连接改善了有限独立序列的行驶覆盖；200ms完整交付门禁仍失败，不批准在线Shadow或车辆控制。** 基线 `6ed646c`，独立分支 `feat/m7e-state-anchored-planning`。原M7D最终road30/1仅作历史回归，默认模型、Planner/Safety/Controller/Physics与全部旧资产不变。

A：新增唯一 `F_state_anchored_slow`，采用实测XY/航向锚定的五次Hermite曲线，再按弧长配置原慢行速度。581条成功构造轨迹的t=0最大状态误差2.78e-17；未吸附中心线，也不是只覆盖首点。13项新增合成几何/合同测试验证非零偏移、小航向、曲线、曲率拒绝、积分、过期和缓存身份。实测输入没有可信初始加速度/Yaw Rate，零计划初始加速度与曲率明确为假设。负实测速度直接拒绝F，不截为零。构造失败标记`InitialStateConnectionInfeasible`，构造成功仍须通过完整几何与风险检查。[轨迹](m7e_state_anchored_trajectory.json)。

B：原488条E候选中，485条有>1μm位置差，472条有>1μrad航向差；按原检查器实际容差，404条触发起点不一致。仅236条是其他几何、动态、余量均通过时，单独被初始状态挡住。115条另有slow reserve失败，122条存在运动支持不足，不能全部归因中心线投影。新F在开发两段分别增加23、26个继续行驶时刻，参数随后冻结。[逐帧归因](m7e_initial_state_failure_attribution.json)、[开发门禁](m7e_development_gate.json)。

C：独立采集三段18秒、每段361帧，按原0.2s/六秒协议共183个规划时刻。新road65/67/28与开发实际Ego路径最近距离97.10m；地图道路以前用于合成回归，本轮未用于开发，不能称未见城市。spawn时加入小幅横向/航向偏差，后续均为真实物理快照，未传送或用Planner控制。

| 新最终序列 | A继续 | B继续 | 停车 A→B | 拒绝 A→B |
|---|---:|---:|---:|---:|
| road65 制动/重启 |0|27|21→16|40→18|
| road67 恒速/曲线 |0|28|44→30|17→3|
| road28 横穿 |0|0|0→0|61→61|

“继续”是离线候选选择，车辆没有执行；连续窗口高度相关，55个增益时刻不是55个独立场景。Person没有当前合法运动走廊继续Unknown。旧最终road30仍无F增益，不能声称所有状态已解决。[对照](m7e_candidate_ablation.json)、[新场景支持](m7e_fresh_episode_support.json)。

D：历史205场景/1662候选重新配对，区域缓存与原结果全部等价；七条历史危险候选重新检出7/7。六秒、全部运动假设、原车身扫掠、精确地图查询、3m和原0.2秒有效期保留。8段旧序列＋3段新序列共671次规划，638条已接受候选（其中F为250条）均用实测未来审核，未发现采样OBB碰撞；所有已接受F的实际参考停车余量检查通过。三种状态仍分开：`RobustWithinHypotheses`、`SafeForRecordedFuture`、`FeasibleToExecute=NotValidated`。0.05s OBB不是精确连续Mesh真值，实际Actor不会响应反事实Ego。[等价](m7e_geometry_equivalence.json)、[未来审计](m7e_actual_future_collision_audit.json)。

E/F：唯一静态区域缓存以Map SHA、Road/Lane/Section、原始Polygon WKB和inset为完整身份；不缓存动态通过结论或过期状态。它只省掉部分union/buffer开销，新F的完整guide审核反而增加工作量。无并发、真实连续输入的完整序列化交付计时见下表及专项报告；不以相同输入预热成绩代替移动时性能。

| 开发序列 | 原A P95 | 缓存A P95 | A+F P95 | A+F Max | A+F过期 |
|---|---:|---:|---:|---:|---:|
|dev_constant_r3|410.45|410.49|646.14|652.41|61/61|
|dev_braking_r20|453.54|459.42|676.84|695.51|60/61|

单位ms。区域缓存命中率接近99%，但未形成可重复的总时延优势。新F增加了完整连接guide审核负担。全体671次回放666次过期；各场景完整分位数与有效比例独立留档。

完整结果保留原timestamp、原输入可用墙钟、计算/对象创建/序列化和payload可读时刻；磁盘归档与真实IPC另不声称通过。预算固定200ms，未给过期结果重新盖时间戳。[完整时延](m7e_full_payload_latency.json)、[时序稳定性](m7e_temporal_stability.json)。

**退出决策：保留离线质量改进，停止当前M7实现的自动扩展，不自动启动M7-F。** 下一阶段先评估计算架构与项目优先级，不继续追加候选、Loss式调参或无界缓存重构。在线Shadow、实际执行、低速动力学、严格20Hz、RGB/Occupancy安全继续开放。[架构决策](../docs/M7E_ARCHITECTURE_DECISION.md)。

三段新采集无碰撞回调；不等于新Planner零碰撞闭环。未来读取白名单、缺帧、错关联、过期等保持离线测试范围。失败记录保留：NumPy延迟导入被数据读取白名单阻止、JSON NumPy布尔序列化失败、未设置隔离Shapely路径的支持脚本失败，均未覆盖或标为成功。规划/采集/参数冻结SHA通过后未按新最终调整。

测试：**466 passed、8 skipped、1 xfailed；CARLA SDK专项67 passed**。其中13项新候选/缓存测试，历史xfail保持。历史源码/资产SHA核验通过。实际车辆跟踪、实时Worker故障、在线Shadow和新模型训练均NotRun。

完整交付与可复现命令：[验收](m7e_acceptance.json)、[实现](../docs/M7E_IMPLEMENTATION.md)、[决策时序图](m7e_figures/m7e_decisions.png)、[起点连接图](m7e_figures/m7e_state_connection.png)、[时延图](m7e_figures/m7e_full_latency.png)、[新最终0.25m偏差连接](m7e_figures/m7e_final_offset_connection.png)。
