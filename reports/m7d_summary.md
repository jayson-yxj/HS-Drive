# M7-D — Non-Real-Time Oracle Temporal Planning Pilot

**总体 Partial。真实时序回放与隔离合同已建立；有限参考未发现已接受候选冲突，慢行覆盖不足，在线计算预算仍 No-go。没有执行本轮规划轨迹，也没有视觉控制。** 基线 `d9899fa`，独立分支 `feat/m7d-oracle-temporal-replay`。

本轮实际执行：审计82份M6A/B日志（68份满足基础字段连续性），选两段历史18秒窗口；CARLA0.9.15/Epic新增6段18秒序列，每段361帧，开发road3/20、验证76/0、新最终30/1。共8段、488个0.2秒重规划时刻、每次六秒未来。相邻窗口高度相关，**488不是488个独立场景**。新最终与开发实际Ego路径最近距离154.10m；同Town03、共享脚本，不是跨地图泛化。

历史205场景/1662候选实际重跑全部等价，历史七条风险单独重跑检出7/7。原模型、历史数据/报告、默认Planner/Safety/Controller/Physics保持不变。完整来源与SHA见release manifest及各报告。

| 序列 | Keep | Left | Slow | Stop | Reject |
|---|---:|---:|---:|---:|---:|
| 开发匀速road3 |4|6|3|48|0|
| 开发制动road20 |4|0|0|14|43|
| 验证重启road76 |4|0|0|38|19|
| 验证静态road0 |0|0|0|0|61|
| 新最终制动road30 |0|0|0|38|23|
| 新最终横穿road1 |0|0|0|0|61|
| 历史前车 |26|0|0|23|12|
| 历史绕行 |51|0|0|0|10|

所有数字是**从实测状态离线生成的选择**，不是车辆执行成功或路线完成率。

A：输入与时间合同在本轮范围通过。当前packet与未来实测参考分库存放；Planner子进程只可读当前packet白名单。原simulation timestamp不改盖，保存原snapshot可用墙钟与新计算墙钟。额外完整payload-ready审计包含返回对象构建和序列化，避免把核心函数结束错称完整交付；它与首轮科学结果逐项对比。过期结果不具备正常在线资格。[时间合同](m7d_temporal_contract.json)、[泄漏审计](m7d_future_leakage_audit.json)。

B：323条已接受候选在实际未来0.05s planar OBB采样中，冲突、3m停车违反、完整车身道路违反均0。新最终只有38条停车接受，横穿场景全部拒绝，不能以零冲突掩盖覆盖不足。`RobustWithinHypotheses`、`SafeForRecordedFuture`、`FeasibleToExecute=NotValidated`分开。实际Actor不会响应反事实Ego轨迹；OBB与OpenDRIVE不是渲染Mesh或精确连续物理真值。[未来审计](m7d_actual_future_collision_audit.json)。

C：慢行未形成稳定、可泛化的真实时序覆盖。仅开发road3有3个时刻选中，分成两个短区间，其他序列0。关键新证据是：冻结慢行首点来自路线中心投影，而真实Ego存在横向/航向偏差，导致`initial_state_mismatch`等拒绝。没有放宽一致性阈值，也没有根据新最终修改轨迹族。[覆盖](m7d_stop_and_progress_analysis.json)。

D：仍有大量停车、拒绝和支持域不足。横穿Person没有当前合法运动走廊，保持Unknown；部分车辆假设完整OBB越出支持道路。单一实测未来安全、但有限多假设无证书的候选单列，不自动称错误拒绝。实际脚本加速度也可能超出声明族，不能宣称完整真实行为鲁棒。[连续性](m7d_decision_stability.json)、[道路支持](m7d_route_support.json)、[实测支持域](m7d_support_audit.json)。

E：一次限定优化有收益，仍不满足200ms。只复用同snapshot Actor/Ego多边形；假设、六秒、3m及查询不减少。两段开发序列的baseline/cold/stream所有科学结果一致。以下是停止CARLA、无其他实验并发的核心计时；完整序列化可用时间另报。完整交付口径下488次中484次过期；新最终制动/横穿P95分别583.49/471.01ms。

| 实测序列 | 基线P95 | 复用后冷缓存P95 | 连续真实帧P95 | 连续Max | >200ms |
|---|---:|---:|---:|---:|---:|
| 开发匀速road3 |511.78|421.05|374.52|376.63|61/61|
| 开发制动road20 |513.30|426.43|406.87|415.22|61/61|

单位ms。相对基线P95下降约26.8%和20.7%，仍全部超预算。本轮停止进一步性能重构，保持在线No-go。[等价](m7d_latency_equivalence.json)、[归因与完整交付时间](m7d_geometry_runtime_profile.json)。

F：具备继续受限离线研究的基础，尚不批准在线Planner Shadow接入或车辆控制。下一步应先研究实际初始状态一致的轨迹连接、受支持运动域/合法路口意图，再单独制定队列上界、时效与非阻塞Shadow合同。原低速执行、舒适性、严格运行时、RGB/Occupancy安全门禁继续开放。[架构决策](../docs/M7D_ARCHITECTURE_DECISION.md)。

保留失败：Low画质Town03触发SIGSEGV，按历史方案改独立Epic实例后采集成功；首次import失败日志保留。独立静态障碍采集脚本记录161次碰撞回调，不等于161次独立碰撞，属于同一接触过程；该脚本不承担避障，不能算Planner驾驶成绩。相关状态原样保留，未删失败提高覆盖率。

测试：**453 passed、8 skipped、1 xfailed；CARLA SDK专项54 passed**。新增时间、缺帧、关联、未来读取、地图中断、过期和优化等价测试。Worker实际故障、新Planner执行及严格20Hz实验均NotRun。全部复现命令见[实现文档](../docs/M7D_IMPLEMENTATION.md)，验收见[m7d_acceptance.json](m7d_acceptance.json)。

诊断图：[时序决策](m7d_figures/m7d_decision_timeline.png)、[计算耗时与实际前车运动](m7d_figures/m7d_latency_and_actual_motion.png)、[慢行空间参考](m7d_figures/m7d_slow_spatial_reference.png)。
