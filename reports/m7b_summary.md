# M7-B — Dynamic Prediction Robustness & Scenario Validity

**总体Partial：完成场景修复和有限动态假设离线基准；新的最终道路未检出接受冲突，在线5Hz预算仍未通过。** 基于3d149ac，独立分支feat/m7b-dynamic-planning-robustness，规则在a18f754冻结后才运行验证/最终。全部是实际服务器执行的离线CARLA0.9.15地图+合成Actor实验；新车辆闭环、GPU推理、训练均0。

## 主要结果

| 集合 | 场景有效/尝试 | CV接受候选/参考冲突 | 多假设接受候选/参考冲突 | 多假设选中场景 |
|---|---:|---:|---:|---:|
| 开发3/20 |46/46|64/6|42/0|32|
| 验证76/19 |45/46|52/4|29/0|25|
| 新最终2/4 |46/46|64/6|42/0|32|
| 历史道路65/67/18回归 |68/69|79/6|48/0|41|

全部207次场景尝试，205个有效；开发v1重复46次单独保留，不并入独立场景分母。同地图、共享23模板、仅两条新最终道路；不把布局复制当独立泛化样本。验证和回归的两个加速参考超出路线支持，保留FixtureInvalid，未延长路线或改阈值后重测。

道路弧长放置替代起点切线，相邻阻塞查询当地邻车道；起步参考初速0、等待后有限加速。新最终独立审计36个Actor车道/航向、42个接受候选的0.4m车身地图网格，未发现违规。参考仍是OpenDRIVE/合成OBB，不是物理Mesh真值。[场景一致性](m7b_fixture_consistency.json)、[未来连续性](m7b_future_motion_validity.json)、[独立地图审计](m7b_independent_map_audit.json)。

历史7条有效CV漏判全部检出；旧错误不修改，历史回归不充当新泛化。[历史七条正确审计](m7b_historical_seven_v2.json)。初版读取JSON时未恢复Stamp的失败产物及traceback保留，不计为有效评测。

新最终CV的6条接受冲突中4条会被选中；B多假设接受与选中冲突均0。有限集合覆盖不同制动幅度、启动延迟、速度、距离和Actor尺寸，参考含不在枚举列表上的加速度。没有证明所有中间参数或任意未来安全，路线末端未知明确拒绝。[假设](m7b_dynamic_hypotheses.json)、[错误接受](m7b_dynamic_false_accepts.json)、[轨迹图](m7b_figures/m7b_motion_hypotheses.png)。

## 不是全部拒绝，但以停车为主

新最终28个动态场景中24个选出无参考冲突候选（85.71%）；32个有有限库见证的场景没有全部拒绝。总体选中32个：A_keep6、B_left1、D_stop25；拒绝14个，其中8个时间/缺失/Worker故障表示。D_stop运动学失败0，独立参考下停车3m采样间距失败0。7个行驶/绕行选择不等于完成路线，25个停车不等于真实舒适停车。

未启用C新增停车生成：开发未发现旧候选全拒绝而解析停车见证可行的缺口。有限见证库没有解也不能证明物理全局无解，报告不编造“正确拒绝所有无解场景”指标。[覆盖报告](m7b_candidate_coverage.json)、[C条件门禁](m7b_C_gate.json)、[风险与覆盖图](m7b_figures/m7b_risk_coverage.png)。

保守性另行公开：只对单一实际脚本未来存在合法见证、但在所有假设下未获证书的全拒绝场景，开发/验证/最终/回归分别3/10/4/12个。它们不能算已知鲁棒解被漏选，也不能从任务能力损失中隐藏。最大减速度的必要停车距离下界另证明5个直道近阻塞场景无法保持3m间距完成原车道停车，5/5拒绝；这是纵向停车无解，不是任意二维逃逸都无解。[覆盖限制与必要停车界](m7b_coverage_limits.json)。

## 性能与保持开放的门禁

B离线组件成本P50/P95/Max：新最终302.92/367.02/529.80ms，38/46超过200ms；开发P95 332.21ms，验证198.43ms，历史回归278.45ms。组件成本已排除A对照、停车见证和独立参考，但不是在线调度实测。主要热点包含地图exact查询及完整车身swept polygon；批量SAT已与500个标量样例及全部开发判定对照一致，未删除检查。[性能报告](m7b_planning_latency.json)、[延迟图](m7b_figures/m7b_latency.png)。

六摄40packet SHA、K/T、同步和来源合同再次通过，没有重复跑DirectHead。Unknown非Free、原timestamp与过期丢弃、禁止视觉控制继续有效。Bench2Drive/PETR无新增执行。[Shadow回归](m7b_shadow_regression.json)。

测试：**424 passed、8 skipped、1 xfailed；SDK专项51 passed**。默认Planner/Safety/Controller、Physics、模型、M7-A及此前资产均不覆盖，完整历史源码字节核验与最终资产清单留存。没有严格20Hz或共享GPU新证据。

## 五项决策

A：场景生成在两条新最终道路限定通过；道路连接/短路段支持仍Partial。

B：历史与新最终动态风险识别在有限假设/参考内通过，未升级为任意未来安全。

C：存在真实合法规划候选，非全部拒绝；以停车为主，任务完成能力仍有限。

D：在线200ms预算NoGo，不直接接入5Hz控制。

E：可以提出受限Oracle离线或非实时Pilot研究；实际在线控制接入暂不批准。下一阶段优先精确缓存/几何复用、连接道路支持、合理预测覆盖和继续行驶能力，并使用新的最终集验证；不再次在本轮最终集调参。

[验收JSON](m7b_acceptance.json)、[完整结果](m7b_robust_candidate_evaluation.json)、[新最终道路](m7b_fresh_final_roads.json)、[实现及复现](../docs/M7B_IMPLEMENTATION.md)、[架构决策](../docs/M7B_ARCHITECTURE_DECISION.md)。ControlNoGo、RuntimePartial、视觉CandidateNoGo、Occupancy未通过、纯视觉驾驶NotValidated均保持。
