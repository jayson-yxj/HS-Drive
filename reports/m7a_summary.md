# M7-A — Scenario Planning Benchmark & Six-Camera Shadow Integration

**总体Partial：完成可复用的离线规划与六摄Shadow基准；动态规划、车辆执行和正式视觉闭环没有通过。** 基础`eb2c8f4`，独立分支`feat/m7a-scenario-shadow-benchmark`。默认模型、Depth/HeightSplat、Planner/Safety/Controller、Physics和main保持不变。

## 实际执行范围

| 证据类型 | 本轮实际工作 |
|---|---|
| 离线CARLA SDK/地图+合成Actor场景 | Town03六个道路位置、84场景、180候选；开发/验证/最终规则分开冻结 |
| 历史真实CARLA回放 | M6-A13次、M6-B69次完整结果和2次初始化失败；另16个冻结快照Planner冷重算 |
| 实际RTX4090离线推理 | 两帧冻结Mobile+Direct，另独立数据读审计重复两帧；无训练 |
| 六摄数据审计 | 40份packet、240张RGB、38个历史位置组 |
| 官方源码审查 | 固定Bench2Drive0.0.4，4项传感器验证函数测试 |
| 单元/工程回归 | 413 passed、8 skipped、1 xfailed；SDK专项40 passed |
| 本轮新CARLA闭环、训练、PhysX编译、220评测 | 全部NotRun/0 |

所有新Planner场景明确是“真实保存OpenDRIVE上的合成Actor参考”，不冒充新的CARLA车辆试验。历史真实执行是单独分母。相关JSON均包含配置、源码和输入资产SHA。

## A：M6是否正式收束？

**是。** 保留ResearchCompleted、CausalityPartial、InstrumentationNotRun、ControlNoGo、RuntimePartial五项状态。SHA核验13,534份基线文件、50,867份继承资产、111份M6-G交付资产及4份0916身份文件全部一致。没有重启底层研究。[正式结题记录](m7a_dynamics_research_closeout.md)。

## B：规划基准是否可复用？

**工具可复用，能力门禁Partial。** 统一OracleSceneState、PlanningRequest、CandidateTrajectory、PlanningEvaluation，显式区分PlanningFeasible与VehicleExecutedSuccessfully。使用原MotionPlanner，新增离线评价完整车身包络、旋转OBB、六秒CV预测、车道合法性、时间、速度积分、曲率/加速度/Jerk及拒绝语义。

| Split | 道路ID | 原预期满足 | 场景评价总耗时P95/Max |
|---|---|---:|---:|
| 开发v2 | 3、18 | 26/28 | 297.89 / 347.19ms |
| 验证 | 67、19 | 26/28 | 313.69 / 358.85ms |
| 最终 | 65、20 | 24/28 | 232.74 / 248.55ms |

该计数是预设逻辑期望，不是安全通过率。每split的5类预期拒绝×两路共10项均拒绝；缺失/worker_error是离线故障表示，不是新增真实Worker崩溃测试。相邻片区最小采样路线间距27.36m，模板高度相关，不把84场景当84个独立真实道路样本。

74个接受候选经独立0.025s多边形碰撞和0.4m完整车身地图网格审计：**CV参考碰撞0、Driving道路违规0**。这仍是采样/OpenDRIVE参考，不是连续物理Mesh保证。

必须保留的失败：

- 初始运动一致的前车制动参考下，**7个已接受候选会相交**。Planner只能读当前快照；CV不能覆盖制动未来。不能用CV安全覆盖预测失配。
- 原始另6个restart冲突来自脚本初速度1m/s与未来前两秒静止的不一致，标参考无效，不混入有效动态结论。
- 最终6个场景沿初始切线放置障碍，在弯路偏离目标路线约2.81m或14.93m；保留FixtureMismatch，未改位置后重跑。
- 左绕行在开发道路3可行，但验证道路67完整guide path越界而转为D_stop；右绕行仅有合成合同测试，没有本轮独立实际道路支持。
- 新完整道路/guide审计与旧Safety在部分历史路口快照存在差异，没有据此证明旧Safety错误；原默认安全模块不替换。

[规划总账](m7a_oracle_planning_benchmark.json)、[候选质量](m7a_candidate_trajectory_quality.json)、[参考有效性](m7a_fixture_validity.json)、[场景目录](m7a_scenario_catalog.json)、[候选图](m7a_figures/m7a_candidates.png)。

## 实际执行与Runtime仍独立

M6-A13次、M6-B69次运行的碰撞、路线距离、终点、停车和舒适失败完整保留。M6-B原三正常路线两重复虽达到路线末端，新增终点要求仍0/6；低速三臂舒适0/24。四次限定合法绕行达到几何终点，但舒适仍失败；阻塞后的实际OBB间距原3m要求仅1/4满足。**均不能用本轮规划几何通过升级。**

历史M6-B控制间隔P50/P95/P99/Max：49.992/50.512/51.171/**178.083ms**；world tick P95/Max 2.365/94.276ms；Worker P95/Max87.259/569.282ms。15次deadline miss、16个跳槽、过期规划用于正常控制0。新评价器完整离线审计P95超过200ms，也不认证5Hz生产运行。没有CARLA与视觉共享GPU实测，严格20Hz保持Partial。

[历史执行与重放](m7a_historical_regression.json)、[Runtime状态](m7a_runtime_status.json)、[离线耗时图](m7a_figures/m7a_planning_latency.png)。

## C：六摄Shadow接口是否稳定？

**限定历史回放接口通过，感知CandidateNoGo。** 40个packet的六摄顺序、原始frame/time、384×640 RGB/SHA、K/T、坐标往返及Ego元数据均检查通过；覆盖38个历史位置，不是新的封存泛化集。

实际两帧冻结RGB推理输出9候选，另进程带数据读白名单复现，位置逐值相同。GT只在单独Evaluator读取，2个历史关联Actor Center的XY误差约0.187/0.155m；7个未匹配候选保持Unknown。**两个Train样本小误差不推翻M5-O CandidateNoGo。** Ego真值位姿仅用于离线对齐，不进入本轮模型，不意味着视觉里程计已实现。

GPU峰值allocated140.41MiB；首帧冷启动722.63ms，第二帧14.72ms，不能由两帧推断稳态P95。离线队列按原始有效期将冷启动结果判ExpiredDropped，未重盖时间戳；live队列与正式控制NotRun。

[六摄合同](m7a_six_camera_shadow_contract.json)、[来源隔离与全部候选](m7a_oracle_vision_isolation.json)、[六摄候选图](m7a_figures/m7a_six_camera_shadow.png)。

## D：Bench2Drive是否具备后续接入条件？

**仅具备有限源码/接口准备基础，完整接入和官方评分NoGo。** 固定官方0.0.4提交`7ec25d1c9f7522d923ce5f3420986cef1cb2d956`，六RGB在SENSORS配置检查通过；Qualifier最多4RGB不适用。官方Agent返回VehicleControl、ScenarioManager拥有tick/apply，不能与HSDrive双Owner混用。[官方接口源码](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/leaderboard/leaderboard/autoagents/agent_wrapper.py)。

README要求CARLA0.9.15和附加地图，但仓内ScenarioRunner版本文件仍0.9.13，依赖含numpy1.18.4等；不能据版本号直接认定完整兼容。当前缺Town11/12/13/15，HS历史10Hz相机与20Hz逐帧等待需单独适配。全路线XML传输截断/超时，完整ID/地图清单NotVerified；没有下载训练数据或启动220路线。

正式Driving Score为完成度×处罚的220固定分母均值；Success是Completed/Perfect且无非min-speed违规。需要220个预期唯一ID和失败/崩溃完整性，不能只检查条数。Dev10只作开发诊断，不冒充新0.0.4全评测。[官方合并脚本](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/merge_route_json.py)。

[兼容性报告](m7a_bench2drive_feasibility.json)区分实际函数检查和完整框架NotRun。

## E：是否重启成熟Camera-only 3D架构P0？

**当前不批准。** 本轮没有新的合法官方Checkpoint、完整隔离环境和4090前向显存证据，M6-A PETR资产NoGo保持；不重启Direct Head扩训。

下一阶段优先修正开发场景生成/未来连续性、补充动态预测不确定性及道路连接审计，预留新的最终道路；六摄Shadow继续保持离线。Bench2Drive可另做隔离环境准备，但正式视觉闭环评分仍需先有合格候选。

[验收JSON](m7a_acceptance.json)、[实现及复现](../docs/M7A_IMPLEMENTATION.md)、[独立决策](../docs/M7A_ARCHITECTURE_DECISION.md)。原停车、舒适、Physics-P1、G1、Occupancy、视觉驾驶和共享GPU实时门禁均不升级。
