# M7-A实现与复现

基础`eb2c8f4`，独立分支`feat/m7a-scenario-shadow-benchmark`。新实现全部Opt-in，不连接车辆控制。实际工作是SDK离线OpenDRIVE查询、历史CARLA快照重放、有限冻结RTX4090前向、源码兼容性审计及合成单测；本轮新CARLA闭环、训练、PhysX编译均0。

## 接口与来源

`datasets/m7a_contracts.py`定义值对象，不携带CARLA Actor对象：

| 合同 | 内容及限制 |
|---|---|
| OracleSceneState | Privileged；原始frame/time、世界坐标Ego状态、当前OBB/速度/角速度；无未来真值 |
| PlanningRequest | 上述快照、路线、终点；严格Oracle类型检查，拒绝RGB结构隐式转换 |
| CandidateTrajectory | 含t=0的61点、0.1s采样、六秒时域、世界坐标XY/yaw/speed、原始stamp/有效期、guide path和来源 |
| PlanningEvaluation | reasons、碰撞/道路/运动学/时间/进度/终点指标；PlanningFeasible与VehicleExecutedSuccessfully分开 |
| RGBSpatialEvidence | RGB估计、候选及模型SHA；OfflineShadow/CandidateNoGo，非Occupancy质量保证 |
| DetectedObjectHypothesis | 相机、像素区域、研究性ActorBoxCenter估计；不含Actor引用和GT字段 |
| Unknown | 信息不足，safety_free恒False |

所有M7-A对象的控制导出均抛异常；不实现旧控制类型的隐式适配。这是工程数据隔离，不是针对恶意Python代码的认证边界。

坐标沿用HS世界/自车x前y左z上，camera optical x右y下z前；模型预测在HS Ego。保留原始时刻，不以推理完成时间重盖stamp。六摄顺序固定为front、front_left、front_right、rear_left、rear_right、rear。

## Planner-only

`diagnostics/m7a_planning.py`复用已有MotionPlanner、M6B offset_path及M47连续平移SAT，未修改它们。独立封装A保持路线、B左、C右、D停止，按冻结顺序选择；无可行候选显式Reject。D是规划停止，不保证当前车辆能舒适执行。

碰撞采用OBB、Tesla实测中心偏移和尺寸、安全余量；每段连续平移SAT并增加角度位移上界。道路使用完整车身扫掠多边形及车道并集，能检查内部孔洞，圆弧buffer按多边形逼近误差外扩；仍不是精确动态Mesh连续碰撞。道路参考由0.5m CARLA waypoint构造、内缩0.03m、并执行未投影到道路的SDK查询。它不能替代独立路缘/网格真值。

独立检查初始状态、位置速度积分、航向切向、曲率0.10、速度5.6、加速度1.5/减速度1.8、内部Jerk2.5，以及原始年龄≤0.2s和完整六秒。保持原3m静止间距。guide path另做静态障碍/道路意图检查，时域外动态安全不作承诺。

开发道路3/18，验证67/19，最终65/20，均来自保存的Town03.xodr。最小跨split采样路线间距27.36m；共享场景模板、地图和部分相邻片区，不能声称独立地图泛化。每路14项，同步保留预期与实际差异。规则在`1eb7613`冻结后未依据验证/最终结果调整。

最终场景发现6项障碍切线放置与曲线路线不符，另6项restart未来初速度不连续，保留为FixtureMismatch。没有原地改位置、改预期或重开最终集。修复需独立开发版本和新的最终道路。

`tools/m7a_reference_audit.py`对74个接受候选用0.025s Shapely相交和0.4m完整车身CARLA点查询复核。可检出错误，但采样无碰撞不证明全部时间无碰撞；仍共用OpenDRIVE，非独立物理真值。未来轨迹只进入Evaluator：初态一致的制动参考有7个错误接受；不一致restart的6个原始冲突单列。

## 历史执行与回归

`tools/m7a_historical.py`保留M6-A13次、M6-B69次实际运行，以及B的2次初始化失败。计算真实行驶距离、终点/碰撞/车道、控制间隔、tick、worker、失期/跳槽/降级；所有真实执行保持Privileged。

额外16个历史快照使用冻结MotionPlanner冷启动重算，与原Safety和新Evaluator比较。没有复现历史worker完整内部加速度状态，不能称逐位复刻；新的完整guide path审计也比旧六秒Safety范围更大，因此结果差异不能直接称旧规划回归退化。历史停车/舒适失败独立标ActuationFailure，不被Planner-only通过覆盖。

## 六摄Shadow

`tools/m7a_shadow_prepare.py`仅在审计侧读取历史记录，导出严格白名单packet，模型manifest无GT路径。40个packet覆盖38组，每组首记录另加两个Train组configuration1。检查240张RGB尺寸/SHA、六摄frame/time、K/T刚体性质、Ego元数据一致和投影往返。

`tools/m7a_shadow_infer.py`只接受packet与两个旧Checkpoint；正式输入仅RGB、K、T。Mobile骨干/BF16视觉，FP32 Direct重建；Ego位姿仅作记录对齐，未进入模型。输出均CandidateNoGo；DenseBEV本轮不前向、原结果仍NoGo。无Depth、Instance、Actor或未来真值输入。

`tools/m7a_shadow_audit.py`用Python数据文件访问白名单实际再运行两帧推理，禁止读数据目录/历史cache中的非批准文件。它不是OS沙箱，文件钩子也不证明所有可能C++旁路，报告明确范围；该实际推理未访问GT。

`tools/m7a_shadow_evaluate.py`单独进程读取GT侧车，验证9候选区域与历史一致、两次输出逐值相同；2个关联Actor Center，7个未匹配Unknown。仅两帧历史Train，不能推断泛化。

`ReplayQueue`以值序列化防别名，容量2，拒绝错序、过期、未来、不同步和GT额外字段，溢出丢旧。离线模拟首帧冷启动722.63ms导致超过0.35s有效期而丢弃；第二帧14.72ms。实测峰值allocated140.41MiB，不等于总GPU占用；无CARLA共享负载。live queue/20Hz认证NotRun。

## 环境及命令

没有更改现有两个venv。几何仅将Shapely2.1.2安装到独立`runs/m7a_geometry_deps`，通过PYTHONPATH opt-in：

```bash
.venv/bin/python -m pip install --target runs/m7a_geometry_deps --no-deps shapely==2.1.2
PYTHONPATH=runs/m7a_geometry_deps:. .venv-gpu/bin/python -m pytest -q
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m pytest -q tests/test_m7a_planning.py tests/test_m7a_shadow.py tests/test_m6b_maneuvers.py tests/test_m6b_obstacles_sdk.py tests/test_m6a_system.py
```

以下规划命令可指定新suffix，不覆盖封存报告；不要把最终集重复运行用于调参：

```bash
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7a_scenarios --split development --suffix _reproduce
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7a_scenarios --split validation --suffix _reproduce
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7a_scenarios --split final --suffix _reproduce
.venv-gpu/bin/python -m tools.m7a_shadow_infer --manifest reports/m7a_shadow_inference_manifest.json --output /tmp/m7a_shadow_reproduce.json
```

原始执行序列如下，其输出排他创建；复现时在新worktree运行或保留原输出后选择新名字，禁止覆盖历史资产：

```bash
.venv-gpu/bin/python -m tools.m7a_shadow_prepare
.venv-gpu/bin/python -m tools.m7a_shadow_audit
.venv-gpu/bin/python -m tools.m7a_shadow_evaluate
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7a_historical
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7a_reference_audit
.venv/bin/python -m tools.m7a_bench2drive_audit
.venv-gpu/bin/python -m tools.m7a_figures
.venv-gpu/bin/python -m tools.m7a_release
.venv-gpu/bin/python -m tools.m7a_finalize
.venv-gpu/bin/python -m tools.m7a_release --seal
```

Frozen production sources are hash-checked, not edited. 原始实验源码SHA、协议SHA、文件及Checkpoint SHA随JSON/最终清单保存；未提交时先执行的辅助审计保留字节SHA，不冒称最终提交是最初执行版本。

## Bench2Drive审查

官方0.0.4固定`7ec25d1c9f7522d923ce5f3420986cef1cb2d956`。文件/URL/git blob/SHA保存在`reports/m7a_sources`，保留许可证。只运行官方传感器验证函数AST摘取的4项测试；没有伪装成完整框架导入/Agent Smoke。SENSORS允许8RGB，六摄符合；Qualifier最多4RGB不符合；Oracle地图传感器被拒绝。

Agent的run_step收input_data/timestamp并返回VehicleControl；ScenarioManager拥有tick/apply_control，不能与HSDrive并存两个Owner。官方collection按当前frame等待；HS历史相机10Hz与20Hz评估循环需明确适配，不能重复旧帧冒充同步。

官方README要求CARLA0.9.15+AdditionalMaps，但vendored ScenarioRunner/CARLA_VER为0.9.13；numpy1.18.4/networkx2.2/Shapely1.7.1等旧依赖未验证兼容现有Python3.10。当前Dev10需要Town11/12/13/15而本地缺失。没有下载地图、训练数据或大型模型。

Driving Score按route completion乘处罚因子后在220固定分母平均；min_speed处罚在本版本unused。Success要求Completed/Perfect且除min_speed外无infractions。必须另核验唯一ID/预期路线集合，官方merge脚本count检查不够。0.0.4完整735,811bytes XML在有限多种官方读取中持续截断/超时；部分Range字节保留为失败材料，不能当完整XML/220完整性通过。停止下载，没有无界修复。

## 验收与失败记录

完整最终测试413 passed、8 skipped、1 xfailed；SDK专项40 passed。8个skip为未配置CARLA的GPU环境合同，SDK另补测；原M6-D已知失败xfail保留。初始合成高速横穿单测错误地把目标放在Ego不会到达的x，修正测试输入后通过，原失败日志保留；此修正早于验证/最终规则冻结。

开发v1、v2、验证/最终、官方树/文件下载错误、历史车辆失败均保留。协议没有按结果放宽。最终状态及下一步见[架构决策](M7A_ARCHITECTURE_DECISION.md)，数据分母和限制见[总结](../reports/m7a_summary.md)。
