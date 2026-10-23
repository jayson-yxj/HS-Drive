# M7-D — Non-Real-Time Oracle Temporal Planning Pilot

基线 `d9899fa`，分支 `feat/m7d-oracle-temporal-replay`。全部新代码为离线 opt-in 工具。默认 Planner/Safety/Controller/Physics、视觉模型和历史结果不变。没有模型训练、视觉控制或本轮规划结果执行。

## 数据与时间合同

`datasets/m7d_temporal.py` 定义 Episode、PlanningInput、FutureReference、PlanningResult。世界坐标沿用 HS 的 X/Y-left/Z-up；车辆位置是 Actor 原点，Actor 障碍坐标是实际 planar bounding-box center，不混用二者。车辆尺寸和偏移来自实际 bounding_box；障碍尺寸采用原 M6B 的 roll/pitch 平面包络，不是渲染 Mesh。

输入保存原 frame、simulation timestamp、实际 snapshot 可用墙钟、带符号车速、当前 Actor ID/姿态/速度/尺寸、当前可用的预声明 Ego 路线。历史仅保存前四帧的时间戳，本轮不估计加速度或意图。负向速度低于 -0.05m/s 超出冻结前进规划器支持域；原值留档，小于该幅度的数值噪声在 request 速度端截为0，绝不称真实静止保护已验证。

`result_clock` 另存计算开始/结束墙钟和实际耗时。离线模型中的结果年龄是输入延迟加计算耗时；它不是声称 CARLA 在这段墙钟内推进了相同时间。所有轨迹从**原 snapshot 时间**起算，超过0.2秒有效期不得正常在线使用。输出没有 actuator adapter。

## 隔离与不可变工件

1. `m7d_inventory` 审计82份真实历史trace，缺字段和不连续记录不补造。
2. `m7d_collect` 是独立 CARLA 车辆/行人脚本，不导入 HS Planner/Controller。唯一 tick owner，0.05s；记录实际状态和命令，二者分开。
3. `m7d_prepare`/`m7d_history_prepare` 把当前 packet 与未来121个实测快照分到不同目录。Planner manifest没有未来文件路径。
4. `m7d_replay` 子进程预载源码/地图，然后安装文件读取审计白名单，仅允许当前packet；未来Evaluator在规划结果落盘后才运行。不持有 CARLA Actor/Client。
5. `m7d_evaluate` 单独进程读取未来；没有反馈选择路径。这是应用数据流隔离，不是对恶意native代码的操作系统安全沙箱。
6. JSON写入使用exclusive `x`，重新执行需使用干净输出目录/独立checkout。不能覆盖本轮失败或历史报告。

`reports/m7d_execution_logs/pre_final_freeze.json` 在验证/最终采集前冻结规划、采集和协议SHA；`m7d_support_audit.json` 核对这些源码未改。开发road3/20，验证76/0，新最终30/1；同Town03，共享脚本模板，不能称跨地图泛化。实际地理间隔另测。

## 冻结规划器与唯一优化

沿用 M7C `generate/evaluate_cached/ConnectedArcRoute/slow_candidate` 和 M7B有限运动族。六秒、0.05s动态假设、3m停车、加减速度/Jerk/完整车身检查不改。道路索引只聚合真实 Driving Lane，保留road/lane/section；当前Actor沿唯一合法后继延拓，遇歧义不任意选择。无当前意图，不从未来倒推。静止车辆仍有加速假设，不伪装成静态几何。

Person 没有预声明运动走廊时明确 Unknown。道路支持检查覆盖所有已枚举未来OBB，不只查中心线。Ego道路审核同时使用lane polygon和原生CARLA waypoint查询。

唯一性能改动为 `diagnostics/m7d_geometry.py` 的**同snapshot多边形复用**：future-body和slow reserve共用Actor多边形；相同候选和时间采样复用Ego多边形。保持原提前失败顺序、最小margin定义和查询数量。snapshot对象身份不符即拒绝；不跨帧缓存Actor状态。M7C GeometryCache只复用完整精确身份的静态几何。

所有205/1662历史候选重跑精确回归；两个开发episode分别进行baseline、每帧cold、连续stream的干净顺序对照。不把同一快照反复预热当连续移动场景。地图索引构建时间单列；实际packet序列均每0.2秒重规划。

## 实际未来评价的边界

每次规划后，对全部候选保留结果，对全部已接受候选和选择结果统计真实未来0.05s planar OBB冲突、最小间距、3m停车及完整车身道路。Ego轨迹插值到实测Actor时间；没有把Actor插值当独立测量。不是精确连续碰撞或物理Mesh真值。既有连续平移SAT加旋转保守界仍用于规划检查。

真实Actor按采集脚本响应，**不会对反事实Ego轨迹重新反应**。单一实测未来安全不证明全假设鲁棒；有限假设鲁棒不证明真实执行。三个字段分别为 `RobustWithinHypotheses`、`SafeForRecordedFuture`、`FeasibleToExecute=NotValidated`。

没有候选仅说明有限库/当前支持域无证书，不能宣称物理上无解。实际未来安全但被拒绝只说明保守差异，不能拿未来调整本轮选择。0.2s窗口高度相关，不用快照数当独立场景数。

## 复现命令

使用基线已有 `.venv` CARLA SDK 与 `.venv-gpu` 测试环境；Shapely2.1.2使用现有隔离目录，无环境升级。完整资产先按release manifest验SHA。以下命令用于新输出工作区；已有报告会拒绝覆盖。

```bash
export PYTHONPATH=runs/m7a_geometry_deps:.
.venv/bin/python -m tools.m7d_inventory
# UE4不能以root运行；端口2400独立实例，Town03须使用Epic。
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2400 Epic
# 在另一终端，顺序执行采集/准备；源码冻结后再采验证和最终。
.venv/bin/python -m tools.m7d_collect --split development
.venv/bin/python -m tools.m7d_prepare --split development
.venv/bin/python -m tools.m7d_history_prepare
.venv/bin/python -m tools.m7d_replay --episode dev_constant_r3 --mode stream
.venv/bin/python -m tools.m7d_replay --episode dev_braking_r20 --mode stream
.venv/bin/python -m tools.m7d_collect --split validation
.venv/bin/python -m tools.m7d_prepare --split validation
.venv/bin/python -m tools.m7d_collect --split fresh_final
.venv/bin/python -m tools.m7d_prepare --split fresh_final
.venv/bin/python -m tools.m7d_batch --split historical_regression
.venv/bin/python -m tools.m7d_batch --split validation
.venv/bin/python -m tools.m7d_batch --split fresh_final
.venv/bin/python -m tools.m7d_evaluate --episode dev_constant_r3
.venv/bin/python -m tools.m7d_evaluate --episode dev_braking_r20
.venv/bin/python -m tools.m7d_batch --split historical_regression --evaluate
.venv/bin/python -m tools.m7d_batch --split validation --evaluate
.venv/bin/python -m tools.m7d_batch --split fresh_final --evaluate
.venv/bin/python -m tools.m7d_equivalence
# 停止自己的CARLA实例及其他计算任务后，顺序执行独立计时。
.venv/bin/python -m tools.m7d_paired --episode dev_constant_r3
.venv/bin/python -m tools.m7d_paired --episode dev_braking_r20
.venv/bin/python -m tools.m7d_support_audit
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_m7a_planning.py tests/test_m7b_dynamic.py tests/test_m7c_geometry.py tests/test_m7d_temporal.py -q
.venv/bin/python -m tools.m7d_report
.venv-gpu/bin/python -m tools.m7d_figures
```

实际运行日志保存在 `reports/m7d_execution_logs/`。Low画质Town03启动崩溃、首次错误import、静态障碍采集碰撞均保留。GPU训练、真实worker故障、M7D车辆闭环、严格20Hz实验为NotRun，不用合成单测替代。

发布前另外执行：

```bash
.venv/bin/python -m tools.m7d_seven
.venv/bin/python -m tools.m7d_leakage_probe
.venv/bin/python -m tools.m7d_release
.venv/bin/python -m tools.m7d_release --seal
```

`m7d_forbidden_future_probe.json` 是合成主动读取测试：在真实Planner进程内尝试打开未来文件，必须被白名单拒绝；不算车辆安全试验。`m7d_historical_seven.json` 单独重跑七条历史漏判，避免只依赖旧报告。

时间字段补充：首轮Planner的 `clock` 在核心函数结束前建立，包含规划检查但不包含完整返回对象构建和归档序列化。不能把它当作IPC交付时间。`tools.m7d_availability` 在**不改冻结plan函数/选择规则**的条件下重新回放全部原始packet，外包测量返回对象完成、原科学结果完整序列化完成；输出 `*_availability.json` 留下原 `core_clock` 并独立保存真实 payload-ready 墙钟。与首轮所有候选/原因/选择逐项等价，明确只是日志口径审计，不是重新选择最终规则。磁盘归档时间及真实IPC交付仍不属于在线能力。

```bash
.venv/bin/python -m tools.m7d_availability
```

新增采集的触发原因：历史日志虽有68份基础字段可用，但未提供本轮未见road30/1的制动/横穿组合，且缺少行人类别运动支持与新的连接片区。没有把68份高度重复历史试验当68个独立新测试。仅新增6段，低于12段上限；第一次Low渲染失败发生在任何Actor采集前，另存失败目录。

函数级成本补充（非干净计时）：

```bash
.venv/bin/python -m tools.m7d_profile
```

前三个开发快照cProfile见 `m7d_runtime_profile_supplement.json`：完整几何检查、精确地图调用、动态检查及诊断对象递归构造均占明显成本。累计调用时间互相包含，不能相加当独立总耗时；不据此追加本轮第二次优化。
