# M7-E — State-Anchored Trajectory Feasibility & Planning Latency Gate

基线 `6ed646c`，新分支 `feat/m7e-state-anchored-planning`。新增功能全部opt-in；默认Planner、Safety、Controller、Physics、感知模型与历史资产只读。没有网络训练、视觉控制或本轮规划输出驱动车辆。

## 唯一连接算法

`diagnostics/m7e_trajectory.py` 保留原E_slow_follow，以独立F候选增加一个五次Hermite **二维向量**曲线。参数u从0到1；位置边界是实测XY与路线前方12m位置；一阶导数分别是长度12m乘实测航向和终点路线切向；初始二阶导数假设为0，终点二阶导数按路线局部曲率和法向构造。求解唯一3×3线性方程获得三至五阶系数，不搜索其他规划器。

曲线按0.02m名义参数密度采样计算弧长，再将原0.5m/s目标、4.5s余弦减速曲线的解析行程映射到弧长。轨迹包含t=0，随后61点覆盖六秒，保持原0.1秒轨迹采样。曲线公式在t=0满足实测位置/航向/速度，**没有强制覆盖第一个点然后跳回中心线**。未完成12m回归时允许保持合法的部分回归；这不意味着六秒必须已经居中。

初始加速度、Yaw Rate不在当前可信输入合同中：明确假设计划初始加速度和曲率均0，不声称测得或与实车初始加速度连续。支持域预声明为横向/路线距离≤0.8m、航向差≤0.2rad、非负前向速度≤5.6m/s。任何负实测速度都拒绝F；旧A保留M7D的小负速截断，公平对照不改历史。路线不足、奇异曲线、曲率超限、积分错误返回 `InitialStateConnectionInfeasible:<reason>`。成功构造不等于可接受，后续完整几何/道路/动态/停车余量仍可拒绝，报告分别统计。

原速度、加速度、减速度、内部Jerk、0.10曲率、六秒、0.2秒有效期、3m语义全部保留。新F执行与原E相同的保守停止余量检查；不新增预测假设。完整车身扫掠和精确地图查询保留，连接guide也从实测起点开始，不能用虚假的中心线guide代替。

## 初始状态归因

`tools/m7e_attribution.py` 从原始当前packet计算路线投影进度、带符号横向误差、航向误差、车身道路有效性、实测速度和原始时间。逐候选记录起点位置/航向/速度、第一段积分/曲率/车身、动态碰撞/停车余量、支持域和未知Actor走廊。类别可重叠，不把所有失败归因于投影。

审计使用严格1e-6的误差统计，同时保留旧检查器的实际拒绝原因；旧检查器`np.allclose`含相对容差，所以误差计数和原`initial_state_mismatch`计数不同。这是保留旧行为，不是新放宽门槛。新F另外验证t0误差≤1e-12。速度原值与旧request截断差异分列。

## 一次静态区域缓存

`diagnostics/m7e_region.py` 只缓存 `union(polygons).buffer(-inset)` 的不可变Shapely2几何。key包括Map SHA、完整Road/Lane/Section集合、每块原始polygon WKB及inset。洞、几何变化、车道集合变化和地图变化均不能错误命中。没有缓存Actor动态结果、过期通过状态或修改车身尺寸。

`tools/m7e_equivalence.py` 重跑历史205场景/1662候选，逐项比较原与缓存区域、几何原因、动态碰撞ID、Unknown、3m和选择；F不列入等价声明。历史七条风险另行重跑。缓存只有这一个新增类型，不做第二次性能重构。

## 开发、冻结与新最终

先使用M7D开发road3/20的122个快照，候选参数和非平凡增益门槛在协议中预声明。开发接受后记录 `pre_final_freeze.json`，之后不改连接/规划/缓存/采集协议。M7D其他六段包括原最终road30/1全部仅作历史回归。

新采集三段18秒：road65制动/重启、road67恒速、road28横穿。三者从地图长度/曲率审查选定，与本轮开发不同道路；这些地图道路过去用于合成回归，不称未见城市。初始Ego在spawn时设置小幅横向/航向扰动，settle后只记录实际状态，没有运行中传送或速度重置。六秒未来只供Evaluator，独立采集脚本不导入M7E Planner。Person没有合法当前走廊仍Unknown。

当前/未来独立目录和子进程白名单继承M7D模式。读取审计安装在源码/静态地图预载之后，仅允许当前packet；主动未来读取测试必须拒绝。应用层隔离不是恶意native程序安全沙箱。初次NumPy延迟导入被白名单阻止，修复为启动时显式导入，未放宽数据读取名单，失败日志保留。

## 完整时效口径

保存原simulation timestamp、原input availability墙钟、新读取开始、候选核心计算与全部诊断对象构造、JSON序列化后payload可读取时刻。研究年龄=从当前packet读取开始至完整科学结果bytes可用的实际墙钟耗时，绝不重盖轨迹timestamp。原有效期0.2s；过期标记保留，不能正常在线使用。

分项包括地图/索引准备、snapshot解码校验、request构造、道路查询、假设生成、未来body支持、候选生成、F生成、每候选几何/动态SAT/停止余量、选择、对象创建、序列化。地图准备单列不混入稳态；完整科学payload不包含其自身末尾计时的诊断外壳与后续磁盘归档，非IPC或真实控制时延。

干净配对停止CARLA并等待所有其他实验结束。连续原始输入逐帧运行，两个开发序列顺序分别为 A_uncached→A→B 和 B→A→A_uncached，限制顺序漂移；每臂新进程、新初始cache，没有重复相同帧预热。A_uncached是冻结M7D候选/多边形复用而无本轮区域缓存；A仅加缓存，B再加F。完整有效比例和每候选成本均保留，不仅报几何核心时间。

## 安全含义

`RobustWithinHypotheses` 是冻结有限运动族中的证书；`SafeForRecordedFuture`是实测0.05s planar OBB、停车余量及道路检查；`FeasibleToExecute=NotValidated`。采样参考不是精确连续Mesh碰撞，实际Actor不响应反事实Ego；车辆脚本运动不算新Planner跟踪成功。连续快照高度相关，单列Episode和快照分母，不报告伪独立置信区间。

## 复现

沿用已有SDK环境 `.venv` 与全量测试环境 `.venv-gpu`，不升级原依赖。Shapely2.1.2在 `runs/m7a_geometry_deps`。所有JSON独占创建，需独立输出工作区重跑，不能覆盖现有报告。原始数据/未来/输入/回放/测试/失败日志与源码SHA见release manifest。

```bash
export PYTHONPATH=runs/m7a_geometry_deps:.
.venv/bin/python -m tools.m7e_attribution
.venv-gpu/bin/python -m pytest tests/test_m7e_trajectory.py tests/test_m7d_temporal.py -q
.venv/bin/python -m tools.m7e_campaign --episodes dev_constant_r3 dev_braking_r20 --arms B
.venv/bin/python -m tools.m7e_campaign --episodes dev_constant_r3 dev_braking_r20 --evaluate
.venv/bin/python -m tools.m7e_development_gate
# gate通过且源码参数冻结后，独立终端启动Epic实例（UE不能root）。
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2400 Epic
.venv/bin/python -m tools.m7e_collect --split fresh_final
.venv/bin/python -m tools.m7e_prepare --split fresh_final
.venv/bin/python -m tools.m7e_campaign --episodes val_restart_r76 val_connection_r0 final_braking_r30 final_crossing_r1 historical_final_lead_s101 historical_final_bypass_p126_s101 --arms B
.venv/bin/python -m tools.m7e_campaign --episodes val_restart_r76 val_connection_r0 final_braking_r30 final_crossing_r1 historical_final_lead_s101 historical_final_bypass_p126_s101 --evaluate
.venv/bin/python -m tools.m7e_campaign --episodes m7e_final_braking_r65 m7e_final_constant_r67 m7e_final_crossing_r28 --arms B --input-root reports/m7e_inputs
.venv/bin/python -m tools.m7e_campaign --episodes m7e_final_braking_r65 m7e_final_constant_r67 m7e_final_crossing_r28 --evaluate --input-root reports/m7e_inputs --future-root reports/m7e_evaluator
.venv/bin/python -m tools.m7e_equivalence
.venv/bin/python -m tools.m7e_seven
.venv/bin/python -m tools.m7e_support
.venv/bin/python -m tools.m7e_safety
# 停止自己启动的2400实例，等待其他任务结束再运行计时。
.venv/bin/python -m tools.m7e_pair
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_m7a_planning.py tests/test_m7b_dynamic.py tests/test_m7c_geometry.py tests/test_m7d_temporal.py tests/test_m7e_trajectory.py -q
.venv/bin/python -m tools.m7e_report
.venv-gpu/bin/python -m tools.m7e_figures
.venv/bin/python -m tools.m7e_release
.venv/bin/python -m tools.m7e_release --seal
```

构造成功后的道路/动态风险拒绝仍保留原始原因；报告将F几何运动学拒绝汇总为 `InitialStateConnectionInfeasible:<原原因>`，与动态风险、停车余量和Unknown分开。局部曲线密集采样不是数学上对所有连续点曲率的精确证明；原分段保守车身扫掠继续保留，实际车辆执行仍NotValidated。新最终实际航向变化约0.05–0.066rad，仅覆盖轻微曲线，不能声称强弯道泛化。

报告后处理和补充图：

```bash
.venv/bin/python -m tools.m7e_finalize
.venv-gpu/bin/python -m tools.m7e_final_geometry_plot
```

先完成`m7e_release`只读历史SHA验证，再运行finalize；最后`m7e_release --seal`。补充图选择新最终中最大横向偏差的已选F用于展示，未用于规则选择或调参，见 `m7e_final_offset_connection.png`。初次finalize在历史验证未完成时退出，生成流程失败独立保留，不影响实验结果。
