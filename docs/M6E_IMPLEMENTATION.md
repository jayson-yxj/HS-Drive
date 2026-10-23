# M6-E 实现与复现

基线 `feat/m6d-signed-stop-control@ba5ba6e`；独立分支 `feat/m6e-actuation-physics-feasibility`。本轮全部车辆状态是Privileged。原Controller、原Physics、Planner、Safety和M6-D xfail/报告/数据只读。独立P1仅在专用2100实例的独立车辆显式应用，不更改默认Blueprint，不训练网络，不视觉控制，不倒车泊车。

## G1位移保护

`simulator/m6e_guard.py`使用原SignedMotionState，旧OracleState.speed_mps仍为非负范数。

- 原始XY姿态增量投影到前一帧车身前向轴，同时独立投影到路线切向。
- 0.5s窗累计后向位移，保留净锚点后退和至少两个v_forward<−.03m/s证据，不因中间正速清空。
- Risk：净后退≥3mm，或累计≥5mm且净后退≥1.5mm；只提示风险，不改变执行器。
- Emergency：净后退≥8mm，或累计≥12mm且净后退≥4mm，并有负速证据；≤−.15m/s且净位移≥2mm也可触发。原brake=1降级独立优先，锁存至试验结束，不宣称自动恢复。
- Risk只在无新证据≥.5s且前向恢复≥1cm后释放。目标前移、gear或速度符号切换不重置锚点。
- 源/坐标错误、非单调snapshot、缺帧、采样间隔异常或状态过期进入独立故障保护。固定dt=.05s、最大gap=.075s、状态年龄≤.1s。
- 车身前进而路线后退是ForwardRouteContractInvalid，不能误称车身回滑。两套锚点分别在其向前进度满足条件时刷新。

阈值在历史开发p27失败轨迹和<.25mm静止漂移证据上冻结；历史两条11.7mm轨迹仅作回归，不能称盲测。验证位置不再调阈值。新增反向路线单测发现共同锚点重置问题，修复发生在实时Guard试验之前，失败JUnit和源码快照保留。原M6-D严格xfail保持，不修改历史取得Passed。

## 执行接口与数据

`simulator/m6e_actuation.py`与`tools/m6e_probe.py`提供独立诊断：

A：原M6-D raw auto路径；.15油门、.01普通制动、.12静止保持。

B：明确前进1挡及handbrake保持/释放；油门和普通制动水平不变。这是初始化/执行策略组合对照，不能拆成单个gear的因果消融。

C：官方`VehicleAckermannControl`，速度参考按.5m/s²爬升到1m/s，减速阶段按.5下降到0；输入acceleration=.5、jerk=1。使用实际服务器读取的默认Ackermann PID，不调用设置接口调增益。raw初始化结束切换Ackermann时显式清除handbrake。

每组独立生成Tesla。初始1.5s稳定、hold2s、冷启动6s、普通停止6s、hold2s、暖启动6s、停止6s、hold2s。A/B为固定命令，C为闭环速度接口，实际制动初速不同；所有表中同时给初速，不把较小原始减速度直接解释为匹配初速下更好制动能力。

官方0.9.15源码中TargetJerk接收后未用于控制计算；低于.1m/s且目标接近零时FullStop施加brake=1；内部加速度有滤波。仅以外部原始速度、API加速度、位移和Jerk验收，不将输入jerk当实际保证。记录本地SDK和服务器0.9.15、服务器默认PID读回及官方源码SHA；未对服务器二进制内部插桩。

参考：[官方API](https://carla.readthedocs.io/en/0.9.15/python_api/)、[官方Ackermann实现](https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/AckermannController.cpp)。

## P0与唯一P1

P0不变SHA `3c4932cd550d47893d9051e335bf798424dfd07138ebf150a98820a57af047e6`。开发spawn27/126、验证66/170（实际road4/3/20/67），Town03，seed301/302。验证位置未参与本轮参数选择，但被历史项目观察，不能称全新盲测城市。

先12个开发A/B/C全部原始舒适失败，再由开发数据冻结P1：只将四轮max_brake_torque从700降至350N·m。mass、friction、engine、transmission、handbrake torque均不变。选择轮端族是因.01制动产生11–25m/s²强瞬态，研究扭矩缩放敏感性；简化`4*T*.01/(mass*radius)`仅约.041m/s²，已提示不能断言峰值直接由最大制动扭矩解释。

P1为**仿真研究变体**，没有真实Tesla车辆证据。每车生成后保存完整Python公开Physics字段，world-space wheel.position单列并归档，原历史不变签名只排除此世界位置；不伪称有轮胎接触/求解器隐状态。

应用后严格读回：仅声明四个字段等于预期，其他字段完全相同；试验末再读回确认。P1退出前恢复原始Physics，再销毁Actor；后续独立P0新车签名断言防止残留。公共API应用Physics可能重建车辆状态，本轮无“原值重新应用”sham组，因此不把全部差异唯一归因于参数数值。

P1固定预算：4次A完整cycle（开发27/验证66×两重复）；开发126/验证170×两重复×coast/emergency/obstacle×P0/P1=24次。单一变体，不再搜索。紧急非劣化门槛预声明为增加停车距离≤.25m、时间≤.25s，并保留真实OBB≥3m、无碰撞/边界和保持要求。障碍probe初始意图gap5m，实际每帧OBB记录；brake=1是诊断紧急保护，不能计正常NoBypass闭环。

## 测量、Runtime和退出

每次记录原始/Applied命令、gear、handbrake、frame、仿真timestamp、pose、signed速度、三维范数、road slope、API acceleration、Physics完整数据及SHA、源码快照。原始Jerk用真实dt差分，阶段开头包含此前两条样本，保留控制切换尖峰；初始化和Emergency独立报告，不删除日志。Hold针对测得姿态，不能冒充到达导航目标。

所有新probe使用固定50ms墙钟槽位和.05s仿真dt。只有一个tick owner；不重置epoch、不补tick。Apply间隔、Tick、guard/geometry/road和命令阶段耗时、Deadline Miss和Skipped Slots保留。无Planner的probe，Worker/计划年龄/过期正常控制字段为NotApplicable，不能填0伪称安全规划通过。旧5HzPlanner/20HzController/.35s有效期/6sSafety合同不改。

若基础动力学不通过，完整9m/25m/前车/绕行/NoBypass/Worker矩阵NotRun；不为数量强行执行。报告中的Physics安全probe和正常闭环严格分开。离散道路corner检查和末尾dense footprint不是精确连续路缘Mesh验证。

## 可复现命令

现有`.venv`含CARLA；`.venv-gpu`含PyTorch/Matplotlib。RTX4090用于CARLA渲染，无训练。所有输出采用exclusive create，不覆盖历史；整批复现在独立副本/无m6e输出目录执行。每trial source/manifest是实际执行源码，最后HEAD不是所有试验版本。

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
# 另一终端，先确保专用实例无其他tick owner
.venv/bin/python -m pytest -q tests/test_m6e_guard.py tests/test_m6e_actuation.py
.venv/bin/python -m tools.m6e_guard_audit
.venv/bin/python -m tools.m6e_probe development
.venv/bin/python -m tools.m6e_analyze development
# 条件：只在12个开发trial完成且预声明失败依据成立时
.venv/bin/python -m tools.m6e_select_variant
.venv/bin/python -m tools.m6e_sequential
# sequential依次validation、guard、variant及各批分析；不并行tick
.venv/bin/python -m tools.m6e_finalize
.venv-gpu/bin/python -m tools.m6e_figures
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m6e_full_tests.xml
.venv/bin/python -m pytest -q tests/test_m6e_guard.py tests/test_m6e_actuation.py tests/test_m6d_sdk.py tests/test_m6b_obstacles_sdk.py tests/test_m5n_sdk.py tests/test_geometry_extended.py::test_six_mounts_against_carla_math tests/test_imaging_rig.py tests/test_session.py::test_settings_restored_after_exception tests/test_snapshot.py --junitxml=reports/m6e_sdk_tests.xml
.venv/bin/python -m tools.m6e_preserve
.venv/bin/python -m tools.m6e_release
```

已有物理声明文件需在复现副本保留原冻结文件并验证，不通过覆盖生成新候选。`select_variant`输出exclusive create，重复执行会拒绝。P1任务表、来源和失败日志均保留；最终决策见m6e_acceptance及架构决策文档。

后处理保留说明：首次quiet-hold汇总统计锁存risk/emergency帧，不能当作新误触发。最终报告另外统计新状态转换（0），并保留各240个历史锁存帧及初版报告于m6e_intermediate_reports。没有改变Guard或重新仿真。

### 实际加载源码与并行审查限制

`m6e_loaded_source_audit.json`明确区分磁盘快照与Python已经import的版本。development最后一次trial快照中的Runner文件已加入仅P1分支的读回/恢复逻辑，但该批进程仍执行启动时Runner；validation最后四次trial保存了修复Guard磁盘文件，该批仍使用启动时旧Guard仅作旁路观测。**实际函数版本应以该批第一个trial源码快照和启动提交为准**，不能只用每trial生成时的磁盘SHA冒充内存版本。该差异没有改变P0命令路径；guard实际干预和variant批次都在修复之后启动，整批源码一致。后续应在每批启动时一次冻结全部加载模块，避免这种磁盘/内存provenance歧义。

主pytest全套在早期P1仿真期间运行，SDK补测也在同阶段，因此Runtime观察包含测试CPU/GPU任务并发，未将其排除或称作独占负载。无专门GPU训练。Physics分组的wall-clock差异更不能解释为Physics导致。所有实测长尾保留。
