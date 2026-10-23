# M6-D 实现与复现

基线 `feat/m6c-safe-stopping-dynamics@d808c21`；新分支 `feat/m6d-signed-stop-control`。全部状态来自 Privileged CARLA 0.9.15，实验Tesla Model3 Physics SHA `3c4932cd550d47893d9051e335bf798424dfd07138ebf150a98820a57af047e6`。没有修改任何M6-C及更早文件，没有网络训练、视觉控制、默认Controller替换或Physics修改。

## 接口

`simulator/m6d_state.py`增加不可变SignedMotionState。原始CARLA世界速度先作`[vx,-vy]`转换，yaw取负；正向轴为`[cos(yaw),sin(yaw)]`，侧向轴为`[-sin(yaw),cos(yaw)]`。分别投影得到前向和侧向速度。路线速度用当前polyline最近段单位切向独立投影，不由车身速度或gear推断。总速度仍为原三维范数，旧OracleState、Planner、Safety和TimedPlan均不变。

字段保存snapshot frame/timestamp、源、坐标系、坡度和实际gear/brake/handbrake。控制状态来自紧邻snapshot的`ego.get_control()`查询；CARLA ActorSnapshot不提供应用控制，不能将这部分称为原子snapshot控制。原`measurement.long_speed`按三维pitch前向轴计算，新字段按要求采用XY轴，二者同时保留。

`simulator/m6d_targets.py`分别保存导航终点有符号切向距离、沿路线剩余量和实际OBB gap减3m。活动目标为navigation_goal、obstacle_constraint或forced_degradation；源、时间戳、原因切换均归档。估计能力保持独立字典，不参与覆盖目标。无真实障碍时障碍距离为null；强制降级没有虚构的停止位置。

F0为了严格保留C正常输出，仍给C传原`remaining_distance`适配量；这个旧接口不被改名成导航Goal。新上下文并列存储两种目标，明确旧适配量的用途。新F1如日后获准必须直接消费目标上下文，本轮没有实现F1。测试固定真实`.871m`和虚构触发`.000428m`，保证二者不别名。

`simulator/m6d_motion.py`将冻结C作为组合对象。正常命令与相同输入下C逐字段一致。观测过期/非单调、持续负速或路线反向等使F0进入原brake=1降级，保护状态锁存，不声称可自动恢复。判据预声明为v_forward<−.05持续.10s，或≤−.15立即触发；**实际暖启动间歇负速绕过该判据，F0保护资格No-go**。原策略保留，不在结果出来后改阈值。严格xfail记录该已知缺口，不计通过。

HoldEvidence仅观察MOVING/SETTLING/HOLD_REQUESTED/HOLD_STABLE/ROLLBACK_DETECTED/RELEASE_REQUESTED/RESTARTING/DEGRADED_STOP。HOLD_STABLE要求应用保持动作、速度≤.02、位置漂移≤.05m、误差增长≤.02m、计划有效且持续≥1s；其状态不覆盖C内部状态、不改变执行命令。静止保持证据不代表已经到达原导航目标。

## 固定协议和实际执行

`configs/m6d/protocol.json`在实验前提交。开发spawn27/126（road4/3）、验证spawn66/170（road20/67），两个重复seed201/202；均Town03，非独立城市。位置简称Road126等不是OpenDRIVE road_id。

32个完整保持/释放试验：每个位置每个重复执行四个诊断设置——冷态service brake=.12/auto/throttle=.10；冷态handbrake/auto/.15；冷态handbrake/manual1/.15；warm stop后service/auto/.15。保持3s、释放6s，再.01普通制动和保持/重启。warm prepare油门.20自然达到名义1m/s，实际速度单独记录，未强设速度。所有原始初始化、prepare、制动和释放记录完整。

四个额外冷态零油门释放最多10s，没有自然回滑，不计恢复成功。开发p27 warm释放已有负速证据后，另行提交`guarded_warm_followup.json`，仅两次相同准备、相同SignedGuard和原紧急brake=1诊断。两次都出现间歇负速，却没有触发保护。未继续追加参数或试验。

38个探针只有固定仿真步长.05s，无墙钟20Hz pacing，无Planner；记录Apply耗时，但不混入驾驶Runtime成绩。`.01`制动阶段若仅有5条idle记录，继承的综合窗口指标无法计算时标Unknown，而非虚构舒适性失败；实际移动的启动/制动原始门槛仍失败。初版分析及纠正说明均保留。phase指标含切换前两条测量，不删瞬态。

开发路线：两位置9m/1m/s、两seed、C/D2/F0共12次。按前置失败退出规则，未跑F1、25m、新封存位置、阻塞/绕行/移除/前车/突然出现的完整矩阵；这些明确NotRun。历史M6-C能力只引用，不能当本轮执行。

固定tick65的8次C/F0故障实验实际上尚未起步，保留为静止故障。`moving_fault_correction.json`预声明再做8次：连续3帧速度≥1m/s且tick≥65才注入，最多500tick；未改变车辆/Controller/Safety。实际均tick79、约1.51–1.60m/s注入。四类故障的原Safety/Worker有效期逻辑不变。

路线Runner是独立命名的M6-C Runner派生；复用原Worker、5Hz/20Hz、.35s有效期、6s安全预测、固定epoch槽位和不补tick策略。常规正常C执行不变。没有亲和性扫参。新增观测计时包含在controller_and_geometry段，不冒充纯算法耗时。

## 环境与命令

使用已有`.venv` CARLA SDK环境和`.venv-gpu` PyTorch/Matplotlib环境；RTX4090可见，但本轮没有GPU训练。首次后台启动CARLA进程退出并发生连接超时，日志保留；通过持续exec session启动同一原脚本成功。没有重新下载PETR或安装GPU依赖。

以下整批命令必须在独立工作副本且无`m6d_*`输出的目录执行；写出采用exclusive create拒绝覆盖。唯一CARLA tick owner，2100端口专用实例。

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
# 另一终端：
.venv/bin/python -m pytest -q tests/test_m6d_contract.py tests/test_m6d_sdk.py
.venv/bin/python -m tools.m6d_probe development
.venv/bin/python -m tools.m6d_probe_analysis development
.venv/bin/python -m tools.m6d_probe validation
.venv/bin/python -m tools.m6d_probe_analysis validation
.venv/bin/python -m tools.m6d_physical_gate
.venv/bin/python -m tools.m6d_campaign
.venv/bin/python -m tools.m6d_guard_probe
.venv/bin/python -m tools.m6d_moving_faults
.venv/bin/python -m tools.m6d_audit
.venv/bin/python -m tools.m6d_results
.venv-gpu/bin/python -m tools.m6d_figures
```

实际38个probe在两个源码版本执行：原36个probe，以及修正hold切换帧anchor计算次序并加入guarded_release的2个followup。原36个probe切换帧distance_to_anchor使用旧anchor，这是记录合同缺陷；hold最终稳定指标使用实际姿态漂移，最终静止结论不因此升级。每trial源码快照是精确复现依据，不能把最后HEAD冒充所有试验版本。

单次可复现开发路线（新名字，不覆盖既有）：

```bash
.venv/bin/python -m tools.m6d_drive --name m6d_repro_001 --arm F0 --scene stop --spawn 27 --length 9 --speed 1 --seed 201 --ticks 1000
.venv/bin/python -m tools.m6d_drive --name m6d_repro_fault_001 --arm F0 --scene straight --fault timeout --fault-on-motion --speed 2.5 --seed 201 --ticks 500
```

测试及封存：

```bash
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m6d_final_full_tests.xml
.venv/bin/python -m pytest -q tests/test_m6d_contract.py tests/test_m6d_sdk.py tests/test_m6b_obstacles_sdk.py tests/test_m5n_sdk.py tests/test_geometry_extended.py::test_six_mounts_against_carla_math tests/test_imaging_rig.py tests/test_session.py::test_settings_restored_after_exception tests/test_snapshot.py --junitxml=reports/m6d_final_sdk_tests.xml
.venv/bin/python -m tools.m6d_preserve
.venv/bin/python -m tools.m6d_finalize
.venv/bin/python -m tools.m6d_release
```

`m6d_intermediate_reports/`保存补充运动故障前的报告，不删除早期误分类记录。最终验收以m6d_acceptance及各顶层模块JSON为准。历史base 10338个文件和50867个继承资产做只读SHA核验。所有试验原Physics SHA逐次断言。最终发布清单包含源码、配置、原始记录、图像、测试和报告SHA；发布commit作为清单后继提交保存在Git，避免自引用哈希。
