# M6-B 实施与复现

基线 `feat/m6a-system-baseline@7ec866ff4ec5fe621c506056f41750651adcc4ab`；独立分支 `feat/m6b-oracle-maneuvers`。所有输入和实验均为 **Privileged**。旧控制器、Physics、视觉模型和默认入口保持原样。本轮无视觉推理驱动车辆、无神经网络训练、无 PETR 安装/推理。

## 合同与实现

- `simulator/m6b_motion.py`：`StopGoal` 使用 HS world X/Y左、米、弧度，参考 Ego Actor 原点；目标时间戳是创建时的仿真时间。0.5m、5°、0.08m/s、连续1s门槛预先固定。导航路径终点只有通过完整车身道路检查后才作为停止位姿。它不是车头/保险杠位置。
- `goal_route()` 为旧 Planner 追加1m切向辅助点，抵消旧 Planner 对导航终点的1m内部扣减；实际目标仍为原终点。没有修改旧 Planner；障碍物间距仍使用原 `clearance()` 和3m停车余量。规划速度/加速度/内部Jerk仍由原 MotionPlanner 产生。辅助点不赋予越过目标的许可。
- A 用旧 Planner/RouteController；B 仅显式目标及当前位姿距离；C 在 B 上用实验性低速控制器。三个实验臂共用新墙钟记录器；A不是旧M6-A调度器逐位复现。各配对同一路线、速度上限、Tesla Model3、Physics、仿真步长。
- C 的 `APPROACH/CREEP/SETTLE/HOLD/EMERGENCY` 保存转换原因、速度和距离。CREEP不抬高安全轨迹目标速度；SETTLE释放油门，HOLD仅在近目标、测得速度≤0.02m/s进入；目标前移可恢复。任何无有效规划/故障均立即 brake=1，绕过舒适限制。该控制器为候选，**不是已经标定成功的逆动力学**。
- 高速前馈系数来自 M4.7 的2–5.5m/s拟合；低速 coast distance 使用近似 `dv/dt=-(0.2+0.4v)` 积分。其在不同道路/倾角上不精确，最终实验不再调参。C仍存在低速启动非线性、服务制动突变和目标越过风险，不改默认。
- `simulator/m6b_bypass.py`：确定性五次横向偏移，5–30m进入相邻车道、55–80m回归；只研究100m静态道路样例。要求双方 Driving、同方向、非路口、Broken标线且允许去/回变道。左右候选分别审计，无候选返回 `NoFeasibleBypass` 并沿原路线停车。
- 候选检查完整旋转障碍OBB，车身长度/宽度额外0.2/0.15m，路径每0.25m、车身网格0.2m；静态碰撞额外膨胀覆盖平移/旋转采样间隙。曲率≤0.1/m（2.8m轴距下约15.6°，低于车型转角上限）；实际转角和跟踪仍另行评估。道路网格检查不是连续路缘Mesh证明。障碍使用快照Actor transform与BoundingBox本地偏置、局部旋转、roll/pitch的平面包络。
- `simulator/m6b_worker.py` 保留原六秒 Safety、5Hz工作请求、20Hz控制和原始时间戳 .35s 过期规则。未修改 `motion_oracle.py/m47_motion.py/m48_motion.py/m5a_async.py`。绕行是在执行前选择路径，每次在线轨迹仍调用旧 Safety。
- `tools/m6b_drive.py` 是唯一CARLA tick owner，Worker仅使用离线OpenDRIVE。墙钟使用固定 `epoch+slot*.05`；超期只记录跳过的墙钟槽位，不补发tick、不重置deadline。跳槽导致的长控制间隔仍算失败，不能因此宣称20Hz。

## 测量口径

日志每行含 `wake_up→snapshot_end→submit_poll_end→safety_end→controller_end→apply_control→apply_end→tick_start→tick_end`，计划输入的原始仿真/单调时间、frame、已应用控制和本轮请求控制。Measurement是apply之前的快照，下一行才反映本轮命令效果。磁盘写入在试验循环后；循环内JSON记录缓存在内存。相邻tick间还含记录构造与休眠，离线报告分开核算，不能将其忽略。

速度导数使用实际仿真时间差，原始 acceleration、jerk 不去异常值；250ms指标是五点速度滑动平均后的导数，另列而不代替原始峰值。运动jerk包含三帧中任一 |v|>0.2m/s 的窗口。普通命令/降级命令效应另列；从命令转换到物理效应的归因不是引擎内部因果证明。

停车位置、航向、稳定保持、道路、碰撞、舒适性分别验收。普通导航保持原无任何lane invasion、CTE P95≤0.5m和完成率条件，再加终点要求。仅审核过的绕行允许Broken事件；Solid始终违规。没有独立路缘Mesh，`curb=Unknown`。OBB离线检查包含偏置与roll/pitch，仍是近似几何。

## 实验设计及真实限制

`configs/m6b/protocol.json` 记录预声明门槛。开发最多两次控制器版本；最终不根据结果反调。停车24组：2道路位置 × (9m/1m/s、25m/2.5m/s) × 2重复 × A/B/C。不是长度与速度的完全析因；重复seed也不是新的独立世界。

绕行/拒绝/移除在Town03的126、170两个spawn；直道、弯道和路口复用历史路线；前车常速、停止、重启夹具明确为Privileged。原拒绝试验两次因刚spawn的第二障碍未进入snapshot而退出；保留原失败目录，修复唯一tick owner初始化后使用新名字补测。无任何最终控制参数改变。

初始第65tick故障注入发生在尚未起步时，不当成行驶停车验证；补充第350tick行驶中故障及绕行Worker失败。补测是诊断覆盖修正，不改成功门槛。

Runtime四臂各600tick×2重复，第二重复反序：A静止基础tick，B旧Oracle闲置系统，C主进程CPU2/3，D主进程2/3与Worker4/5。没有独占CPU/共享视觉GPU试验；A静止而B/C/D运动，不能把差别完全归为Worker开销。短测无严格实时认证效力。

## 可复现命令

所有生成器采用独占新目录/`open('x')`，拒绝覆盖历史。下面历史批次名已存在；在新的工作副本与报告目录中执行整批复现，或用新的 `--name` 运行单次。必须专用CARLA实例，无其他tick owner。

```bash
# 本机已执行；安装目录按服务器实际路径
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
# 单次新的复现目录；绝不重用已存在name
.venv/bin/python -m tools.m6b_drive --name m6b_repro_stop_C_001 --arm C --scene stop --spawn 170 --length 9 --speed 1 --seed 101 --ticks 1500
.venv/bin/python -m tools.m6b_drive --name m6b_repro_bypass_001 --arm C --scene bypass --spawn 126 --length 100 --speed 2.5 --seed 101 --ticks 2200
.venv/bin/python -m tools.m6b_drive --name m6b_repro_fault_001 --arm C --scene bypass --spawn 170 --length 100 --speed 2.5 --fault crash --fault-tick 350 --ticks 600
# 以下在干净的报告输出目录，顺序执行
.venv/bin/python -m tools.m6b_audit
.venv/bin/python -m tools.m6b_calibrate
.venv/bin/python -m tools.m6b_campaign stop
.venv/bin/python -m tools.m6b_campaign maneuver
.venv/bin/python -m tools.m6b_campaign runtime
.venv/bin/python -m tools.m6b_supplement
.venv/bin/python -m tools.m6b_analyze
.venv/bin/python -m tools.m6b_body_audit
```

`m6b_body_audit` 使用保存的 `runs/m6b_logs/Town03.xodr` 和 `reports/m6b_body_reference.json`，离线执行；不tick。地图可从专用服务器的 `client.get_world().get_map().to_opendrive()` 保存，只读参考体盒来自 `ego.bounding_box`。原始地图/日志冻结索引随交付保存。

```bash
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m6b_full_tests_repro.xml
.venv/bin/python -m pytest -q tests/test_m6b_maneuvers.py tests/test_m6b_obstacles_sdk.py --junitxml=reports/m6b_sdk_tests_repro.xml
```

每个trial保留实际config、goal、physics、原始/选择route、trace、plan、state transitions和执行时源码副本/manifest；报告git_head可能是未提交新增代码之前的父提交，**执行源码副本及SHA才是该次运行的准确版本**。新CLI故障选择和场景初始化补丁不改变已冻结纵向控制。最终源码索引、测试与资产清单见 `reports/m6b_acceptance.json` 及 `reports/m6b_asset_manifest.json`。

## 保留状态

低速停车不是自动泊车。倒车入库、换挡规划、车位识别/边界、复杂交通/红灯路权未验证。PETR仍停在官方资产不可得，未执行模型能力判断。本轮不调整任何感知或安全门槛。

注意旧 `route_projection` 返回的CTE实际是到有限polyline的最近距离，越过末端时包含纵向超调。报告另外计算投影段切向的横向误差，不把1m的末端超调描述成横向越界。正常路线的原polyline误差门槛仍保留。计划采样内部Jerk与跨规划重置、控制指令与实际运动是不同口径，不互相替代。
