# M6-C 实施、测量与复现

基础 `feat/m6b-oracle-maneuvers@f047ade6d7802205abadcde4363a57e0a1d00981`；独立分支 `feat/m6c-safe-stopping-dynamics`。保留历史默认Controller/Planner/Safety、Tesla Physics、所有感知模型。所有新增内容显式Opt-in、来源Privileged，无视觉控制、无Occupancy训练、无PETR恢复、无倒车泊车。

## 1. 几何合同

`simulator/m6c_geometry.py` 使用当前快照及 `hs_world_x_y_left_z_up_m`。CARLA原始measurement保留Y右，转换后才参与HS几何。目标参考Ego Actor原点，障碍间距参考带本地偏置、roll/pitch投影后的完整Actor OBB。OBB是保守盒参考，不是渲染/路缘Mesh。

| 字段 | 定义 |
|---|---|
| route_remaining_s | 路径总弧长减当前投影弧长，末端超出时保留负数；单位m，s不是秒 |
| goal_signed_distance_m | 目标减当前位置沿目标yaw切向的点积；正为尚未到达、负为越过 |
| goal_lateral_error_m | 同一位移沿目标法向投影；不是有限polyline最近距离 |
| goal_heading_error_rad | wrap(goal yaw − current yaw) |
| obstacle_obb_clearance_m | 当前路径相关障碍与Ego平面OBB的最近欧氏间隙；接触/相交为0 |
| required_safe_gap_m | 固定3m，不缩小 |
| available_clearance_to_safe_stop | min(当前合法路径剩余量, 路径相关OBB间隙−3m) |
| estimated_stopping_distance_m | 指定动作的实测包络上界；不支持时null，不伪造0 |
| stopping_margin_m | available − estimated；负为当前动作不具足够包络余量 |

路线相关性沿用冻结`clearance()`判据，所有障碍仍送入原六秒Safety。绕行侧向经过Actor不要求3m停车间距，但必须保留完整OBB碰撞检查。末端1m保留与M6-B辅助点相互抵消；不移动StopGoal，不缩小车身。Planner仍使用原弧长/尺寸模型，报告其原始可用距离与真实OBB差异，不宣称已经改写Planner合同。

历史失败台账逐帧包含速度/有符号速度、剩余距离、OBB间隙、原coast预测、重算Planner available、mode、command/applied/gear、raw a/j、坡度、计划原始timestamp及接管原因。历史C无有效制动距离预测的字段明确null。重算Planner距离不是冒充历史保存的中间值。

## 2. 有限真实车辆探针

`configs/m6c/protocol.json`先冻结，再执行。开发道路27/126，验证道路66/170，两个固定seed；112组探针全部自然启动，不使用set_target_velocity。

- 自动挡/手动固定前进1挡，Throttle .10/.15持续6s，再coast/hold。
- 自然达到1/2.5m/s后，Brake 0/.01/.05/1，最多20s；到<.02m/s再hold至少2s。
- 自然启动后完全制动再起步的warm initialization；1m/s附近.10油门4s的维持实验。
- 初始/实际gear、控制请求/已应用值、速度、位姿、API加速度、仿真timestamp、坡度和车辆pitch全部保存。Physics文件逐次SHA验证。

manual1仅为响应诊断，未进入D，不修改Physics。自然达到目标速度可能因启动突变跳过设定值，因此分析使用**实际初速度**，不将名义1m/s当精确初始化。未停稳的20s记录是右截尾下界，不算完成停止。坡度使用OpenDRIVE前方2m高度差，车辆pitch独立保留；有限几处轻坡不是全坡度覆盖。

全程raw和250ms数据并列；phase分析保留prepare/warm stop，不能移除启动尖峰以通过舒适性。原始Jerk阈值5、运动P95阈值2.5、raw decel1.8不变。挡位与瞬态相关不证明唯一因果机制。

## 3. 包络与独立验证

`tools/m6c_fit_envelope.py`只读取development样本拟合，先写冻结模型，再评估validation。

A为旧近似coast积分。B按预声明速度bin与动作保存最大剩余停车距离/持续时间，并取跨较低速度bin的单调最大值；内部空bin只从更高已观测bin填充，最大支持速度以上不外推。C增加：

`distance += v*0.35 + 0.5*g*max(0,-grade)*0.35² + 0.5*g*max(0,min_dev_grade-grade+0.005)*duration² + 0.15`

0.35s是保守响应预算，0.005为坡度不确定性，0.15m为未建模距离余量，均预声明。它是经验上界假设，不是分布无关证明。coast在grade<−0.005直接不提供有限保证，原因是开发下坡实测在预算内不停止；验证同样保留unsupported。

报告分开有限覆盖数、边界失败、截尾有限预测错误、停止误差。A在制动探针上只作不同动作的参考，不可将该混合分母直接当公平coast模型优劣。相同动作、相同距离定义才是主比较。开发最高速度区间、轻坡范围和少量重复不足以认证任意状态停车。

## 4. C/D及E退出

C严格复用M6-B。D调用同一C前馈、gear与横向控制，只在停车距离决策上增加包络触发提前SETTLE；若路径相关障碍缺少足够保守停止余量，标记 `InsufficientStoppingMargin` 并进入原立即brake=1路径。保留5Hz规划/20Hz控制、0.35s原始输入有效期、六秒Safety。

D中的brake_001包络是“现有车辆物理可用制动能力”参考，**不表示D自动执行.01普通制动**。实际是否靠正常控制完成、是否借助紧急制动必须单列。D可以因估计过保守提前停住；这不算目标精度通过。

初版开发wrapper误加.01执行动作，与D单变量要求冲突。两次原始结果永久保留在`m6c_development_deviation.json`列名，排除出D-only消融；删除该执行动作后重新开发验证，再冻结正式矩阵。E未运行：D开发改善间距但均需紧急保护，探针没有提供满足raw舒适门槛的执行策略依据；不继续参数搜索。

## 5. 正式评测与Runtime

正式停车包含27/126/170三位置、9m/1m/s与25m/2.5m/s、两seed的C/D，以及部分A历史基线。B只引用历史M6-B，不伪装同批新数据。正常直道、弯道、路口、前车停止/重启、双阻塞、合法绕行、移除、突然出现及四种行驶故障单列。突然出现按首次实际可见快照OBB计算可用距离，不用spawn意图替代实际距离；不可行也保留3m失败。

D最后间距≥3m、无碰撞不等于完整安全停车：还须前置可行、道路合法、无运动中Safety接管、正常舒适性与停车目标满足。记录all OBB gap及路径相关gap，绕行不使用停车终点误判中途侧向通过。

单调墙钟固定epoch+50ms槽位，不reset deadline、不补tick。每行保存server原始platform timestamp、snapshot、submit/poll、Safety、geometry/controller、apply、tick和主进程CPU时长。Worker另存request等待与实际执行。Runtime只比较同一行驶任务C/D，两重复600tick；不做静止减运动的开销归因。

外部低频observer记录CARLA/主进程/Worker的`/proc` CPU、schedstat、系统load和nvidia-smi，不拥有tick。proc字段与服务端平台时间只能辅助定位资源/调度关联，不能凭此宣称UE/PhysX内部根因已确定。严格实时独立保留，全部运行最坏值不可由短测覆盖。

## 6. 实际复现命令

服务器CARLA 0.9.15，2100端口，现有环境不重装；专用实例仅一个tick owner。所有输出目录独占创建。已存在的正式批次名不能重跑覆盖；整批复现用干净副本输出目录，单次用新name。

```bash
.venv/bin/python -m tools.m6c_probe
.venv/bin/python -m tools.m6c_fit_envelope
.venv/bin/python -m tools.m6c_probe_analysis
.venv/bin/python -m tools.m6c_failure_audit
.venv/bin/python -m tools.m6c_drive --name m6c_repro_blocked_001 --arm D --scene blocked --spawn 126 --length 100 --speed 2.5 --seed 101 --ticks 1400
.venv/bin/python -m tools.m6c_drive --name m6c_repro_short_001 --arm C --scene stop --spawn 170 --length 9 --speed 1 --seed 101 --ticks 1000
.venv/bin/python -m tools.m6c_campaign stop
.venv/bin/python -m tools.m6c_campaign maneuver
# observer与runtime并行，observer没有tick调用
.venv/bin/python -m tools.m6c_resource_monitor --until reports/m6c_runtime_execution.json --output runs/m6c_logs/resources.jsonl &
.venv/bin/python -m tools.m6c_campaign runtime
.venv-gpu/bin/python -m tools.m6c_analyze
.venv-gpu/bin/python -m pytest -q --junitxml=reports/m6c_full_tests_repro.xml
.venv/bin/python -m pytest -q tests/test_m6c_stopping.py tests/test_m6b_obstacles_sdk.py --junitxml=reports/m6c_sdk_tests_repro.xml
```

环境差异：`.venv`有CARLA、无Matplotlib；`.venv-gpu`有Matplotlib/PyTorch、无CARLA。仿真与绘图使用各自现有环境，无安装或GPU训练。每trial保存执行源码副本和SHA，报告git_head若为父提交，以冻结源码为准。

## 7. 执行中发现的几何接口错误与重跑

原D包装器将包络触发距离作为C的`remaining_m`传递，导致真实尚余0.871m时HOLD收到0.000428m。这违反真实目标距离合同，**其全部D结果仅作为失败证据，不进入有效候选通过分子**。没有移动GT目标，错误在Controller接口。

修复后仅覆写`coast_distance`的减速触发阈值，真实remaining不变；新增回归测试保证.87m尚未到目标时不会因为近零速误入HOLD。没有改增益、Physics、目标、3m间距或门槛。执行中的当前child正常结束后停止原批次调度；保留原计划/已执行日志。`tools/m6c_resume_matrix.py`按完全相同预声明任务重跑全部D，name以D2结尾，同时完成尚未执行的C/A。D2是合同修复版本，不是E执行策略；E仍NotRun。

因该bug是在正式结果审查中发现，本批结果不能称作完全未经观察的盲测。即使修复后有改善，也只能作为工程回归，下一阶段需新的独立位置才能做候选升级。报告列出invalid D，与修复版分开，不删除其碰撞/长尾/失败总账。

包络消融补充：初始B评估与C共用了“下坡coast不支持”的类别规则，因此并非完全无坡度B。`m6c_envelope_strict_ablation.json`另外执行严格B（忽略所有坡度，仅使用同一冻结实测表）；不改模型参数、C估计或Controller。严格B在Val的coast8/8低估（4次未停稳仍给有限估计），制动三类各4/8低估。C仅4/8 coast提供有限支持，这4次无低估；其余coast明确Unknown。原报告保留，主结论使用严格、同动作消融，不能把拒绝提供估计的样本计成停车成功。
