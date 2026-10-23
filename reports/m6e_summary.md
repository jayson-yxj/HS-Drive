# M6-E — Rollback Guard Hardening & Actuation/Physics Feasibility

**总体Partial / CandidateNoGo。** G1修复已知间歇回滑漏检并完成有限实际保护验证；原Physics的三种执行接口仍未满足正常舒适门槛。唯一轮端制动P1没有一致收益，且延长紧急停车距离，明确No-go。没有开始新Oracle闭环Candidate或第二Physics变体。

基线`feat/m6d-signed-stop-control@ba5ba6e`；独立分支`feat/m6e-actuation-physics-feasibility`。所有实际车辆实验为Privileged CARLA0.9.15、Tesla Model3、Town03。原Controller、Planner/Safety、Physics-P0、视觉Shadow以及M6-D历史xfail只读保留；没有视觉控制、神经网络训练、倒车泊车或main合并。

## 实验与冻结

实际完成**56个车辆试验**：24个P0执行接口cycle、4个Guard保护探针、4个P1 cycle、24个P0/P1滑行/紧急/障碍配对。每次独立生成车辆，两个重复seed301/302；开发spawn27/126、验证66/170，实际OpenDRIVE road4/3/20/67。验证位置未用于本轮参数选择，但属于历史观察过的Town03位置，不称独立城市或全新盲测。

P0开发12次全部舒适失败后，查看本轮验证结果之前冻结P1，只修改四轮max_brake_torque=700→350N·m。无其他参数族、第二候选或结果后阈值搜索。完整Physics公开字段、轮端参数、世界轮位置、实际命令与Applied Control、原始位姿/速度/API加速度、时间戳、源码快照和SHA都保存。

| Physics | 不变参数签名SHA |
|---|---|
| P0 / FrozenBaseline | `3c4932cd550d47893d9051e335bf798424dfd07138ebf150a98820a57af047e6` |
| P1 / Experimental | `d2440c6c44f919556f552b4cb868d3b06b3bb111d26ccdd0b3541bfee93affb7` |

world-space wheel.position另外保存，不纳入历史不变签名。每次P1应用后逐字段读回检查，仅四个声明字段变化；末尾再次核验，退出前恢复P0，再销毁Actor。后续新P0签名断言通过。

## 1. G1能否识别间歇回滑且避免抖动误触发？

**有限证据支持，完整独立正例泛化仍Partial。** G1将Risk和Emergency分开：0.5s累计位姿后退、净锚点回退及至少两个真实负速样本交叉判断。单帧微小负速度不会直接全刹；目标前移、gear切换和中间正速度不清空风险。车身前向与路线切向独立，时间戳异常、丢帧、非单调独立降级。

- 原M6-D两条约11.7mm失败轨迹：**2/2回归检出**。这是开发回归，不是新盲测。
- 历史72个测量筛选的静止保持片段：新Guard误触发0。
- 实际开发位置27两次：先Risk，约0.30s后在净锚点后退**8.16mm**时Emergency，决策观测到Apply约**0.60–0.61ms**，均停稳并保持。先前总位移、锚点定义和保护后位移均保存，不能将8.16mm误称完整过程最大回退。
- 实际验证位置66两次：释放段仅约**0.49/0.85mm**小回退，没有Risk/Emergency，随后向前行驶；属于小扰动阴性对照，**不计危险回滑检出成功**。
- 全部新试验88个静止保持片段没有新触发。6个最终保持段延续此前真实回滑锁存，各40帧；锁存风险/紧急状态各240帧单列，不当作240次误触发。

14项模块单测包括近零噪声、位移不支持的交替负速、侧滑、坡道静止、正常起步、目标前移、过期/丢帧、gear不一致和路线反向。扩展单测曾发现车身锚点刷新错误清除路线反向证据，已在实际Guard试验前拆分锚点；阈值不变，失败JUnit和源码快照保留。

保护是原brake=1紧急路径，保护段Jerk仍约73.4m/s³，不能称正常舒适停车。Guard锁存后的自动释放未开发。

## 2. 原Physics下更换接口是否改善启动和制动？

部分指标变化，但**三种接口均0/8达到完整原始舒适性**。

| 接口 | 主要观测 | 结论 |
|---|---|---|
| A：M6-D raw auto，.15油门/.01制动 | 冷启动约3.6s；普通制动大峰值持续 | No-go |
| B：固定前进1挡、handbrake明确释放 | 冷启动约0.55–0.65s；Jerk及普通制动仍超限 | 延迟改善，不能升级 |
| C：官方Ackermann | 冷启动约3.5s；部分制动峰值下降，但仍反复超限 | No-go |

官方0.9.15源码中TargetJerk只赋值、未进入控制循环；FullStop低速条件会施加brake=1，内部加速度还做滤波。因此设置jerk=1不代表实际满足1m/s³。本轮实际服务器PID参数只读、不调参，所有外部验收使用原始运动数据。[官方实现](https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/AckermannController.cpp)

开发A普通制动最大减速度约21.83–24.70m/s²；验证A约19.17–20.12。验证C约10.53–15.02，Jerk仍约244–322m/s³。C制动初速和输入轨迹不同，不能据较小峰值宣称严格同初速单变量优越。表中每阶段初速、跟踪RMSE、Jerk P95、Applied命令完整披露。

## 3–4. 是否需要独立Physics版本？哪个参数族有改善？

本轮P0失败支持一次**独立敏感性研究**，不证明必须改Physics才能解决问题。唯一研究轮端最大制动扭矩族；**没有得到稳定、可复现的舒适改善证据**。

P1普通制动cycle仍0/4通过；开发普通制动最大减速度约21.73–24.83m/s²，验证约22.20m/s²，启动冲击也未消失。四轮扭矩减半并没有将峰值减半。简单`4*T*.01/(mass*radius)`在P0仅约.041m/s²，与观测峰值相差大，说明仅靠最大扭矩解释不充分；它也不是完整车辆动力学模型。

P1应用可能重建车辆状态，无原参数重新应用的sham组，不能把全部差异唯一归于扭矩数值。本轮不追加传动/摩擦/质量修改。P1只是仿真研究变体，没有外部真实Tesla证据，不能声称更真实。

## 5. 是否损害紧急制动、间距或稳定性？

**紧急停止距离存在明确退化。** 8个紧急/障碍配对中，P1有**7/8未通过预声明的距离非劣化门槛**：停车距离增量约0.148–0.439m，其中7次超过0.25m。停止时长增加约0.10–0.20s。自然准备导致实际初速P1−P0约−0.024或+0.028m/s，完整披露，不伪称完全相等。

四次P1障碍探针最终真实OBB间距约**3.882–4.031m**，仍≥3m；但这是初始意图5m、立即brake=1的有限紧急探针，**不能覆盖停车距离退化，也不是正常NoBypass3m能力**。

开发下坡P0/P1滑行10s都未完成停止，按截尾记录；验证位置两次P1滑行停止距离增加约.028m、到1s稳定停止的时间增加约.70s。56次记录碰撞0、在线道路corner越界0；最终dense footprint、原始roll/pitch与wheel参数保存。没有实际轮胎接触力/精确路缘Mesh，稳定性只能作有限观察，不认证完整物理安全。

## 6. 能否重新申请正常低速闭环Pilot？

**不能。** Guard工程改善不弥补动力学和紧急距离门禁失败。完整9m、25m、上下坡导航、前车停启、绕行、NoBypass、Worker故障及默认控制器配对矩阵全部**NotRun：前置条件未通过**。原历史Oracle能力未重新计为本轮Passed。

原门槛未改：正常原始最大减速度≤1.8、Jerk峰值≤5、运动Jerk P95≤2.5；位置≤.5m、航向≤5°、末速≤.08m/s、保持≥1s、可行障碍OBB≥3m。初始化与Emergency原始日志不删除，但正常和紧急的准入解释分开。

## 7. 是否暂停当前车辆模型研究？

**暂停当前固定策略和半制动扭矩变体的自动扩展。** 不继续调C/D2/F0、Ackermann PID或第二Physics配置。下一步若获独立批准，应先开展可插桩的执行/车辆动力学接口研究，取得轮端制动、轮胎接触和求解器时间序列，验证命令到车身运动的关系，再选择有物理依据的执行架构或模型。当前证据不足以直接宣布另一车型/后端可靠，也不授权自动迁移。

## Runtime、测试与交付

所有56个探针使用固定50ms墙钟槽位、.05s仿真步长，单tick owner，无epoch重置或补tick。最大Apply间隔**168.56ms**，51次结束Deadline Miss、53个Skipped Slots；最坏事件分解包含前帧tick约62.38ms、记录后休眠/唤醒约105.37ms。并未证明UE/OS内部根因。Physics组和场景顺序未随机化，Tick差异不作因果认定。

本轮没有Planner/Worker，队列、规划年龄和过期规划正常控制为**NotApplicable**，不能填0伪称完整异步驾驶通过。历史350.13ms/339.29ms tick失败仍保留。严格20Hz、共享视觉GPU Runtime保持开放。

最终测试：**371 passed、8 skipped、1 xfailed**；CARLA SDK补测**25 passed**。SDK skip在对应环境覆盖；唯一xfail是历史M6-D原Guard已知缺口，未修改，M6-E新回归独立通过。初版反向路线单测失败保留，没有隐去失败次数。图表/报告后处理修正不改变控制或实验结果。

历史SHA核验：11930份基线跟踪文件、50867份继承资产保持不变。服务器恢复异步、零剩余车辆后关闭本轮启动的专用实例。源码、56份config/physics/原始日志/source snapshot、官方来源、测试和所有报告均纳入release manifest。

- [验收状态](m6e_acceptance.json)
- [实现与复现命令](../docs/M6E_IMPLEMENTATION.md)
- [架构决策](../docs/M6E_ARCHITECTURE_DECISION.md)
- [三种接口原始响应](m6e_figures/m6e_interfaces.png)
- [实际Guard风险/保护切换](m6e_figures/m6e_guard_live.png)
- [P0/P1原始制动与Jerk](m6e_figures/m6e_physics_raw.png)
- [真实障碍间距配对](m6e_figures/m6e_physics_safety.png)
- [全部Runtime长尾](m6e_figures/m6e_runtime.png)

源码溯源补充：`m6e_loaded_source_audit.json`记录两处批次内磁盘/已加载模块差异。development最后一个Runner快照含尚未在该进程加载的P1专用逻辑；validation最后四个trial的Guard磁盘快照已修复，但实际仍为该批启动时旁路Guard。精确复现使用各批首个trial源码和启动提交；实际guard干预与variant批次源一致。没有隐去来源不一致，也没有将最后HEAD当所有执行版本。pytest与早期P1 probe有并发，Runtime报告包含此负载，不能作独占实时或Physics耗时归因。
