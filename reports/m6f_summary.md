# M6-F — Vehicle Dynamics Observability & Backend Feasibility

**总体Partial：获得真实轮端遥测，批准有限可观测性研究；正常停车控制、Physics变体和完整Oracle Pilot不批准。** 基线`b0ee249`，独立分支`feat/m6f-vehicle-dynamics-observability`。主CARLA0.9.15、Tesla P0、Oracle Planner/Safety及六秒预测合同全部保留，无视觉控制、训练、车型扩展或main合并。

## 实际完成及证据类型

- **真实服务器安装/启动**：官方CARLA0.9.16完整8,346,095,504字节归档、隔离SDK环境、独立2200端口；一次启动成功。单流下载慢，改同一文件16个Range续传，在900秒总预算内完成。未安装附加地图或破坏0915环境。
- **真实CARLA0.9.16**：26个有限试验，14,810帧。1个API可行性、12个自动/手动冷/暖启动与制动cycle、6个坡道滑行、6个原参数/Sham/P1配对、1个800帧ABBA开销试验。
- **真实CARLA0.9.15**：4个冻结G1试验，两个新位置74/91、各两次。没有人为赋速度、瞬移构造回滑。
- **离线诊断**：逐帧请求/Applied/轮端/车身链、原始速度差分、命令延迟、Sham状态改变、SHA与资源审计。
- **合成/工程回归**：完整380 passed、8 skipped、1 xfailed；0915 SDK补测44 passed、1 xfailed；0916 SDK合同补测9 passed。xfail为原M6-D已知问题，未删除。第一次SDK命令因旧文件名不存在退出4，失败记录保留。

30个试验均完成，碰撞记录0；这是有限直行执行器探针，**不是路线驾驶安全或道路/路缘质量认证**。重复编号不是独立随机场景，帧不当作独立统计样本。全部实际实验均标记Privileged。

## 1. 原Physics下哪些可观测，哪些仍是黑箱？

0915原API仍只有请求、Applied Control、Gear、车身快照和公开Physics等；不存在`get_telemetry_data`。0916实际读到了四轮omega、long/lat slip、contact friction、tire load、debug force/torque，以及RPM/Gear。公开Physics参数与0915 P0逐项相同，但引擎/版本影响不能排除，**不是跨版本单因素对照**。

0916 canonical PhysicsP0 SHA：`5fabd76f75512661a18700622619982f6ed22bcf079485bb82c24e239973f5a9`。旧`3c493…`是不同JSON序列化文件SHA，不代表Physics发生变化。

14,370个有遥测的post-tick样本、57,480个轮端记录；实际外层帧错配0，290个无tick重复请求变化0。Body与Telemetry前向速度绝对差最大约`9.79e-7 m/s`。归一化力在零载荷下316条/字段非有限，全部保留`invalid_raw`，没有填零。

仍不可访问：**真实每轮刹车扭矩、驱动/制动力矩分解、平滑后的PhysX控制输入、离合器/换挡过程、接触冲量、子步时间戳及完整源构建对应关系**。`torque`只能称调试轮端扭矩；本轮它与`−0.37×long_force`最大差仅约0.00058 N·m，支持其为轮胎反作用扭矩的解释，不能把它冒充刹车测量。

字段单位/来源见[Telemetry合同](m6f_telemetry_contract.json)；源码依据为[官方GetVehicleTelemetryData实现](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp)。

## 2. 异常发生在哪一层？

**已缩小范围，未完成完整根因定位。**

- 自动挡冷启动约3.60–3.65s；油门已应用、RPM持续上升时Gear为0、车轮近乎不转。固定1挡约0.55–0.60s起步。与传动等待关联证据明确；自动换挡内部状态未导出，不能断言具体阈值算法。
- 0.01普通制动请求实际读回一致时，仍出现四轮锁止、约−0.935纵向滑移及每轮约−9.7至−12.9kN调试纵向力，随后车身急减速。代表p27 A普通制动峰值24.70m/s²，Jerk440.71m/s³。自动A六次普通制动19.17–24.70m/s²，手动B14.66–22.26m/s²；所有12个cycle正常舒适门槛失败。
- 最大减速度所在帧，调试力可能已归零；它是读到的状态/调试字段，不是跨50ms积分冲量。因此不能用末帧力直接解释整段加速度，也不能判定求解器违反牛顿定律。可能的低速约束/静摩擦/输入平滑机制仍是待验证假设。
- 25个相位切换行中请求晚1tick才应用；Applied与Telemetry之间不一致0。请求时刻不等于真正作用时刻，全部记录在详细审计，没有删除。这也限制Sham响应差异的单因素解释。
- 上坡滑行出现真实负速和四轮反转，末速约−0.076m/s；下坡10秒仍约+0.453m/s，近水平可停。发动机/传动阻力与坡度贡献没有完整分解，不能认证零油门可普遍可靠停稳。

[命令—轮端—车身分析](m6f_actuation_causal_probe.json)、[详细逐帧证据](m6f_execution_logs/detailed_evidence.json)、[轻制动链图](m6f_figures/m6f_light_brake_chain.png)、[冷启动图](m6f_figures/m6f_startup_chain.png)。

## 3. 0916能否替代昂贵的0915插桩？

**可替代本轮有限轮端观测需求，不能替代完整子步因果插桩。** 官方服务端和匹配SDK真实运行，不是根据release说明推断。没有同时开发0915插桩；该条件工作包NotRun。建议保留独立0916研究基线，下一步如需根因，应先评估在0916增加实际模拟输入、扭矩分解和带子步时间戳导出的最小方案。HS-Drive主系统不迁移，不能声称0916更真实。

[0916可行性报告](m6f_0916_telemetry_feasibility.json)、[0915边界](m6f_0915_observability_limit.json)、[源码执行链](m6f_execution_chain_audit.md)。

## 4. PhysicsControl应用本身会改变初态吗？

**会，已实际观察。** 两次Sham和两次P1施加后，未tick时live Telemetry速度由约0.0029784m/s变为0；ActorSnapshot仍是旧缓存，frame没有变化。未施加组没有这次变化。说明相同frame不保证Physics重建前后是同一物理状态；初始化对照单列，不跨重建算差分。

| 同版本、同位置、两次重复 | 普通制动最大减速度 m/s² | 原始Jerk峰值 m/s³ |
|---|---:|---:|
| P0-unmodified | 24.695 | 440.714 |
| P0-sham-reapply | 23.004 | 404.247 |
| P1-wheel-brake-experimental | 21.734 | 386.462 |

施加操作是明确混杂；后续减速差异也有实际命令晚帧因素，**不能把全部差值唯一归于重建或扭矩参数**。即使数值下降，三组仍舒适失败。M6-E P1紧急距离7/8非劣化失败保持NoGo，本轮未重新认证紧急停车。

[Sham报告](m6f_physics_sham_comparison.json)、[原始响应图](m6f_figures/m6f_sham_raw.png)。

## 5. G1有新的独立危险正例支持吗？

**没有。** 两个新位置四次释放段最大回退约2.15mm或1.48mm，最小负速约−0.059/−0.055m/s；位移证据未达到危险回滑门槛，Risk/Emergency均0，后续恢复前进。它们是小扰动阴性对照，不是4/4危险回滑保护成功。实际保护后Jerk/停止距离NotRun，因为没有触发保护；原始释放速度/位移/Jerk完整保存。未扩展位置搜索或改阈值追求正例。

合成抖动、侧滑、目标前移、恢复、非单调/丢帧测试保留通过，但不替代实际灵敏度。G1仍Partial、不升级默认。

[Guard报告](m6f_guard_validation.json)、[独立位置原始图](m6f_figures/m6f_guard_raw.png)。

## Runtime及研究可信度

单次Telemetry RPC P50/P95/P99约0.459/0.739/0.994ms，**Max185.995ms**。静止ABBA中，带两次RPC的循环工作P95约2.63–2.69ms，关闭时1.55–1.82ms；该800帧局部对照无deadline miss，但不能覆盖运动试验长尾。

全部0916控制间隔P50/P95/P99/Max约49.995/50.230/50.435/**200.224ms**，平均19.890Hz，48次工作deadline miss、82个跳过墙钟槽位。没有补tick、重置epoch或缩短Safety预测。无Planner、无共享视觉GPU负载，不能当完整Runtime复验，严格20Hz保持开放。

批次期间独占CARLA，未并发pytest/训练/GPU画图。批前/末保存进程/GPU快照，未声称连续监测所有宿主机争用。服务端日志保留；两实例均恢复异步、车辆0并关闭。

项目模块直接捕获导入字节及module code哈希；所有批次末源不变验证通过。外部模块磁盘SHA不冒充内存哈希。精确执行提交：可行性`c1e2498`、响应`58a8d05`、Sham/开销`29670eb`、Guard`02e7955`。最终提交只包含后处理/交付，不冒称所有试验执行版本。

[开销报告](m6f_runtime_overhead.json)、[ABBA图](m6f_figures/m6f_rpc_overhead.png)、[源及数据清单](m6f_execution_logs/trial_manifest.json)。历史13169份跟踪文件、50867份继承资产SHA核验全通过，失败记录和旧报告未覆盖。

## 6. 后续方向与停止路线

继续：**独立0916的物理执行链可观测性研究**，优先获取平滑输入、刹车/驱动扭矩分解、低速接触与子步时间。现有证据可以提出这个有限研究方向，但还不能恢复正常停车控制或批准新Physics族。

停止自动扩展：原低速策略/PID/Ackermann Jerk调参、半制动扭矩P1、盲选新Physics/车型、0915与0916双线全量插桩、完整Oracle停车Pilot。主后端继续0915，G1只保留研究候选。原舒适性、3m安全间距、故障降级、共享GPU严格20Hz、视觉驾驶和Occupancy门禁均不升级。

复现命令、隔离目录和失败说明见[实现文档](../docs/M6F_IMPLEMENTATION.md)；独立决策见[架构决策](../docs/M6F_ARCHITECTURE_DECISION.md)和[验收JSON](m6f_acceptance.json)。
