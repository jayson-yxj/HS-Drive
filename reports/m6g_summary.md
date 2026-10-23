# M6-G — Wheel Dynamics Causality & Minimal Instrumentation Gate

**研究完成，因果归因仍Partial；最小插桩NotRun，停止当前底层深挖。** 本轮新增了实际Applied对齐、逐轮大滑移事件、同帧调试力无法解释的冲击清单，以及可重复的Sham首Tick垂直状态差异；没有获得新的求解器内部测量。不得把事件时序或公开源码机制升级为Shipping二进制的完整因果证明。

基线`51b4109`，独立分支`feat/m6g-wheel-dynamics-causality`。完整保留M6-F的30组真实CARLA试验；本轮**新增CARLA试验0、编译0、训练0**。0.9.15仍为正式主后端；0.9.16仅为隔离研究。P0、默认控制器、G1阈值、模型及历史报告不变。

## 实际完成及证据类型

| 工作 | 实际结果 | 证据类型 |
|---|---|---|
| 26组0.9.16历史试验重分析 | 14,810帧；174个命令事件、107个制动事件、43个起步事件、19次Gear变化 | M6-F真实CARLA数据的离线诊断，全部Privileged |
| Applied对齐 | 149次无外层延迟、25次延迟1帧；与M6-F延迟清单逐项相等 | 离线诊断 |
| 官方源码 | 固定CARLA 0.9.16与公开PhysX3.4文件、SHA及缺失链路 | 官方源码审计，非二进制身份认证 |
| 构建准入 | Unreal访问未建立，本地无对应源码/匹配工具链 | 实际服务器检查 |
| 条件插桩、等价性、物理事件复核 | `Instrumentation_NotRun_BuildPrerequisiteFailed` | 未执行，无虚构测量 |
| 测试 | 全套390 passed、8 skipped、1 xfailed；0.9.15 SDK环境专项19 passed | 合成单测、已有记录回归；不是新增CARLA闭环 |

每个事件保存原始试验、frame、时间戳、采集源码提交、trace SHA、Physics SHA及完整时间序列行号。分析执行版本`6e9d7d4`，逐轮补充与绘图版本`30b5988`；最终封存代码版本另见JSON。Physics P0 canonical SHA为`5fabd76f75512661a18700622619982f6ed22bcf079485bb82c24e239973f5a9`。安装二进制及原始采集模块字节SHA见原M6-F清单，最终提交不冒充采集执行提交。

## A. 小Brake后，轮速下降还是车身强减速先被观察到？

普通`Brake=0.01`共37事件：33个P0事件、4个历史P1事件。固定描述阈值为轮速绝对值下降≥0.5rad/s、车身原始差分加速度＜−1.8m/s²；它们用于定位事件，**不是精确运动起点或新的验收阈值**。

| 外层观测顺序 | 全部37事件 | P0的33事件 |
|---|---:|---:|
| 轮速下降先出现 | 28 | 24 |
| 车身强减速先出现 | 3 | 3 |
| 同一外层帧 | 6 | 6 |

代表`responses__p27_A_401__command_3`：Applied及轮速下降frame4189；四轮锁止和强减速frame4190；最大减速度frame4191，为24.695m/s²。采集源码`58a8d05`，trace SHA`42d6f94f30db6ba871e263867085e0bc68a5904a6929aefb4e2270abd0cec6be`。3个车身先出现的事件来自p170 B的normal/final stop，说明不能把“轮先、车后”强制推广到所有记录。

首次Applied观测只界定相邻快照之间的作用区间；没有真实子步时间。22/37事件在制动前已存在大滑移，不能把它们写成Brake触发的新滑移。

[制动事件JSON](m6g_braking_event_analysis.json)、[Applied对齐](m6g_timeline_alignment.json)、[逐帧制动图](m6g_figures/m6g_applied_brake_event.png)。

## B. 大滑移与载荷、调试力是否同时变化？

148个逐轮首次高滑移观测中，50个在制动前已高滑移，属于左截断；其余98个首次越过`|long_slip|≥0.5`的外层帧，均观测到载荷变化＞0.01N及调试纵向力变化＞1N。载荷变化范围−213.956至+199.208N。这两个变化幅度仅描述实数差值，不作为显著性或安全检验；没有根据它们调参。

这是**同一个API样本中字段一起变化**，不是证明它们在同一个物理子步计算。车轮载荷/调试力、QueryResult、更新后的omega可能来自不同计算相位。每轮前一帧和当前帧原值均在[补充事件记录](m6g_execution_logs/event_details.json)，未将debug wheel torque改称实际Brake Torque。

## C. 哪些Jerk冲击仍解释不了？

按固定`|raw jerk|>5m/s³`阈值聚类得到181个连续事件。41个峰值帧不能单靠同帧调试力字段解释：40个四轮debug纵向力均接近零（≤1N），另1个为ABBA计划关闭遥测的初始化帧，**缺测不是零力**。这41个事件的阶段为初始化20、普通停车14、最终保持4、final stop3。37个小制动事件中，17个**最大减速度帧**也是四轮debug纵向力近零。Jerk峰值与减速度峰值是两个不同采样点，已分别保存。

例如p27 A的frame4191最大减速度24.695m/s²，frame4192原始Jerk约440.714m/s³，同帧debug纵向力为零。它不等于实际合力或整个50ms内的积分冲量为零。

181个冲击允许多标签：大滑移相关149、锁止/静止接触相关101、Gear切换相关19、重建初始化相关4；8个没有上述关联候选。**标签可重叠，所有内部根因仍Unresolved。** PhysX公开源码存在车轮制动过零锁止、低速sticky约束及不同子步缓存相位，提供MechanismSupported候选解释，但缺少最终模拟输入、真实制动求解项、约束冲量及子步标记，不能判定具体事件命中了哪条分支。

[冲击分类图](m6g_figures/m6g_event_evidence_limits.png)、[固定版本调用链及证据缺口](m6g_cpp_source_trace.md)。

## D. 自动挡冷启动的时序是什么？

响应批次自动挡冷启动：首次Applied油门的同帧RPM已上升；Gear0期间RPM约44.37→2254.53。**在Applied后3.55s，首次观察到Gear1、轮速运动及车身速度超过0.08m/s，三者同一外层帧。** 例如p27 A401的frame4069→4140。

固定1挡B组在给油前已是Gear1，首次Applied同帧出现轮速，车身越过0.08m/s约需0.50–0.55s。M6-F从Request计时的3.60–3.65s，部分差异是请求晚帧，不是内部换挡等待的差异。

这足以把等待定位在“已观察到油门/RPM变化、仍未观察到Gear1和轮端运动”的阶段，无法确定target gear、离合器或autobox timer的精确因果贡献。A/B初始化还同时有service hold与handbrake/manual差异，不能解释为严格单变量换挡消融。重复试验也不是独立场景统计样本。

[起步JSON](m6g_startup_transmission_analysis.json)、[Applied起步图](m6g_figures/m6g_startup_applied.png)。

## E. Sham首次Tick有什么可重复变化？

M6-F已观察到Sham操作后、尚未tick时live前向速度约0.0029784→0m/s，而缓存ActorSnapshot不变。本轮进一步发现：未重建组首次tick世界垂直速度为−0.980000m/s；相同公开P0重新施加后为−0.490000m/s，差+0.490000m/s，车身z高0.024500m。401/402两次重复相同；历史P1重建组也出现相同初始差异。

具体首帧：P0/Sham 401为16953/17610，402为20233/19576。试验内frame和时间序列用于对应，不把跨试验绝对frame当作同时采样。说明重建改变了坠落/初始化过程，**是可重复的车身状态观测，尚无完整内部状态重置列表**。这也是后续制动差异的混杂，不能把全部改善归于较小max brake torque。

跨重建的首个速度差分和相邻Jerk明确无效，未用旧缓存和新状态相减造冲击；后续50ms原始差分保留。P1仍NoGo。

[重建报告](m6g_physics_rebuild_analysis.json)、[首Tick曲线](m6g_figures/m6g_rebuild_first_ticks.png)。

## 构建门禁、Runtime与Guard

CARLA官方源码可访问并固定版本；对应Unreal私有fork经官方API返回404、现有Git身份访问返回Repository not found，本地搜索未发现UE源码，也没有匹配UE工具链。**这里只能说当前环境的合法访问前置条件未建立，不能推断用户是否拥有许可。** 空闲约194.9GB超过官方130GB参考，但不能代替实际峰值构建资源验证。未编译、未下载非官方UE镜像、未修补现有安装；服务端新SHA为null。

| 历史0.9.16样本开销，ms | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| Telemetry RPC，29,030次 | 0.4545 | 0.7377 | 0.9918 | 185.9946 |
| 世界tick调用，14,810次 | 1.2477 | 1.7685 | 2.0324 | 76.7778 |
| 控制Apply间隔，14,784次 | 49.9947 | 50.2298 | 50.4353 | 200.2237 |

平均19.8898Hz、48个工作deadline miss、82个跳过墙钟槽位。RPC人口比M6-F主表多290个同帧重复读，旧主表保留。440个计划关闭遥测/初始化行不是漏帧；已保存trace未发现非计划缺帧或错帧，不能推断未记录链路的丢包率。服务端插桩开销与异步诊断均NotRun；没有同步遥测进入正式控制路径，严格20Hz仍未通过。

G1沿用M6-F的4次小扰动阴性试验，没有自然独立危险回滑正例；`Guard_Partial_NoIndependentPositive`。不重新搜索阈值、不构造伪真实正例。

[构建门禁](m6g_instrumentation_feasibility.json)、[条件试验NotRun](m6g_instrumented_event_comparison.json)、[Runtime](m6g_runtime_overhead.json)、[Guard](m6g_guard_status.json)。

## 三项独立决策

**A：离线证据能否解释异常？Partial。** 新增了准确的Applied时序和可重复的首Tick初态差异，足以反驳“请求时刻即执行时刻”“Sham保持初态”“同帧debug力代表全部制动力”等解释。没有得到当前可观测范围之外的新内部测量；小制动强减速和换挡等待的完整内部根因未解决。

**B：继续子步插桩？当前NoGo并停止。** 证据缺口具体，但合法匹配UE源码、工具链和可比构建身份未建立；按预设停止规则封存为`DynamicsObservabilityBaseline`。这不是启动下一轮无限环境排查的计划。

**C：恢复控制器或Physics研究？不批准。** 缺少可可靠控制且有真实测量支持的新执行机制。默认控制、P0不改；P1NoGo、正常舒适性失败、严格20Hz开放、G1 Partial、视觉驾驶未验证全部保留。

Oracle路线管理、规划逻辑、场景评测和离线感知可以在各自合同下独立推进，无需等待PhysX黑箱完全解释；不可据此宣布停车、Runtime或安全驾驶通过。实施与复现见[实现文档](../docs/M6G_IMPLEMENTATION.md)，停止边界见[架构决策](../docs/M6G_ARCHITECTURE_DECISION.md)。
