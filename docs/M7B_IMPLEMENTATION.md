# M7-B 实现与复现

基于`3d149ac`，独立分支`feat/m7b-dynamic-planning-robustness`。冻结规则提交`a18f754`之后才打开验证与最终结果。全部新增文件Opt-in，无控制输出、训练、Physics更改或默认Planner/Safety替换。

## 数据与算法合同

`diagnostics/m7b_motion.py`提供ArcRoute.at(s)、tangent(s)、lane_id(s)、project(xy)。at返回XY和解缠Yaw，按Polyline弧长插值，越过路线末端显式报PredictionOutOfSupport，不外推、不夹住末端后假称继续运动。lane_id是采样路段ID；本轮单road/lane内部路径，跨junction连接和车道终止延拓尚不支持。

`tools/m7b_fixtures.py`在道路真实弧长处放置Actor，相邻阻塞使用该处CARLA邻道Waypoint，避免全路固定横移。生成审计检查OBB车道多边形、角点SDK查询、中心、初始两两重叠。未来检查t0位置/速度、首段差分≤0.12m/s、加速度[-3,1.5]、方向、道路OBB。重新起步初速为0，等待后有限加速。横穿小目标有单独穿越corridor，不能说其初始位置必须在Driving lane。

道路中心误差在生成审计侧使用同一路径，单独`m7b_post_audit`以CARLA中心航向与0.4m车身网格独立复核新最终集；同源OpenDRIVE仍不是独立Mesh真值。动态参考dt=0.025s，检查道路OBB时取0.1s采样；不是连续完整道路证书。

ObservedMotion只包含当前快照合法关联的Actor ID、道路路径、s0及类别。MotionHypothesis记录初态、完整位置/yaw/speed序列、加速度、延迟和来源。静态几何保持不动；停止车辆仍可能起步，二者不可因速度0而混淆。道路先验和Actor状态均Privileged，不能由此称RGB模型输入。

有限假设为a={-3,-1.5,-0.5,0,.75,1.5}m/s²，onset={0,1,2}s，速度[0,8]m/s，范围6s，采样0.05s，重复的停止假设去重。精确积分速度饱和后的距离，禁止倒车和隐式瞬移。沿道路弧长得到位置/yaw；旧七条历史回归例外明确使用其原始声明切线corridor，以保留原错误输入，不把它当正确新道路参考。任一需要防范假设超出路径，整体PredictionOutOfSupport，不把可算子集冒充完整鲁棒性。

`evaluate_dynamic_hypotheses`不接收Evaluator未来。每个相邻时间段做连续平移SAT，旋转用半角位移上界扩张OBB。实际采用分段直线中心/最短角插值；不保证未枚举加速度、横向换道/意图、曲线段内任意物理运动或Mesh碰撞。批量SAT与500组随机标量SAT一致，优化前后46开发场景判定一致。停车3m沿用M7-A整个候选时域的OBB间距采样要求，独立参考也复核；间隔之间的严格3m连续最小值未证明。

A：原generate+原CV evaluate。B：原generate+原运动学/道路/静态审计+多假设动态审计。B不保留错误的“停止车辆永远静态”路径意图假设；仍通过保持停车假设和动态碰撞检查约束它。未来参考只用于独立Shapely审计，不影响生成和排序。最终顺序A_keep、B_left、C_right、D_stop；无解显式拒绝。过期、未来、缺失、worker_error是离线输入故障表示，并非新的真实Worker崩溃。

C门禁未满足：开发集未发现B全部拒绝、但解析有界Jerk停车见证可行的场景，因此不启用新停车候选。见证v(t)=v0/2*(1+cos(pi*t/T))，T=2.5..5.5s，由同样运动学/3m/道路/动态检查验证，仅用于“存在可行轨迹”统计。不存在见证不证明全局无解；NoFeasibleCandidate表示当前有限库不能提供证书，不可说瞬间停车可行。

`tools/m7b_coverage_audit.py`额外统计只在单一Evaluator脚本未来安全却被鲁棒规则拒绝的场景，避免循环定义成功；并用v²/(2a_max)与直道剩余3m安全停车空间比较，提供有限的原车道纵向无解证书。所有此类结果为评后审计，不改变生成器或选择规则。

## 实验与隔离

开发Road3/20，验证76/19，新最终2/4，历史回归65/67/18。选择最终道路时只读地图长度/曲率/拓扑，没有查看规划结果。最终与开发/验证最小采样路线距离28.665m；同一Town03、共用23模板，仅2条新最终道路，不是46个独立地图样本，不提供虚假的独立置信区间。

验证与历史回归各有1项参考超出路线，保存FixtureInvalid，全部分母单列；未知预测不当作空闲。新最终未调参或重测。历史M7-A84场景/180候选/74接受/7危险候选、6起步错误、6放置错误原字节不变。

六摄40packet再次验证原RGB SHA、K/T、同步、原timestamp和白名单，无新GPU前向。沿用旧GT独立Evaluator、Unknown非Free、无法控制导出合同。Bench2Drive没有新安装或smoke；PETR/新3D训练NotRun。

## 实测命令

环境沿用两个venv与已有隔离Shapely目录，没有安装新依赖。命令从仓库根目录执行：

```bash
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m pytest -q tests/test_m7b_dynamic.py tests/test_m7a_planning.py tests/test_m7a_shadow.py
PYTHONPATH=runs/m7a_geometry_deps:. .venv-gpu/bin/python -m pytest -q
# 可运行的复现：指定新的输出后缀，排他创建，旧报告不会被覆盖。
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7b_experiment --split development --suffix _reproduce_01
# 最终集只允许复算审计，不得用于选择或调参；原最终只运行一次。
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7b_experiment --split fresh_final --suffix _audit_reproduce_01
```

原运行记录：development_v1为标量SAT；development_v2为批量SAT；validation_frozen/fresh_final_frozen/historical_regression_frozen在a18f754之后运行。每份JSON含实际执行时Git及源文件字节SHA，不能把最终提交冒称原执行提交。`reports/m7b_execution_logs`保存完整stdout、pytest XML、函数profile、失败；数据文件保存所有候选、假设、参考、错误与计时。

固定名的历史七条/汇总/绘图工具是封存流水线，排他写入；不能在包含既有输出的工作树直接重跑。完整重建需准备单独的无M7-B输出工作副本并挂载原输入资产。不能删除旧资产规避冲突。

失败保留：历史七条初次JSON重建未恢复Stamp导致结果无效、随后profile报AttributeError；`m7b_historical_seven.json`仅是失败产物，正确记录为`m7b_historical_seven_v2.json`，不是模型漏判。还保留开发旧版及验证/回归FixtureInvalid。日志末尾空白使git diff --check报告空白问题，保留原profile日志不改实验内容。

## 性能与测试

424 passed、8 skipped、1 xfailed；SDK专项51 passed。skips和旧M6-D xfail沿用，未降低测试标准。函数profile分离swept_polygon、exact地图查询、SAT和Shapely；分场景记录生成、多假设生成、道路/静态/完整车身、动态碰撞、排序、独立参考。body/road/static阶段仍聚合计时，不能从累积profile重复求和。

离线B组件总成本排除见证、A对照和独立GT审计，但包括全部B候选检查；不是端到端在线周期，不含IPC、采集、调度及CARLA tick。最终P95=367.02ms/Max=529.80ms，38/46超过200ms，不具备直接在线5Hz批准条件。原严格20Hz、共享GPU、真实低速执行和舒适性门禁不变。
