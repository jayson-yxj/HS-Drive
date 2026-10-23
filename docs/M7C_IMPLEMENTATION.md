# M7-C 实现与复现

基线684b6e6；分支feat/m7c-planning-efficiency-coverage。冻结87cd11d后才运行验证19/76与新最终23/28。所有实现Opt-in、离线SDK+合成Actor，不连接控制器，不训练，不改变Physics、Depth、HeightSplat、默认Planner/Safety或历史文件。

## 几何缓存与等价范围

`diagnostics/m7c_geometry.py`提供GeometryCache和PreparedCandidateGeometry。采用有限LRU（每store最多50000项），缓存完整双精度输入，不做空间量化。sweep键含地图SHA、两端位姿、膨胀车身尺寸与中心偏移；地图region含合法Lane集合、原始Polygon WKB和inset；exact查询含地图SHA、Road/Lane对、z及原始x/y。PreparedCandidateGeometry包括候选SHA、原stamp、车身、每段sweep、guide sweep、曲率、加速度和Jerk，以及带完整场景/时间键的检查记录。动态结果和时效判定每次重算；检查记录不作为永久放行缓存。

原M7-A evaluate实现独立复制为evaluate_cached，仅替换重复几何计算，保留六秒、3m、时间、动力学、静态/道路及全部拒绝规则。维护时应以冻结原实现为对照，不能独立修改它的安全语义。不同地图需不同不可变cache实例，query必须为对应固定地图函数；不允许复用实例后私自修改其map_sha或底层地图。

初版205场景等价通过；接入PreparedCandidateGeometry后再次对205有效场景、1662条含见证候选执行全量配对。每条CV、几何、动态、stop gap、Unknown、选择及数量一致；数值容差1e-9（排除耗时），没有以误差容忍放行危险候选。过期等故障按原时刻检查。初版与最终版报告分开保存。

运行冻结最早版本时输出日志尚在增长，87cd11d包含中途stdout；完成后封存完整stdout，源文件/输入SHA按实际执行记录保留。运行期间对比较helper只做等价移动、源文件末尾空白规范化，已加载评价器不受影响；报告provenance记录的是产出时字节，不能冒称完全逐字运行快照。最终冻结规则清单提供实际验证/最终代码SHA。

## 性能口径

基线和缓存候选在同一CPU/SDK环境配对。cold是每候选新cache；warm是相同候选即时重算；stream跨冻结场景复用，动态和时间仍重算。计时是组件和，不含进程启动、序列化、IPC、CARLA tick；warm是最有利的回放上界，不代表车辆移动后的收益。原始profile分离get_waypoint、swept_polygon、union/buffer、footprint、静态/动态碰撞；重复函数的累积时间不可再次相加。

最终历史配对P95基线333.67ms、冷313.97ms、预热110.28ms、连续复用217.35ms；预热Max111.38ms，连续Max240.16ms。包含新未来车身道路审核及Slow Follow的最终核心P95509.11ms、Max624.10ms。一次几何优化已经证明局部收益，但整体门禁NoGo，不继续自动重写Planner。部分开发/回归计时期间有其他只读诊断，不能给出严格隔离CPU尾时延保证。在线5Hz/20Hz没有实测。

补充`tools/m7c_clean_timing.py`只运行原基础候选的生成、假设、几何和动态检查，交替两臂，不用见证/CV/GT审计预热cache。205场景全等价，基线P95335.10/Max363.81ms，缓存P95218.11/Max245.75ms，19/205超200ms；作为连续核心性能的主要证据。此前等价审计中的stream会受其他审计计算预热，不能作为唯一速度证据。完整新增研究wall time也不包括JSON落盘。

## ConnectedArcRoute

`diagnostics/m7c_routes.py`以Waypoint.next(0.5m)逐步扩展，最多60m额外路径。保留原prefix字节；校验相邻位置≤0.75m、Yaw差≤0.15rad、正向位移、Driving lane及正宽度。最近Waypoint可能偏约0.5m，利用同Road/Lane get_waypoint_xodr和有符号s偏移修正锚点，不平移原路线。

唯一合法后继可扩展；多个后继无合法意图时AmbiguousSuccessors；无后继NoLegalSuccessor；不延长末端切线。接口支持显式合法后继意图，但本轮真实历史及新最终未注入Actor未来意图。intent与Evaluator未来严格分离。节点记录Road/Lane/section、宽度、junction、pose/s，转换记录前后关系、全部合法后继及选择原因；补充段长/累计进度见m7c_connected_body_support。真实地图例0/1/60覆盖唯一连接；6/18/19覆盖歧义；终止/逆向/坏连接/非法意图有合成单测，不把合成终止说成真实地图终止实测。

18/19只能补齐末端约几米就到多分支。原restart_1等部分假设获得位置支持，但旧加速参考仍不完整，绝不任意选分支。支持长度并不代表完整车身支持；新实验对每条未来假设0.05s的完整OBB额外做corridor覆盖检查。m7c_connected_body_support补充历史body审核；原M7B结果不改写。

Actor支持延拓不改变Ego路线目标或原guide path。开发v1意外将延拓后的Actor路径用作Ego guide，导致额外拒绝；v2恢复原Ego route/goal，保留v1失败，修正发生在87cd11d冻结之前。默认控制模块无改动。

## Slow Follow与三种可行性

固定target0.5m/s、duration4.5s，一个解析族：

v(t)=vt+(v0-vt)/2*(1+cos(pi*min(t,T)/T))。

解析积分产生距离，T以后以vt前进，不剪切位姿、瞬时停车或放宽加速度/Jerk。开发有6个合法鲁棒前进见证才启用；选择顺序Keep、Left、Right、Slow、Stop。原C停车研究继续NotRun。

Slow沿用全部几何/动态假设，另在每个样点对每个Actor要求OBB间距≥3+v²/(2*1.8)+v*(1.8/2.5)，终点道路剩余也满足同样余量。这是保守的规划距离筛查，可能拒绝无害邻道Actor；不是实际CARLA执行可行性或完整递归安全证明。没有将3m降为0，没有删制动假设，没有用未来参考选择候选。停止候选仍执行原3m全时域采样合同。

SafeForOneReference只指单一Evaluator脚本；RobustWithinHypotheses指有限假设集合；FeasibleToExecute本轮NotValidated。没有对所有可能减速策略证明必要停车；逐场景归因报告允许多原因且标明有限候选库边界。

## 执行命令

现有.venv含CARLA SDK，.venv-gpu含全量测试依赖；Shapely仍从已存在的独立target目录加载，没有安装新包。

```bash
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m pytest -q tests/test_m7c_geometry.py tests/test_m7b_dynamic.py tests/test_m7a_planning.py tests/test_m7a_shadow.py
PYTHONPATH=runs/m7a_geometry_deps:. .venv-gpu/bin/python -m pytest -q
# 指定新suffix，排他写入，不覆盖封存产物。
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7c_equivalence --suffix _reproduce_01
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7c_scenarios --split development --suffix _reproduce_01
# 最终集仅可复算审计，不能用于再选择规则。
PYTHONPATH=runs/m7a_geometry_deps:. .venv/bin/python -m tools.m7c_scenarios --split fresh_final --suffix _audit_reproduce_01
```

原始序列包含profile-only、connected_audit（v1失败/v2成功）、slow_development、development_v1/v2、两次全历史equivalence、冻结、validation/fresh_final、independent_audit、route_quality、results、figures、release。固定输出名的审计工具排他写入；完整重建需无M7C输出的独立副本和原输入资产，不删除历史文件规避冲突。

测试最终436 passed、8 skipped、1 xfailed，SDK专项63 passed。早期两次全量测试因导入实验工具间接要求CARLA而失败；拆开纯函数与SDK依赖后通过，日志保留。失败轨迹、坏源文件路径命令和空白检查失败均记入failure ledger。原6摄40packet字节/K/T/同步与来源合同再次校验，无新GPU推理。Bench2Drive/PETR不运行。

图表、原始日志及SHA见reports/m7c_*；独立最终检查对34个Actor与42个接受候选做0.05s/0.4m车身网格，和0.025s Shapely未来参考分开。均同源OpenDRIVE/OBB，不称连续物理Mesh真值。新道路23/28仅2个独立片区、共享23模板，同Town03，不是跨地图泛化。
