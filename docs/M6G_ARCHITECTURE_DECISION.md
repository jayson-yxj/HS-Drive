# M6-G 架构决策：封存执行链观测基线，结束本轮底层研究

状态：`ResearchCompleted / CausalityPartial / InstrumentationNotRun`。依据[验收JSON](../reports/m6g_acceptance.json)及[总结](../reports/m6g_summary.md)。基础`51b4109`，研发分支`feat/m6g-wheel-dynamics-causality`，不合并main。

## A：离线证据的解释能力

接受Observed：25个请求晚帧、Applied对齐后的起步等待、逐轮滑移/载荷/调试力变化、Sham首Tick垂直状态差异。

接受Correlated：Gear等待与轮端不起动、大滑移/锁止与冲击共同出现。接受MechanismSupported：固定公开PhysX源码存在制动过零锁止、sticky低速约束与子步更新相位机制；范围仅为源码参考。内部求解是否真正命中、各机制贡献多少仍Unresolved。

结论Partial。没有将外层时序、末帧调试力或参考源码当作精确内部因果观测。

## B：插桩去留

当前决定NoGo。当前环境无法建立匹配Unreal源码合法访问、工具链及观测前后构建身份，状态为`Instrumentation_NotRun_BuildPrerequisiteFailed`。没有新二进制、等价性结果、子步时间或受控物理事件；全部NotRun。

封存M6-F/G为[ DynamicsObservabilityBaseline ](../reports/m6g_dynamics_observability_baseline.json)，清单引用原始数据及SHA，不复制覆盖历史资产。原始30试验、14,810帧、代码及异常清单可复用；停止无明确新增证据收益的PhysX深挖。将来只有独立建立合法匹配源码、限定成本、行为等价性办法和具体内部量收益，才可另提方案；本轮不授权自动继续。

## C：控制与Physics边界

不批准恢复新控制器、PID/Ackermann搜索、低速阈值试错、P2/P3 Physics、车型替换或停车Pilot。原P0及0.9.15正式主后端不变；0.9.16不迁移正式后端。原P1继续NoGo，不用较低Jerk表象覆盖其紧急制动风险。

G1为`Guard_Partial_NoIndependentPositive`；正常舒适性未过；严格20Hz未过；纯视觉安全、正式Occupancy、可靠三维感知均不因本轮封存升级。禁止视觉输出进入控制，禁止替代动力学后端成绩冒充原Physics结果。

## 主工程独立前进

后续可规划Oracle输入下的路线查询、候选轨迹、道路几何和超时契约测试，或继续Offline Shadow感知研究。这些工作不依赖内部刹车机制完全解释，但必须独立报告路线、动力学、实时性、安全及视觉能力。涉及真实停车或控制安全的质量仍继承开放门禁，不能通过封存研究转移责任。

本轮没有修改Runtime安全合同、六秒预测、Tick Owner或主控制路径。研究遥测不得同步进入0.9.15关键路径；若将来独立提出异步观测，仍需有界队列、原始帧时间、超时丢弃和错帧拒绝，本轮未实现也未认证。
