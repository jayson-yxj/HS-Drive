# M7-A：M6动力学研究正式收束

基础为`feat/m6g-wheel-dynamics-causality@eb2c8f4`。已完整复核M6-F总结、M6-G总结/验收、DynamicsObservabilityBaseline和M6-G架构决策。核验记录在[保全JSON](m7a_execution_logs/preservation.json)。

| 独立状态 | 本轮处理 |
|---|---|
| ResearchCompleted | 已完成低速执行、轮端遥测及有限机制研究，正式结题 |
| CausalityPartial | 外层时序和机制候选成立，内部完整因果仍未知 |
| InstrumentationNotRun | 对应UE源码、工具链及等价构建条件未建立，未插桩 |
| ControlNoGo | 正常低速停车和舒适性仍不合格 |
| RuntimePartial | 严格20Hz与共享GPU实时性未通过 |

原30组CARLA试验、14,810帧0.9.16轮端记录、4次0.9.15 G1记录、失败数据、源码及二进制SHA全部保留。[原封存清单](m6g_dynamics_observability_baseline.json)不改写。M7-A只新增独立文件：13,534份基线跟踪文件、50,867份继承外部资产、111份M6-G交付资产及4份0916身份文件SHA一致。

不重启PhysX编译、PID搜索、Physics参数族或车型替换。0.9.15继续正式主后端，0.9.16仅作为封存研究记录；P0不变、P1 NoGo、G1无独立危险正例。研究结题没有转成控制Passed。

规划基准只评估有限时域候选及当前预测假设；六摄回放只评估接口和研究候选。历史实际停车、间距、舒适和实时失败仍独立列入M7-A总账，不用离线规划结果覆盖。
