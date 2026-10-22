# M5-M — 可信接触几何与选择性目标定位

**结论：Partial。接触定位、Vehicle 残差继续训练、具有可信位置的 Object Query Pilot 均为 No-go；Shadow 仅通过静态进程隔离工程检查。**

基于 `f2f4e1a`，独立分支 `feat/m5m-selective-object-geometry`。旧模型、48 Depth Bin、HeightSplat、控制器和 Runtime 默认行为保持不变。没有视觉控制、没有训练新残差或 Contact Head。

## 接触参考与定位

历史 Train/Val/M5-K Test/M5-L Test 均重新审计；旧测试仅作回归。补采4个开发位置、4个冻结后封存位置，每位置9配置；封存2帧配对姿态超差被剔除，主评测34帧，失败原数据保留。原生实例轮廓、可靠侵蚀像素分开统计。多点碰撞道路查询和渲染道路对照不是精确 Mesh 接触真值。

所有参考仍无 Gold：轮胎物理圆半径和脚部骨骼原点尚未校准到可见轮胎接触斑/鞋底；遮挡、坡度和姿态稳定性也不足以认证。故 Contact Head 不训练，`VerifiedPosition=0`。候选保留 `Unknown` 和可选位置假设，Unknown 不表示不存在或 Free。

|集合|seed|已知可见 Actor-view|检出|规则条件定位|Coverage|XY median / P95 / Max (m)|
|---|---:|---:|---:|---:|---:|---|
|val|101|76|53|33|43.4%|1.91 / 7.31 / 11.34|
|val|102|76|51|36|47.4%|2.09 / 10.70 / 17.75|
|test|101|46|21|16|34.8%|1.37 / 4.18 / 5.06|
|test|102|46|23|19|41.3%|1.72 / 9.28 / 9.34|
|sealed|101|38|32|28|73.7%|0.93 / 4.38 / 4.66|
|sealed|102|38|32|28|73.7%|0.77 / 3.73 / 4.18|

预声明要求至少30%可见目标覆盖、20个匹配、3个位置、P95≤5m且不差于基线，>5m错误率≤5%。Validation置信度0.6仅留下8/76=10.5%；未通过覆盖门禁，不通过拒绝大多数目标换取成功。完整同目标Dense Expected/Mode、类别/相机/位置/距离分层和风险曲线见完整JSON。

## 固定残差与道路平面

无Mobile残差 seed101 的全部20–35m Bias：原始−0.357m、无条件−4.226m、RGB q≥0.9门控−0.493m；但35–80m仍由−9.833m恶化为−12.518m。seed102同方向。分层查明：35–80m的783条可靠射线中，575条背景仅由−9.491变为−9.518m；208条未关联Actor的前景由−10.778恶化为−20.813m。不能把这部分直接称背景或已确认Vehicle。全部远场及逐场景门禁仍失败，停止这类残差训练。二值前景不能替代Vehicle分类；Pedestrian/Cone/Unknown分别记录，不宣称小样本类别泛化。

固定地面与历史平路渲染路面约有0.25m高度偏差；局部Oracle平面和GT底部像素分别干预。Oracle道路只作诊断。原Depth估计的RGB局部平面不具备认证法向，不新增道路网络、不启用推理替换。

## Shadow 实测

唯一Tick Owner、独立Worker、2Hz最新完整六摄包、250ms原时间戳/墙钟过期拒绝；6秒Safety和控制代码未改。每条件两次，第二次交换idle/shadow顺序。首次推理和CUDA初始化独立记录。

|重复|条件|Tick P95 / Max (ms)|
|---:|---|---|
|0|alone|65.85 / 269.01|
|0|loaded_idle|62.53 / 69.72|
|0|shadow|83.49 / 98.11|
|0|recovery|61.37 / 65.87|
|1|alone|62.71 / 77.12|
|1|shadow|84.45 / 87.99|
|1|loaded_idle|61.57 / 64.31|
|1|recovery|62.38 / 71.93|

严重退化停止触发：False。静态六摄资源试验不能外推驾驶实时性。队列put时间不包含feeder线程完整序列化，queue age包含等待/调度/传输，不能虚称已精确拆分。

## 下一阶段决策

1. **接触几何：No-go升级。** 先获取可核验轮胎/鞋底接触与可见性标注，保留条件假设。
2. **Vehicle Depth：No-go继续同类训练。** RGB门控有研究增益，但全部远场表面与逐场景保护失败；可靠背景大体受到保护。
3. **Object Query：暂缓位置型Pilot。** 输出合同已就绪；可信位置与置信度尚未满足最低覆盖/尾部门禁，不等于SparseDrive或规划许可。
4. **Shadow：工程隔离通过、资源隔离Partial。** 冷启动/稳态/顺序效应分别报告，进一步测试需维持纯观察及停止条件。

完整复现、资产SHA、失败日志见 `docs/M5M_IMPLEMENTATION.md`。来源类型在每份JSON中明确标注；测试不等价于泛化或安全门禁。

测试：全量304通过/5跳过，CARLA SDK专项14通过。所有失败记录保留。历史资产完整性以`m5m_integrity.json`为准。

图：[风险覆盖](m5m_figures/m5m_risk_coverage.png)、[接触空间](m5m_figures/m5m_contact_geometry.png)、[尾部失败样本](m5m_figures/m5m_contact_failures.png)、[残差Bias](m5m_figures/m5m_residual_bias.png)、[Shadow原始tick](m5m_figures/m5m_shadow_ticks.png)。

补充：[轮胎/脚部候选投影](m5m_figures/m5m_anatomy_candidates.png)。物理轮/骨骼附近有渲染表面支持不等于精确接地，仍无Gold。

Shadow两轮平均16.4Hz，全部测量tick间隔均超过50ms；未触发严重退化停止不代表实时门禁通过。冷启动完整年龄约0.99/0.98秒，全部被拒绝；稳态端到端P95约133ms。
