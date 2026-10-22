# M5-N — Direct Actor-Center Localization & Uncertainty Calibration

**总体 Partial：仿真 Box Center监督合同和训练管线通过；Direct Head跨位置定位与选择性可靠性未通过。暂停当前训练路线，不申请位置型Object-Center Query Pilot。**

基线`29ad67a`；独立分支`feat/m5n-direct-actor-localization`。原Mobile、ResNet Depth、48 LID、HeightSplat、Controller与Runtime默认实现未改。无视觉驾驶、无接触Gold升级。

## 参考与数据

八个已变换Box顶点均值定义仿真Box Center；反变换恢复局部中心、旋转与尺寸，Actor原点、可见表面均值、轮廓底部、物理接触分别保存。历史静态帧时间戳与标定检查通过；新采集显式记录Actor Snapshot帧号/时间及bbox局部参数。Box Center不是Mesh质心或物理接触Gold。

复用Train/Val各122帧、6位置；每种子可靠唯一监督Train64/65、Val60/59个候选。Vehicle32/33训练，Pedestrian各13、Cone各19，后两类不具备泛化支持。K/L/M旧测试全部回归，不用于训练或阈值选择。M5-M两帧配对失败独立审计：单帧静态几何相容，但旧sidecar依赖配对自车Mask，仍不纳入主评价；没有自动认定单帧无效或有效。

## 固定小预算训练

冻结Mobile P8，候选框3×3采样共576维视觉特征加24维合法标定/区域特征；小MLP预测log光轴Z与候选中心相对Box中心投影偏移，再用K/T重建XY。两种子各600步；几何特征对照使用相同候选、样本顺序、预算。只有回归Head更新，没有NLL或新Depth训练。Checkpoint由Validation位置宏平均XY选择。

|集合|seed|方法|定位/可见Actor-view|MAE|Median|P95|>5m比例|
|---|---:|---|---:|---:|---:|---:|---:|
|val|101|plane|45/93|2.38|1.41|6.44|13.3%|
|val|101|dense_expected|68/93|8.34|8.79|16.65|73.5%|
|val|101|dense_mode|68/93|7.28|5.89|19.55|54.4%|
|val|101|direct|68/93|3.83|2.88|8.79|26.5%|
|val|101|geometry_prior|68/93|2.56|1.39|9.77|11.8%|
|val|102|plane|48/93|2.92|1.63|10.00|18.8%|
|val|102|dense_expected|67/93|7.53|8.12|15.99|67.2%|
|val|102|dense_mode|67/93|6.71|4.39|19.55|47.8%|
|val|102|direct|67/93|4.16|3.02|11.67|26.9%|
|val|102|geometry_prior|67/93|2.86|1.19|10.88|20.9%|
|sealed|101|plane|34/75|1.57|1.45|2.91|0.0%|
|sealed|101|dense_expected|55/75|9.51|9.97|15.76|74.5%|
|sealed|101|dense_mode|55/75|7.21|3.31|22.33|41.8%|
|sealed|101|direct|55/75|4.09|3.58|9.83|25.5%|
|sealed|101|geometry_prior|55/75|3.14|1.30|7.00|14.5%|
|sealed|102|plane|38/75|1.61|1.52|3.51|0.0%|
|sealed|102|dense_expected|54/75|9.30|9.54|17.16|70.4%|
|sealed|102|dense_mode|54/75|7.44|3.25|22.32|38.9%|
|sealed|102|direct|54/75|4.12|3.52|11.05|31.5%|
|sealed|102|geometry_prior|54/75|3.55|1.90|7.06|13.0%|

表中是各方法自然覆盖，不能直接当成严格同目标消融。同目标比较见`m5n_risk_coverage.json`：Fixed Plane会拒绝困难目标；Dense摘要是同一RGB候选内原Depth概率均值的Expected/Mode，可能定位可见表面而非Box中心。没有在Test挑选摘要。

Direct训练Loss接近零而Validation尾部未改善，符合小样本过拟合表现；几何先验在Validation的均值更低，未证明视觉区域特征带来稳定增量。未因短训练失败追加步数或调Loss。

## 不确定性与拒绝

用Validation留一位置拟合logXY误差的岭回归，仅输入预测位置/候选形状/前景概率。经验90%半径和置信分数不是真实误差上界。Validation只有6位置，且Checkpoint已使用同一Val选择，故不能称严格独立Conformal校准或真实道路保证。

预声明30%覆盖、20匹配、至少3位置、P95≤5m、>5m≤5%，并在相同目标上优于Fixed Plane及Dense两摘要；两个种子没有共同通过的阈值。冻结阈值0仅保留研究估计，不代表准入。无法定位候选保留DetectedNoLocation；本轮任何输出都不得为VerifiedContact。

## 七项决策

1. **可以不依赖接触Gold训练Actor中心。** 仿真定义中心与接触合同分离，真实GPU有限训练已完成。
2. **没有稳定优于全部冻结基线。** 相对区域Dense有收益，但未稳定优于Fixed Plane和合法区域几何先验。
3. **置信度尚未满足选择性定位要求。** 经验校准与风险曲线已完成，不能同时满足覆盖与尾部门禁。
4. **行人、遮挡、坡道尚无充分收益证据。** 分层保留，样本和位置支持不足，坡道新测试范围有限。
5. **候选实例质量是限制之一。** 合并/分裂与漏检单列；唯一关联子集仍有跨位置误差，不能把全部失败归给候选实例。
6. **停止当前600步Head路线的自动扩展。** 保留模型和参考接口；若另立研究，应先补独立目标尺度/车型/遮挡支持并与几何先验公平比较。
7. **不批准Offline Object-Center Query Pilot。** 当前输出合同可作离线诊断，但位置准入与可信度未满足预声明门禁；更不涉及正式SparseDrive/Occupancy/Temporal/控制。

## 工程与运行预算

实际RTX4090离线六摄：冷帧 614.5ms，稳态P95 32.9ms，峰值PyTorch分配 293.1MiB。输入预加载GPU，不含磁盘/H2D/IPC。无CARLA竞争，不外推在线20Hz。

M5-M Runtime Partial继承，本轮未重复静态Shadow压力矩阵。完整命令见`docs/M5N_IMPLEMENTATION.md`，原始日志/Checkpoint/SHA保留。

补充：新封存52帧/4位置、75个已知可见Actor-view；全部配对与Snapshot检查通过。盒中心和新显式local参数的最大中心差约0.000029m，验证的是仿真Box定义，不是Mesh真值。

Validation误差预测与实际误差Spearman约0.412/0.101，封存约0.391/0.235，跨种子稳定性不足。按事后排序诊断能同时满足5m尾部和严重错误率的最大覆盖仅Val17.2%/22.6%、封存9.3%/13.3%，均不能作为Test新阈值。

几何先验在封存也存在Max50.99/66.52m极端错误，不能据平均值较低升级它；本轮任何分支都没有获得规划许可。

[不确定性与真实误差](m5n_figures/m5n_uncertainty.png)，类别、遮挡代理、坡度与Empty/Removed完整分母见`m5n_stratum_support.json`。

测试：310 passed, 6 skipped in 21.42s；SDK：15 passed in 0.21s。

图：[训练与泛化](m5n_figures/m5n_training.png)、[风险覆盖](m5n_figures/m5n_risk_coverage.png)、[方法对比](m5n_figures/m5n_localization.png)、[参考语义](m5n_figures/m5n_reference_semantics.png)。
