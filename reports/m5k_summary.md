# M5-K — Target-Aware Depth Localization & Adaptive Spatial Sampling

**总体 Partial；默认架构和控制系统未改，Occupancy 联合 Pilot 不批准。**

## 实际执行与支持

实际 RTX4090 无训练对齐诊断、CARLA 新 Train/Val/Test 各4位置各104帧（每位置26配置），均与历史位置隔离≥45m。新测试在科学配置冻结后采集一次。可靠可见首表面参考不是独立 Mesh 真值。

|集合|20–35m严格Actor P8射线|位置|Vehicle/Person/Cone|训练支持门禁|
|---|---:|---:|---|---|
|train|179|4|{'vehicle': 179, 'person': 0, 'cone': 0}|False|
|val|152|4|{'vehicle': 152, 'person': 0, 'cone': 0}|False|
|test|159|4|{'vehicle': 159, 'person': 0, 'cone': 0}|False|

训练决策：**NoTraining_SupportInsufficient**。严格 Mask 与最低支持未调整。

## 固定射线 Depth 与 BEV

|方法|Actor20–35m MAE|Actor20–35m 1m概率|Mobile101 P/R1|Mobile102 P/R1|
|---|---:|---:|---|---|
|original_p8|8.779m|2.96%|0.37%/4.24%|0.29%/4.38%|
|native16_to_p8|8.423m|2.76%|0.40%/4.92%|0.32%/4.79%|
|replay101_p8|6.885m|3.45%|0.25%/4.65%|0.23%/4.92%|
|replay102_p8|6.685m|3.64%|0.25%/4.65%|0.23%/4.92%|

以上所有 P8 深度指标均用严格且 GT 在 BEV 内的同一射线分母。Mobile LearnedDepth 正式投影不接受 GT Mask。S16 原生密度另报。Replay 仍是历史诊断，不升级。

## 同预算采样

|seed|Depth|P8 R1|S4 R1|Adaptive R1|Uniform R1|
|---:|---|---:|---:|---:|---:|
|101|oracle|41.72%|57.87%|58.41%|41.72%|
|101|original|4.24%|22.71%|25.72%|4.24%|
|101|aligned|4.92%|23.12%|25.72%|4.92%|
|102|oracle|42.13%|60.47%|59.37%|42.41%|
|102|original|4.38%|24.21%|25.72%|4.38%|
|102|aligned|4.79%|25.31%|28.59%|4.79%|

Adaptive 与 Uniform 逐帧额外射线数精确相等；没有 ROI 的目标仍在分母。Learned S4 只是概率插值；当前 masked dense kernel 未实现稀疏加速。Oracle/采样改善不自动代表真实 BEV 资格。

## 输入路径与验证

端到端逐次测量，包含转换/H2D/完整推理，非相加模块 P95。实际 BGRA 传感器数据回放不含在线 tick/队列/控制。

|seed|路径|P95 ms|Max ms|>50ms /40|
|---:|---|---:|---:|---:|
|101|file_sha|117.32|133.83|40|
|101|memory_float|40.41|44.04|0|
|101|memory_pinned_uint8|38.74|42.27|0|
|101|native_bgra_pinned|46.81|48.78|0|
|101|torch_serialized|47.10|55.32|2|
|102|file_sha|118.21|129.68|40|
|102|memory_float|37.53|46.89|0|
|102|memory_pinned_uint8|40.21|42.52|0|
|102|native_bgra_pinned|44.01|47.72|0|
|102|torch_serialized|55.53|56.04|6|

全量测试与 SDK 专项结果在 `m5k_tests.json`；原始失败记录保留。历史资产完整性在 `m5k_artifact_manifest.json`。

## 七项结论

1. S16→P8 对齐不能仅凭 MAE 下降升级；正确空间质量和两个 Mobile 种子的真实 BEV 指标决定资格。无法从此干预给出“全部错误中多少百分比由特征分布导致”的因果比例。
2. 当前没有同时通过近场、Bias、Actor 空间质量与真实 BEV 门禁的升级候选。
3. 支持门禁未通过，Mobile 条件残差未训练；其有效性仍未验证，不能称无效。
4. 自适应采样研究决策：同预算收益未稳定满足门禁，保持 P8。
5. 深度 MAE 与空间概率/BEV Precision、Recall 并不等价；详见同射线配对表，不使用 Oracle2D 单项改善替代实际 Mobile 验收。
6. 文件读取/SHA 和每帧序列化可以从未来在线输入路径移出；实际回放结果支持优化方向，但不证明严格实时驾驶。
7. 不批准 Occupancy 联合 Pilot；先解决报告中的空间支持及定位门禁，再单独申请有限规模研究。

复现见 [实施文档](../docs/M5K_IMPLEMENTATION.md)，定义与门禁见 [架构决策](../docs/M5K_ARCHITECTURE_DECISION.md)。逐帧输出与完整 SHA 索引保留服务器，Git 仅精简报告与图片。

## 验收复核与下一步

自适应覆盖收益在 Val/Test 的两个种子上方向一致；No-go 的明确原因是已确认背景 FPR 增量超限，而非没有采样收益。Test 原 Depth 的 Adaptive Recall 25.72%，Uniform 4.24/4.38%；背景 FPR 从 0.4008/0.5014% 增至 0.6555/0.6706%，增量 0.2547/0.1691pp 超过 0.1pp。Precision 虽升至 1.92/1.90%，绝对质量仍很低。

原 M4 近场相同 P8 射线上的 MAE 1.360→1.108m，说明 S16 原生分布对近场有作用；远场 Test MAE 8.779→8.423m 仅改善约4.1%，同时同层1m概率 2.96→2.76%。该实验不支持“直接对齐即可解决目标定位”的解释。

中远场行人 Train/Val 原生语义像素1106/1108，可靠内部像素335/326，固定 P8 中心可靠命中0/0。下一阶段首先补充不同方位、距离和目标投影相位下的可靠行人表面；不放宽边界规则凑数，不把重复相同UV布局视为独立支持。达到两类别支持后才重新申请600步残差 B/C 对照。

新测试仅4位置，Actor布局与天气方案跨 split 复用；空间背景构成与M5-J不同，不能把跨轮聚合数值差异直接归因为模型退化。真实表面定位、背景错误和Unknown需继续分开审查。Residual训练路径本轮未执行；工程Passed限定实际执行接口，不认证条件训练代码的完整收敛。

诊断图：[概率与空间落点](m5k_figures/Town01_ktrain0_m5k_ray.png)、[同预算采样](m5k_figures/m5k_sampling.png)、[输入完整延迟](m5k_figures/m5k_input_latency.png)。
