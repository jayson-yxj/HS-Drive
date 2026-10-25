# M8-D — Frozen PETR Camera-Order Contract & Paired Ablation

**相机顺序会改变PETR预测，但指定B排列没有获得检测或可信定位改善。保留A，不批准训练或模型采用。**

基础提交9af1cb2，独立分支feat/m8d-petr-camera-order-ablation。实际RTX4090执行12帧Hook A/B和192帧完整A/B；固定原Checkpoint、FP32、预处理、参考坐标、900Query/6Decoder/10类。全部A重放与M8-B的Box/Score/Class逐位一致；12帧全部Decoder输出还与M8-C逐位一致；558权重张量不变。历史SHA全部保留。

| 指标 | A 原HS顺序 | B OfficialConverterOrderCandidate |
|---|---:|---:|
| 原GT / 位置 / 帧 | 155 / 38 / 192 | 相同 |
| 2m XY + 2m Z，score>=0.1 TP | 45（29.03%） | 45（29.03%） |
| FN | 110 | 110 |
| 20–40m TP | 1/83 | 1/83 |
| 4m XY敏感性 TP | 56 | 56 |
| score>=0.1候选 | 11090 | 10837 |
| Annotation-Unmatched | 11045 | 10792 |
| 严格身份支持 | 6 | 5 |
| 共同5对象XY mean | 4.631m | 5.061m |
| 共同5对象XY P95 | 5.083m | 6.034m |
| 共同5对象>5m | 1/5 | 2/5 |
| 20–40m严格身份支持 | 0 | 0 |

主2m指标逐GT为共同成功45、A丢失0、B新增0、共同失败110；38位置主TP均不变，Vehicle29/105、Person9/30、Cone7/20不变。4m指标共同成功55、丢失1、新增1、共同失败98。不能把候选减少253当作确认物理误检减少，因为静态背景标注仍不完整。

共同身份5对象中1个XY略改善、4个变差，样本不足以作总体显著性判断。s111在B中front mask recall从0.8288变为0.7972，低于冻结0.8，降为ProjectionProxyOnly；没有调整门限。A全部6对象P95为7.93m、B全部5对象6.03m，**分母不同，不能当改善**。原代理口径A92/B91，P95为18.57/18.62m，仍无可信远场身份，不能声称解决深度失败。

## A–G 独立回答

- **A 来源：TrainingPKLOrderNotVerified。** 官方converter→字典/序列化→Dataset→augmentation/Collect3D顺序可追踪，并通过合成mmcv序列化+真实Dataset调用；实际训练info PKL没有取得。B只能称OfficialConverterOrderCandidate。其他sweep流程顺序不同，未移植或作为第三组。
- **B 合同：PermutationContractPassed。** 192帧六摄RGB SHA多重集合一致，K/T/增强K/投影/时间同一物理相机不变，逆排列完全恢复，解析投影残差0。无参考系变换。
- **C 敏感性：OrderEncodingChanged / OrderEffectOnPredictionsSupported。** 按物理相机对齐的几何位置编码逐位相同，5摄序号正弦编码不同；192/192帧Box和Score有变化。完整900Query×6层、300TopK保留。
- **D 质量：OrderChangeNoQualityGain。** 2m主检测无得失；共同身份定位未改善。没有可复核的正向采用证据。
- **E 一致性：无跨类别/距离/位置的正向检测趋势。** 主检测各组完全相同；5个共同身份样本不能证明总体退化或显著性。
- **F 限制：FarFieldIdentityInsufficient / AnnotationSupportInsufficient。** 远场A/B均32代理、51无配对、严格身份0。多视角自动规则不是人工身份认证；无新独立最终集。
- **G 下一步：优先申请数据与身份/背景标注补强，TrainingReadinessNoGo。** 不继续顺序搜索，不把一次负结果包装成域适配依据。合法实际训练PKL仍可作为来源核查补充；有限训练需要独立数据/评价提案。

## 运行时、验证和证据类型

实际全量无Hook：稳态前向P95 A503.28ms / B502.95ms；本轮E2E P95 A1023.59ms / B1134.07ms。B排列和显式Tensor复制配对平均多47.54ms，模型前向配对平均差0.046ms。GPU峰值allocated约1.03GiB（reserved另列报告）。这些是本次同步审计环境计时，不覆盖M8-B约338ms E2E；跨会话前向速度差未归因，Online NoGo及共享GPU门禁不变。

16项M8-D测试、18项M8-B和13项M8-C回归通过；官方链条合成测试和192帧实际输入合同另列。没有新CARLA采集/闭环，没有训练/微调、默认替换、视觉控制。首轮small实验因extract_feat原地改Tensor shape失败，保留原源码和日志后修复显式view，未修改PETR。全部指标来自记录CARLA数据离线评价和实际GPU推理；合成测试不代表闭环。

- [完整验收](m8d_acceptance.json)
- [配对检测及38位置](m8d_paired_detection_quality.json)
- [严格身份和共同子集](m8d_paired_identity_support.json)
- [全部条件定位字段](m8d_paired_localization.json)
- [逐层输出及位置编码](m8d_gpu_output_difference.json)
- [共同身份图](m8d_figures/common_identity.png)、[边界样本](m8d_figures/s111_identity_threshold.png)、[38组与类别距离](m8d_figures/group_and_strata.png)
- [复现命令](../docs/M8D_IMPLEMENTATION.md)、[架构决策](../docs/M8D_ARCHITECTURE_DECISION.md)

最终：DiagnosticCompleted_QualityPartial / OrderChangeNoQualityGain / TrainingReadinessNoGo。感知、安全、Occupancy与Runtime历史未通过门禁全部保持。
