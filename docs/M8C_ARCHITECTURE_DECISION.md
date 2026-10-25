# M8-C Architecture Decision

结论：`DiagnosticCompleted_QualityPartial / TrainingReadinessNoGo`。根据实际冻结模型Hook、全192帧历史输出审计、26帧背景审计及官方源码核查作决定；无训练或视觉控制。

| 独立问题 | 决策 |
|---|---|
| A ReferenceGeometry | 相机index编码与官方转换器次序存在可验证差异。LiDAR原点高度/六摄nuScenes真实K/T未获得，ReferenceIssueNotEstablished用于高度因果；不能用RADAR_FRONT标定替代。 |
| B ProjectionCorrectness | HS旧投影合同保留；合成刚体变换、非方形box/yaw/速度逆变换通过。真实B参考未定义，ReferenceVariantNotRun。 |
| C DepthFailure | 6个多视角身份支持车辆样本有明显径向偏远，XY均值5.34m；仅14–20m样本。远场32代理/51无关联，无严格身份支持，不能证明远场主要是深度误差。 |
| D AppearanceSupport | 20–40m原图可见面积中位295px、裁后高度中位19px，采样稀疏有实证。所有已知目标仍有像素；裁剪不是整目标消失原因。色彩/曝光域偏移因无可比nuScenes RGB队列仍Unresolved。 |
| E AnnotationQuality | 154个固定未匹配候选：126不确定、23缺失标注类barrier、5重复几何假设；不能外推全部11045未匹配为物理FP。身份确认与多Actor支持不足。 |
| F FrozenReferenceAblation | NotRun，没有可信T_E_R。Hook只是观测，12帧输出与M8-B逐位相同，不是改进实验。 |
| G TrainingReadiness | NoGo。先核查输入次序/真实标定并补标注与独立数据支持，再考虑有限适配训练。 |

下一步优先级是**修复/核实预训练输入分布合同并补足评价标注**，不是增加训练步数。最小新提案应先取得原训练数据转换顺序证据，单独比较HS当前顺序与官方次序（图像/K/T一并排列，其他不变），以固定Train开发/Val描述性规则衡量；不可同时移动参考高度或调整裁剪。该对照本轮没有运行，也不自动授权下一阶段。

并行的资料准备可以获取合法nuScenes metadata，验证LIDAR_TOP及六摄真实K/T；需有该资料且明确ego原点关系，才能提出唯一物理B参考。数据方面补全道路设施/背景类别、为远场和右侧提供可核验身份、增加遮挡多Actor支持并保留新独立最终位置。当前6个身份样本不能被包装为完善的定位benchmark。

默认Encoder/Depth/48 LID/HeightSplat/Planner/Safety/Controller/Physics保持原样。M8-B及所有历史质量、Occupancy、Runtime和视觉驾驶门禁不变。没有PerceptionPassed或SafetyValidated。
