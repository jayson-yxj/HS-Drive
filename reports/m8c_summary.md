# M8-C — PETR Reference Geometry & Data Support Audit

**诊断完成，质量仍为Partial；下一步先核查预训练输入分布合同并补足评价标注，不批准训练。**

基础 `af4266be5859e55db51e77e4b1b15ef122fce61f`，独立分支 `feat/m8c-petr-reference-distribution-audit`。本轮实际执行12个固定六摄样本RTX4090 Hook观测，192帧历史输出/GT离线审计，26帧154个背景候选审计，以及合成几何和回归测试。训练0步、新CARLA闭环0次、B参考模型对照0次。原M8-B所有结果保持不变。

| 问题 | 实际结论 |
|---|---|
| A ReferenceGeometry | **新发现相机序号编码差异**：官方converter顺序与HS顺序不同，官方多视角位置编码依赖camera index；5个相机的index编码不同。K/T配对正确仍不能保证预训练分布一致。原训练PKL顺序尚未实证，模型影响未做消融。nuScenes LIDAR_TOP/Camera实测标定未获得，不能确定高度偏差。 |
| B ProjectionCorrectness | 保留原HS投影闭合；合成T_E_R下投影最大残差3.55e-15m，box/yaw/速度逆变换通过，倾斜参考不允许暗中压成yaw。实际物理B缺依据，**ReferenceVariantNotRun**。 |
| C DepthFailure | 原92投影代理中只有6个达到预声明多视角实例支持，86仍是代理；另外63目标无原规则配对。6个支持车辆均14–20m，XY mean5.34m、P95 7.93m、>5m 2/6；径向有符号均值+5.32m、光学深度绝对4.08m、横向绝对3.39m。局部偏远现象成立，不能据此概括远场。20–40m为32代理+51无配对，可信身份0，因此远场主要深度还是识别失败**仍无法确证**。 |
| D AppearanceSupport | 实际水平FOV约100°，源640×384。20–40m可见像素中位295px，裁后原图高度中位19px；放大不增加信息。48/265视角部分裁剪、无整目标完全丢失；14.26%像素损失不是全部远场失败的充分解释。无同口径nuScenes RGB，色彩/曝光域偏移未归因。 |
| E AnnotationQuality | 154个固定背景候选：126 AmbiguousOrUnlabeled、23 UnsupportedClass（标注不含barrier）、5 DuplicateHypothesis。不能外推11045未匹配为物理FP。AI助手查看所有图页，未假装成人类认证；身份和完整背景真值不足。 |
| F FrozenReferenceAblation | **NotRun**。官方教程可追溯标定样例是RADAR_FRONT，不是LIDAR_TOP，不拿0.5m当LiDAR高度。12次Hook结果与M8-B对应帧逐位一致，无改进或性能因果声称。 |
| G TrainingReadiness | **NoGo**。先明确camera index约定、取得真实参考资料、补足远场身份与背景标注，再申请单变量诊断；当前不具备直接批准域适配训练的依据。 |

实际位置编码：6×44×16×64×3几何采样，64个深度1–59.3477m。每摄23.2%–27.0%的几何位置超出position_range，20–40m采样约44.74%越界，主要Z。computed geometry mask占1.99%–19.32%，**官方forward丢弃该返回mask**，仅使用pad mask。越界坐标经inverse_sigmoid夹边界；这不是目标概率质量，没有nuScenes同口径分布便不能认定异常高度原因。

新增13项测试、M8-B18项合同回归和历史42项测试通过。保留速度旋转测试失败/修复、报告关键字失败、官方网络访问失败，以及barrier语义字段更正；均未用于训练或默认模型。Hook性能本会话约510ms，无Hook约503ms，额外4.6–10.1ms；不同于M8-B53ms，跨会话差异未归因，不覆写旧时延和实时门禁。

可复核结果：

- [6个身份支持的多视角图](m8c_figures/identity_supported.png)
- [远场投影代理图](m8c_figures/far_proxy.png)
- [像素支持、位置编码和身份分级](m8c_figures/support_position_grades.png)
- [全部背景图页清单](m8c_visualization_manifest.json)
- [参考源码与可得资料审查](m8c_nuscenes_reference_audit.md)
- [验收独立状态](m8c_acceptance.json)
- [复现与实现](../docs/M8C_IMPLEMENTATION.md)
- [下一阶段决策](../docs/M8C_ARCHITECTURE_DECISION.md)

最终状态：`DiagnosticCompleted_QualityPartial / ReferenceDistributionMismatchSupported_CameraIndexEncoding / ReferenceIssueNotEstablished_Height / AnnotationSupportInsufficient / TrainingReadinessNoGo`。默认感知、Planner/Safety/Controller/Physics及所有历史门禁不变；不合并main、不连接视觉控制。
