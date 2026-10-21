# M5-I — Reliable Visual Representation Transfer & Depth Qualification

**总体 Partial。** 实际完成CARLA支持扩充、RTX4090受控输出迁移与轻量架构对照（特征迁移按预声明条件决定是否执行）、只读Replay Depth资格验证、封存评测和联合Shadow。原默认网络、48LID、HeightSplat、四层BEV、控制器/Physics均未修改。

完整决策见[Architecture Decision Record](../docs/M5I_ARCHITECTURE_DECISION.md)，复现命令见[实现文档](../docs/M5I_IMPLEMENTATION.md)。所有独立状态见`m5i_acceptance.json`。

## 基线与控制

原ResNet18是官方ImageNet1K V1初始化，再经M4 CARLA训练，并非从零初始化；原权重/Checkpoint/manifest SHA验证通过。M5-H正区KD没有可靠背景监督，原GT高度重叠；Replay整体Bias负向变化已确认。

本轮新增4训练/2验证位置，实际运行的A/B/架构组使用相同共享数据与1632训练视角的完整首轮曝光采样，同预算2400、种子101/102。A=GT，B=GT+可靠正负软KD，C=GT+独立教师FPN特征（仅B条件失败才执行），Mobile=官方MobileNetV3Large ImageNetV2+GT。未组合多个KD。冻结Backbone缓存减少成本，实际只训练FPN/PixelHead，C未执行；不是完整Backbone微调上限。

B正类要求可靠GT vehicle/person与教师同意>=.7；负类必须GT可靠非目标且教师<=.05。漏检GT正类、cone、Unknown不作负KD。C的特征对齐与训练适配层已实现并通过合成梯度测试，但B在两seed均满足预声明验证收益条件，因此未生成教师特征缓存、未执行C训练，也不声称C实际有效。教师输出缓存不使用GT。

测试新六位置，Town01两处加Town03/04/05/10HD，每处16配置；Town02无45m隔离候选，Town06本机未安装，地图选择失败日志保留。全新Test在模型/阈值冻结后采集，M5HTest未作选择。

## 二维与架构对照

|模型|seed|P|R|AP|背景FPR|Removed局部FPR|Actor-view漏检|
|---|---:|---:|---:|---:|---:|---:|---:|
|gt|101|48.90%|43.99%|39.30%|0.24725%|0.76%|44.12%|
|gt|102|50.75%|43.93%|37.04%|0.22932%|1.11%|41.91%|
|output_kd|101|55.74%|41.29%|43.06%|0.17635%|0.58%|47.79%|
|output_kd|102|54.69%|43.32%|40.00%|0.19302%|1.00%|43.38%|
|mobile_gt|101|74.61%|87.72%|85.34%|0.16058%|0.26%|7.35%|
|mobile_gt|102|67.62%|87.34%|78.52%|0.22499%|0.37%|5.88%|
|M5G_fpn_bce|101|43.29%|22.48%|19.87%|0.15844%|1.18%|73.53%|
|M5G_fpn_bce|102|50.96%|26.85%|28.27%|0.13899%|0.56%|66.18%|
|COCO_teacher|0|95.52%|66.23%|94.00%|0.01538%|0.00%|25.00%|

Teacher主行只支持vehicle/person；严格同类Mask比较见supported。主阈值和低FPR工作点仅由Val选择。场景相关性不按百万像素计独立样本。各camera/类别/距离和逐场景曲线在原评测JSON。

## Depth只读资格

本轮零Depth训练、零温度/全局Bias校准。固定M5H Replay101 step1000、Replay102 step1200；默认原Head。近场每scene≤1.05×原Head；全局绝对Bias最多比原增0.25m，10–20/20–35段最多增0.5m；空间质量/Oracle投影/远场支持独立验收。

|模型|seed|全局MAE m|全局Bias m|近场MAE m|20–35m语义前景MAE / n|20–35m同层1m质量|
|---|---:|---:|---:|---:|---|---:|
|original|0|2.113|-0.131|0.812|8.132270406793666 / 135|3.63%|
|replay|101|1.847|-0.657|0.487|6.620887994766235 / 135|2.89%|
|replay|102|1.841|-0.593|0.496|6.432896698845758 / 135|3.02%|

Expected MAE为Softmax概率期望光轴深度；Mode/GTBin/层/0.5,1,2m质量/范围外质量分别保存。主表actor scope沿用语义前景Mask；具体Actor关联子集另见`m5i_associated_actor_depth_test.json`，资格额外要求该子集支持与空间质量通过。有效射线去重按sample-camera-UV，仍是相关表面观测，不称统计独立。20–35支持门禁≥100严格射线、≥4位置、≥2类型；缺少任一项不得泛化通过。

资格决定：[{"seed": 101, "qualified": false}, {"seed": 102, "qualified": false}]。逐场景近场、原S16任务、Bias范围/相机/天气、空间与支持独立原因见`m5i_depth_qualification.json`和`m5i_depth_bias.json`。

## BEV联合Shadow

|seed|组合|同层1m Recall|可见表面Precision|确认背景FPR|
|---:|---|---:|---:|---:|
|101|new_old|43.86%|13.87%|0.73%|
|101|new_replay|43.83%|14.03%|0.79%|
|101|old_old|16.71%|2.71%|1.28%|
|101|old_replay|17.31%|3.01%|1.30%|
|101|oracle_oracle|72.06%|100.00%|0.00%|
|101|teacher_oracle|70.82%|17.98%|1.02%|
|102|new_old|41.71%|10.75%|0.96%|
|102|new_replay|42.01%|11.29%|0.98%|
|102|old_old|24.42%|8.00%|0.65%|
|102|old_replay|25.42%|7.95%|0.71%|
|102|oracle_oracle|72.06%|100.00%|0.00%|
|102|teacher_oracle|70.82%|17.98%|1.02%|

new是Val固定选择；若没有合格Student，明确只是最佳诊断候选，不能叫合格部署模型。所有学习推理RGB+合法标定，无GTMask。Oracle仅离线，Teacher不是实时Student。背景表面不是安全Free，Unknown不当负样本。粗体素中心3D误差不是精确Actor定位。

## 六摄计算预算

|模型|P50 ms|P95 ms|峰值MiB|
|---|---:|---:|---:|
|gt_101|7.62|7.71|181.4|
|gt_102|7.40|7.69|181.4|
|output_kd_101|7.40|7.56|181.4|
|output_kd_102|7.43|7.50|181.4|
|mobile_gt_101|6.68|6.79|145.1|
|mobile_gt_102|6.70|6.89|145.1|
|M5G_fpn_bce_101|7.60|7.82|222.5|
|M5G_fpn_bce_102|7.80|8.17|221.5|
|joint_101|18.59|18.71|221.5|
|joint_102|17.50|17.88|221.5|
|COCO_MaskRCNN_sequential6|807.24|812.74|739.8|

独立GPU计时，不含磁盘、渲染、Controller；不等于严格20Hz驾驶。源码/配置/权重/数据/报告SHA与日志均保存，旧资产全量复核。Runtime继承Partial；纯视觉安全未验证。

停止规则：本轮没有测试后追加训练、改Loss、换Bin/Splat或升级默认。如果输出KD、特征KD与轻量候选都不能稳定改善，停止继续叠加蒸馏Loss，优先审查数据域、目标分辨率与监督合同。有限Occupancy Pilot须新决策，当前不自动开启。

## 最终决策与关键限制

- **视觉架构：建议下一阶段对 MobileNetV3 感知分支做独立迁移验收，暂不改默认 Encoder。** 在固定Test的6个位置，Mobile两个seed的AP均6/6改善，P/R同时改善分别6/6、5/6；但六摄P/R同时改善仅5/6、3/6，左侧Precision不足，不能宣称全六摄稳定通过。
- **输出蒸馏：有输出校准/排序收益，没有全面检测收益。** 两seed AP、Precision、Removed局部FPR改善，但Recall下降、Actor-view漏检升高；满足预声明B验证条件所以C不执行。没有测试后追加模型或Loss。
- **Depth：Replay不具备升级资格。** 所有逐场景近场和原S16回归通过，但全局/20–35m负Bias超限，新Test中远场同层1m质量下降。严格可关联Actor射线只有87条（6位置、vehicle/person），低于100门槛；语义前景135条不能充作135条关联Actor证据。
- **MAE与空间质量并不一致。** 新Test关联Actor的20–35m MAE由10.13m改善至7.74/7.49m，但同层1m质量由3.09%降至2.80%/2.78%；近场关联Actor MAE仍3.56/3.78m，同层1m质量也降。总体近场MAE受背景数量影响，不能代替障碍定位质量。
- **下一阶段优先视觉分支迁移与几何/Lift研究。** 新视觉显著提高BEV表面Recall，但精确表面Precision仍低；Replay附加收益小。先补足关联Actor的可靠中远场支持、评估真实采样密度和概率空间落点，不自动追加KD或Depth训练。正式Occupancy规模化、Temporal、Sparse Query和视觉控制均不启动。

本轮未使用新的独立Mesh真值。CARLA可见渲染首表面、实例关联及粗网格参考有共同来源和离散化限制。极小目标3–32像素仅6个Actor-view，全部模型该子集Recall均5/6，支持不足以判定稳定小目标优势。类别分项的Precision是二值目标头在“该类别正类+共享非目标背景”下的条件Precision，其他目标类别忽略；它不是多类分类混淆矩阵。

Oracle2D+OracleDepth同层1m Recall仍只有72.06%，因为参考来自全分辨率有效表面，而Lift使用固定P8采样。这是采样/观测覆盖限制，不能把所有未命中都归因于Depth学习。Teacher+OracleDepth使用原始软分数及固定0.2 BEV事件阈值，并非二维0.9工作点先二值化，低分背景累积可导致大量BEV误检；不能把该17.98% Precision解释为Teacher二维识别质量骤降。所有这些结果都是表面事件质量，不能解释为可通行Free或语义Occupancy概率。

实际训练6组×2400更新，两seed；每组约46–49s（含周期评测和Checkpoint写入，不含初次缓存加载），共享训练缓存下峰值约10.7GiB。单次推理峰值、预处理/Teacher成本另列，不将缓存训练耗时当成端到端训练总耗时。测试274通过、5跳过；CARLA SDK环境23通过为重叠子集，不能相加。跳过项及完整原始日志保留。Runtime继承Partial，Oracle驾驶未重验，纯视觉驾驶未验证。

图像见 [学习曲线](m5i_figures/m5i_learning.png)、[同类PR](m5i_figures/m5i_precision_recall.png)、[Depth偏置](m5i_figures/m5i_depth_bias.png)、[六组合BEV](m5i_figures/Town05_itest10_m5i_bev.png) 和 [后向视觉](m5i_figures/Town05_itest10_m5i_rear.png)。分层证据见 [补充诊断](m5i_findings.md)。
