# M8-F — Grounded Scene Generation & Perception Dataset Quality Gate

结论：**实现了可拒绝的落地质量门禁，但两个Development Smoke未通过；停止扩采，TrainingReadinessNoGo。** 没有通过增加数量、放宽接触或身份阈值获取Passed。

基线12bd124b443cc01c60ee4251fd1be20d868928c1，分支feat/m8f-grounded-scene-quality。本轮M8-E全部资产只读，默认模型、Rig、Planner/Safety/Controller/Physics Control未修改。没有PETR前向、训练或驾驶闭环。

## 七项验收

| 项目 | 实际结果 | 状态 |
|---|---|---|
| A GroundPlacement | 8个请求均保存15点×2姿态射线；不命中或不合格直接拒绝，无固定Z回退；唯一生成目标又做30次最终足迹射线 | NoFallbackImplemented；有限代理验证 |
| B NaturalScene | 7个请求生成前拒绝，只有1个行人通过；两处均未获得合格车辆 | **SmokeNoGo** |
| C RenderIdentity | 1个单视角行人，Present 65像素、Shifted 30像素；实例出现/移除检查通过 | SingleViewSupported；新多视角未演示 |
| D Background | 保存完整环境库存，按12个相机视图输出逐像素审核Mask；歧义与Unknown保留 | RegionLimitedIdentityPartial |
| E FarField | 1个约28.33m的自然几何代理+渲染身份目标；Shifted约31.30m不重复计数 | 1个局部支持、**0个独立训练合格**；远低于30目标 |
| F Independence | 新Development均与历史场景包络重叠，历史覆盖不完整；补查发现部分请求距旧目标足迹3.6–4.6m | HeldoutIndependenceNoGo；新Final为0 |
| G TrainingReadiness | Smoke、类别覆盖、独立性、背景标注均不足 | **TrainingReadinessNoGo** |

## 实际采集与拒绝

CARLA0.9.15、RTX4090渲染、Epic、原640×384六摄。冻结Town02两个Development位置，每组Empty/Present/Shifted/Removed，实际8帧、48张RGB、192个RGB/Depth/Instance/Semantic载荷，时间/帧号/SHA全部通过。后续Validation、Calibration、Final均NotRun。两个Development是调试支持场景，并未被称为独立新分布。

| 场景 | 车辆 | 行人 | Cone | Barrier |
|---|---|---|---|---|
| Development_00 | NONE语义的上方命中，拒绝 | 约7.9m树冠命中，保守拒绝 | 部分足迹无合法Lane/静态Box重叠 | 部分足迹无合法Lane，Shifted另有静态重叠 |
| Development_01 | 上方NONE命中及部分足迹无Lane | **通过请求、最终支撑及单视角身份** | 部分足迹无合法Lane | 部分足迹无合法Lane |

这里的“拒绝”不能全部解释为真实危险。树冠高于行人体积仍触发当前保守规则；NoLaneAtFootprint可能包含OpenDRIVE与渲染边界不一致；NONE碰撞命中的来源尚未解释。场景选择仍未做到类别足迹可行性预筛，本轮选择了有直路和人行道的位置，却未能在固定请求下得到可靠车辆样本。这些都是尚需修复的工程限制。没有据此放宽阈值继续采集。

唯一生成行人的两姿态实际OBB底部15点支撑间隙约0.030m，请求/实际姿态一致，Footprint位于Sidewalk，独立语义一致率1.0。只能称GroundedByGeometryProxy；不能以物理关闭后保持不动证明脚底接触，也不能从有限点推断完整连续Mesh支撑。

在真实CARLA中补测了完整车身越界、无命中、建筑区域三个拒绝探针，均拒绝；未在非法位置生成Actor。另外在历史可疑XY进行了新的纯几何查询：Town04两处命中Water，被拒绝；Town05旧“疑似悬空”锥桶位置发现约8.1m高的Roads层，单条新射线可成立，仍没有证明历史脚底接触或修复旧采集。**新射线不是历史丢失记录的重建。** 真正多个合法道路高度层的拒绝目前仅有合成单测，不冒充真实桥下验证完成。

## 历史归因与空间隔离

对M8-E80个Present目标逐条保留GroundHitUnknown、SurfaceTypeUnknown、ContactNotVerified、渲染支持、历史重叠及自然资格未知等字段；原28单视角、18多视角、34Unknown、21远场支持全部保留。旧Ray Hit未保存，一律不补造。

采集前索引包含13,463条配置/监督/实测轨迹/M8-E几何记录、450个源文件索引；保留45条已识别覆盖缺口。配置区用80m保守场景/相机包络，缺方向的Box用包围圆，具有可用角点的目标使用足迹距离。它是保守排除工具，不是完整历史FOV恢复。

Smoke后补查6,190个M5原始帧，恢复3,099条去重Ego/Actor几何记录，单独保留Supplement。Development_01的Cone/Barrier请求距历史目标足迹分别约4.60m/3.64m，不能计为新独立样本。补查没有用于改选地点，也不能把原始preflight追认为全面。下一轮若提出新采集，必须把原始帧补充索引纳入采集前检查。

历史全部可用Town已重复使用；M6不同CARLA版本及未完整恢复几何的区域继续标记覆盖不完整。两个位置间距超过150m，仍不消除它们与旧场景的相似性。M8-E旧Held-out仅保留历史支持资料，未重命名为新盲测。

## 静态身份与独立质量字段

完整Actor、EnvironmentObject与匿名Level Box库存分别保留。通过实例/语义/深度与唯一源OBB闭合建立审核区域，不以数值相同连接ID。多个Box竞争仍Ambiguous；Unknown、Ignore均不等于Free。

静态区域审核Mask只覆盖实际有源证据、深度有效且裁剪后保留的像素，未审核补集不作为负样本。两处Present中还发现两个静态EnvironmentObject具备car语义和单视角源关联，但其落地与自然性未认证，因此没有纳入远场训练计数。历史154条候选仍只读，不用新环境库存倒填旧真值；仍不能报告完整物理Precision。

每个观察到的ObjectReference均输出RenderIdentityGrade、GroundSurfaceGrade、NaturalPlacementGrade、AnnotationGrade、SplitIndependenceGrade及TrainingEligibility。八个计划请求的拒绝也全部保留；未知背景没有因其他分数好而被晋级。GT保留在Evaluator Sidecar，RGB包保持原白名单。

## 验证与交付

原52项回归全部通过，新增17项测试通过，合计**69项**。另验证8个RGB包、48图，8次GT字段注入和3次评价文件访问探针被拒绝。文件访问守卫+网络namespace只防误用，不声称恶意原生代码沙箱。没有加载视觉模型。

六张诊断图完成AssistantVisualReview；没有独立人工审核。保存六摄图、四阶段目标图、俯视/3D射线图及最终支撑间隙图。失败保留：NumPy2的ndarray.ptp兼容错误发生在传感器采集前，修复为np.ptp；收尾报告一次重复字段错误也保留。未调整质量阈值。

全部原始数据位于/root/gpufree-data/hs-drive-experiments/m8f-grounded-quality。执行命令、源码SHA、日志和资产索引随本轮提交封存。CARLA已清理本轮Actor、恢复异步状态并停止独立服务器。

入口：[验收JSON](m8f_acceptance.json)、[落地审计](m8f_ray_hit_and_contact_audit.json)、[资格清单](m8f_object_eligibility.json)、[复现文档](../docs/M8F_IMPLEMENTATION.md)、[架构决策](../docs/M8F_ARCHITECTURE_DECISION.md)。

下一步只建议单独批准“小范围放置器可行性诊断”：解释NONE命中、核对边缘Waypoint查询、按目标实际高度检查上方遮挡，并先完成历史原始足迹的预采集索引。需要独立正负几何用例验证这些修正，不能把树冠忽略或Lane查询放宽直接用于生产数据。本轮不自动扩大采集或申请PETR训练。
