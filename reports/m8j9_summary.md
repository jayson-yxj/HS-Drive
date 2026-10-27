# M8-J9：冻结A缓存的官方nuScenes评测接口对齐

实际日期：2026-10-27 UTC。基础`136023d`，独立分支`feat/m8j9-frozen-a-official-eval`。A导出及首次官方评测源码`0df0a1e6874d6dc864fd97686e6831f82748b2cb`；附加审计修复及第二次评测源码`7af366b33e8adb7f84517cf342492b07d8ea2b66`（完整实际值以supervisor为准）。

**接口/协议通过。冻结A在mini_val上的本地官方devkit评测，使用指定缓存与输出策略：mAP=0.276715、NDS=0.325554。** 两个独立进程对同一结果文件的官方metrics_details逐字节相同，metrics_summary除eval_time外全部数值一致，包括官方不适用项NaN位置。新增模型前向、参数更新、反向均为0。没有重新评估B，八帧过拟合线保持关闭。

## 三个独立状态

|维度|结论|
|---|---|
|接口/协议|Passed：81个sample完整对应、字段来源明确、原导出器复用、官方评价器未修改、重复评测一致|
|缓存限制|足以完成此指定输出策略的官方评测；已有Top-300、score≥0.2和源XY范围限制，低分段已截断；不是完整模型重新前向的无阈值基线|
|实际质量|以下真实官方分数；没有设质量阈值，也不以分数高低决定接口是否通过|

这不是排行榜提交、完整nuScenes val成绩或广泛泛化认证，不能与论文全量结果直接比较。mini_val只有两个场景，连续帧高度相关；A的预训练来源限制沿用M8-J8记录，本轮未重新宣称完全未见。

## 固定输入、预测选择与导出

A身份沿用M8-J8：原始官方pth SHA `16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad`，安全tensor归档SHA `cf33dccb1d549c425a89cf346d6f05401c49f6cccbd5a579cb7d8937b5a6d591`。本轮只核对来源记录及81份A缓存SHA，**没有读取或加载checkpoint**。

实际核对本地`v1.0-mini`的官方`mini_val`：scene-0103 40帧、scene-0916 41帧，与M8-J8固定manifest逐token、scene和sample timestamp一致，缺失/重复/额外样本均为0。每个sample在结果JSON中均有条目；没有用空列表掩盖缓存缺失。

缓存保存最后一层完整900×10分类logits和框编码，以及官方NMSFree Top-300解码框、分数、类别、Query映射；原decoder中心范围为±61.2m XY及[-10,10]m Z。正式输出继续保留M8-J8的score≥0.2、strict |x|/|y|<51.2m，未改用低分缓存、未增加NMS、未补框。

- 原缓存24300条，原有score/XY过滤后8645条。
- 原MMDetection3D导出器按类别ego距离移除851条，输出7794条。
- 每sample输出66–141条，小于官方500上限；没有额外截断。
- 官方评价器再移除1条距离边界预测及10条自行车架内bicycle预测，最终7783条。这是官方评价行为，**不是导出端GT筛选**。

私有结果：`/root/gpufree-data/hs-drive-private/m8j9-frozen-a-official-eval-retry1/export/results_nusc.json`，SHA256 `7298432cf47d19694ad15ed09eb94f19ac9204596ce1d97ddb83b352322eb9d7`。保留原顺序及原分数，结果文件在两次评测前后完全未变。

## 字段和几何合同

直接调用PETR CustomNuScenesDataset继承的冻结`NuScenesDataset._format_bbox`，没有重写解码或导出属性规则。适配层只把缓存prediction tensor和原始元数据位姿组装为原接口所需类型。

|官方字段|来源及处理|
|---|---|
|sample_token / 类别|固定81帧token；保留原十类顺序映射，官方汇总仍按全部十类|
|translation|缓存LiDARInstance3DBoxes底面中心加h/2得到几何中心；按reference LIDAR_TOP自己的标定和ego pose变换到global|
|size|缓存完整w,l,h，米；不是extent半尺寸；不交换长宽、不用GT尺寸|
|rotation|原导出器恢复`yaw_nusc=-yaw_second−π/2`，构造wxyz四元数，然后依次乘lidar→ego、ego→global旋转|
|velocity|缓存预测列7:9，和同Query raw decoder列8:10精确一致；沿用原平面vz=0规则，旋转到global再输出XY，m/s|
|attribute_name|原导出器按预测global XY速度是否>0.2m/s及类别默认值确定，不是GT属性，也不是独立属性Head输出|
|meta|原配置camera-only：camera=true，lidar/radar/map/external=false；LiDAR参考坐标不是点云输入|

类别属性规则包括：高速车辆moving、两轮with_rider；低速car/truck等默认parked，低速bus stopped、低速pedestrian standing；其余使用原DefaultAttribute。记录的是规则结果，不能解释为视觉属性识别经过独立验证。

六张相机原先各自的时间戳/位姿已经体现在冻结前向的lidar2img中。本轮不重新生成相机输入，也不套用CARLA严格同步假设；global导出使用LIDAR_TOP keyframe自身的pose，而非随意使用任一相机pose。

独立矩阵/四元数链逐框核验原导出器：中心最大差1.20e-7m，尺寸0，旋转矩阵最大差1.63e-7，速度最大差1.78e-15m/s，分数0；均满足预声明1e-5。官方DetectionBox序列化往返通过。

导出只读取sample/sample_data/calibrated_sensor/ego_pose/scene/sensor表；不构造会自动加载GT的NuScenes，不读sample_annotation或属性/instance表。NPZ只访问预测键，不读同归档的gt/gt_labels。GT和权重访问canary均被拒绝。原始sample.json不含运行时`data`字段，补齐的是官方keyframe反向索引规则，不是GT推导。

## 官方GT和项目3559条GT为何不同

实际使用已有nuscenes-devkit **1.1.9**、`detection_cvpr_2019`，未升级PyTorch/MMCV/MMDetection3D。官方匹配距离0.5/1/2/4m、TP误差2m、min_recall=min_precision=0.1、max_boxes_per_sample=500、mean_ap_weight=5保持不变。

官方类别ego XY半径：车辆五类50m，行人/摩托车/自行车40m，cone/barrier30m。它不同于项目统一LiDAR XY方形范围±51.2m。由官方加载器读取原始GT并执行官方过滤：

|阶段|GT数量|
|---|---:|
|81帧原始标注（所有类别）|4514|
|官方十类加载后（排除73条非任务类）|4441|
|官方类别距离过滤后|3785|
|零LiDAR/Radar点数过滤后|3393|
|自行车架过滤后|3393|

项目有效3559条与官方3393条交集3393，项目独有166、官方独有0；这166条在本数据中均为官方类别距离排除。没有把3559硬塞给官方评测，也没有在模型输入中使用点数或GT。完整阶段及分类支持见[过滤审计](m8j9_GT_filter_audit.json)。166条距离排除逐条核验汇总见[距离差异](m8j9_radius_difference.json)；最小超出对应半径0.143m。逐annotation集合对照仅私有保留。

边界处的1条预测在导出器Ego XY内距半径约0.50mm，在评测器global XY相对Ego距离超半径约2.90mm；姿态roll/pitch引起两种平面距离差异，不能仅归因为`<`和`<=`。两套官方规则均未修改。

官方自行车架规则会使用标注环境几何过滤预测和GT，这是官方评测协议的一部分。它没有反馈给导出器、模型或控制。

## 实际分数与支持量

成绩名称：**冻结A在mini_val上的本地官方devkit评测，使用指定缓存与输出策略**。

|官方指标|实际值|
|---|---:|
|mAP|0.276715|
|NDS|0.325554|
|mATE（m）|0.820901|
|mASE|0.458738|
|mAOE（rad）|0.758446|
|mAVE（m/s）|0.787388|
|mAAE|0.302567|

|类别|官方GT记录 / instance|官方AP（四距离均值）|ATE|AVE|
|---|---:|---:|---:|---:|
|car|1913 / 114|0.492095|0.630605|0.254479|
|truck|95 / 4|0.332107|0.822778|0.122826|
|bus|33 / 1|0.399041|0.849517|1.591610|
|pedestrian|1067 / 67|0.484889|0.714331|0.623558|
|motorcycle|214 / 10|0.384522|0.777178|0.070572|
|bicycle|36 / 7|0.097226|0.856144|1.636058|
|traffic_cone|35 / 8|0.577273|0.558460|官方N/A|
|trailer|0 / 0|官方0|官方默认1|官方默认1|
|construction_vehicle|0 / 0|官方0|官方默认1|官方默认1|
|barrier|0 / 0|官方0|官方默认1|官方N/A|

无GT三类仍参与官方十类均值；没有删除它们后将有支持类别均值冒称官方mAP。无GT的AP=0和部分TP error=1是devkit原行为，不是实测到1m误差；这些类别质量不可评价。cone朝向/速度/属性和barrier速度/属性按官方为NaN，原文件不改成0。bus仅一个instance、truck四个；不能把33/95条相关记录当独立支持量。

原项目AP只采用2m、不同GT范围、有GT类别平均及另一积分约定。官方mAP汇总四距离、固定十类及min_precision/min_recall处理，不应要求两者相等。官方mATE是其匹配/召回积分下的平移误差，不是旧项目所有成功匹配XY均值。官方mATE也不检查竖直中心，mASE不替代中心Z检查；旧M8-J8竖直中心、框高及尾部误差报告保留有效。

两份原始官方聚合文件：[第一次metrics_summary](m8j9_sources/retry1/eval1/metrics_summary.json)、[第一次metrics_details](m8j9_sources/retry1/eval1/metrics_details.json)、[第二次metrics_summary](m8j9_sources/resume/eval2/metrics_summary.json)、[第二次metrics_details](m8j9_sources/resume/eval2/metrics_details.json)。这些是官方生成的聚合曲线/指标，不含逐对象预测。

## 失败保留、复现与结束

1. 首次export因raw sample缺少runtime data键失败，未导出或评测。保留原目录及非零退出记录；补官方keyframe反向索引并增加测试后在独立retry1目录重试。
2. retry1导出通过，eval1已成功写出官方指标；随后附加过滤诊断对NumPy ndarray做JSON比较失败，进程非零退出。没有回写为Passed。只修复诊断规范化，新进程完成过滤审计，再执行eval2；原官方指标、预测JSON均保留。
3. 官方指标实际计算恰好两次；额外audit仅运行官方过滤，不再计算指标。summary仅eval_time有差异（3.6275s与3.6923s），details逐字节一致。评分耗时不代表模型推理延迟。

最终19项有限测试通过：原导出器继承、token完整性、无GT读取、不对称尺寸/底面中心/速度旋转、真实空预测、缺速度拒绝、原属性规则、raw表关联、诊断NaN/NumPy处理、重复指标比较。原导出器/devkit及历史资产SHA保持不变；所有自有进程均退出。CPU禁网执行、无GPU、没有载荷下载或CARLA，没有对外提交成绩。

详细命令和失败后精确续跑方式见[实现说明](../docs/M8J9_IMPLEMENTATION.md)，执行证据见[重复性](m8j9_reproducibility.json)及[验收](m8j9_acceptance.json)。原始图像、逐对象预测/标签、权重未上传，main未合并。

本轮以**接口对齐通过、受限缓存基线已建立**结束。FP32训练/恢复与八帧学习结论保持原状，B迁移改善仍不受支持。未因此批准新的训练、扩大数据、视觉控制或任何安全门禁；下一研发里程碑由后续任务另行确定。
