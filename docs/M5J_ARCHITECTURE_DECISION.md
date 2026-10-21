# M5-J Architecture Decision

{
  "engineering": "Passed",
  "mobile_migration": "Go_OptInResearchOnly",
  "visual_absolute_generalization": "Partial",
  "aggregation": "NoGo_PredeclaredGainNotMet",
  "depth_upgrade": "NoGo_NotRequalified",
  "BEV_spatial_quality": "Partial",
  "complete_offline_gpu_budget": "Passed",
  "runtime": "PartialInherited_NoRuntimeChanges",
  "oracle_driving_safety": "NotRevalidated",
  "pure_visual_driving": "NotValidated",
  "occupancy_pilot": "NoGo_ThisRound"
}

Go：仅在相对视觉增益及接口通过时，保留Mobile双分支为显式离线研究Candidate。No-go：不替换默认Encoder，不升级聚合，不升级Replay，不启动Occupancy联合Pilot。

A–C的阈值匹配收益不足，停止继续增加公式/校准复杂度。下一步将空间研究集中到可靠Actor支持、Depth概率落点与真实采样密度；S4Oracle收益不能自动外推到学习S4网络。未来有限训练必须另立协议并使用新封存组。

来源分解见报告：二维错误、GT目标错位、未知射线与累积同时存在。当前干预只能评估固定模型条件下的效果，不声称数学识别唯一因果。Runtime和驾驶安全保持独立开放。


## 具体执行边界

Mobile接口Go并非默认系统Go：只批准已实现的显式离线候选，原Depth/Context继续固定。新增Test相对同预算M5I GT基线也显著改善，见`m5j_matched_resnet_regression.json`。但4位置、6个小目标、零中等像素目标不足以关闭跨域泛化。

聚合No-go是预声明研究判据未满足，不是“数学上不能改善”。门控和正则化有有限收益，Oracle分解显示真实Depth替换收益更大，继续增加校准复杂度没有本轮证据。保留原raw主对照，候选代码只供诊断。

下一阶段先补可靠关联Actor中远场与目标面积覆盖，再对采样密度和深度概率空间质量建立受控实验；仍不擅自改变48LID、默认Splat或启动大规模Occupancy。CPU输入路径本轮为离线校验读取，不能自动成为在线控制链。
