# M5-J — Mobile Visual Migration & BEV Evidence Calibration

**结论：研究有明确Go/No-go，整体质量仍Partial。** Mobile为显式独立双RGB分支；Depth与Context仍使用原ResNet，未交叉喂入Mobile特征。旧默认、48LID、HeightSplat、四层BEV、Controller/Physics全部保留。

## 决策

- Mobile独立研究迁移：Go_OptInResearchOnly；完整双分支GPU预算：Passed。
- 新聚合升级：NoGo_PredeclaredGainNotMet。固定公式A/B/C未通过两seed预声明+5pp Precision、最多2pp同层1m Recall下降与最多2pp Actor检出下降。D条件未满足，未训练校准层；无视觉或Depth训练。
- 正式BEV质量、Occupancy Pilot、Runtime及纯视觉安全门禁没有自动关闭。本轮不启动联合训练、Temporal、Sparse Query或视觉控制。

## 新位置二维对照

|seed|Mobile P/R|旧候选P/R|P/R同增位置|
|---:|---|---|---:|
|101|81.10%/86.08%|35.78%/24.73%|3/4|
|102|78.94%/88.13%|52.48%/30.15%|4/4|

仅4个独立位置，车型/配置有共享，不能按像素或重复帧数量宣称广泛泛化；历史Test仅诊断未训练。小目标支持见逐Actor报告。

## BEV工作点（Val选择，Test不调阈值）

|seed|方法|阈值|精确表面P|同层1m R|Actor检出|背景FPR|
|---:|---|---:|---:|---:|---:|---:|
|101|raw|0.2|17.40%|42.88%|62.96%|0.30%|
|101|gate|0.2|20.79%|41.81%|62.96%|0.23%|
|101|regularized|0.025|20.60%|45.75%|64.81%|0.27%|
|102|raw|0.2|12.50%|44.00%|64.81%|0.46%|
|102|gate|0.15|19.09%|45.44%|64.81%|0.30%|
|102|regularized|0.025|15.64%|46.00%|66.67%|0.38%|

固定0.2、Val锁定、Recall匹配和Precision匹配分别保存。Test曲线只是诊断，不按Test重选。正则化保证score≤raw且分母total+1，不会放大小质量；固定0.2的FPR下降必须同时看Recall损失。

## 高分格来源

|seed|区域|高分格|非目标源质量A|目标错位质量B|未知源E|低q质量比例|
|---:|---|---:|---:|---:|---:|---:|
|101|target|325|6.35%|58.36%|7.25%|0.29%|
|101|background|1543|30.43%|51.18%|18.39%|2.40%|
|101|unknown|13048|6.85%|28.01%|63.27%|8.48%|
|102|target|332|6.47%|58.36%|7.05%|0.36%|
|102|background|2324|38.47%|41.81%|19.72%|6.09%|
|102|unknown|9499|16.03%|40.24%|40.97%|11.73%|

A/B/E为射线源贡献描述，C概率扩散与D累积可重叠，不是独立因果百分比。来源GT不可用时为E；Unknown高分不叫物理FP。ray_bin_count与distinct camera/pixel射线数严格区分；不同射线仍相关。

## 采样与Depth

|Oracle采样|phase|同层1m R|
|---|---:|---:|
|S4|2|80.88%|
|S4|0|80.25%|
|S8|4|68.94%|
|S8|0|67.12%|
|S16|8|48.44%|
|S16|0|50.50%|

新Test严格20–35m关联Actor射线：61；详细类型/位置在m5j_depth_shadow.json。MAE、Bias、GTBin、正确层和同层1m概率分开评估，Replay仍只读，不因MAE单独改善升级。S4仅Oracle深度采样诊断，没有生产S4Depth网络；ROI结果分母包含漏掉的目标。

## 完整推理成本

|视觉分支|seed|P50 ms|P95 ms|峰值MiB|
|---|---:|---:|---:|---:|
|mobile_gt|101|24.68|25.52|227.7|
|mobile_gt|102|24.45|25.05|228.3|
|legacy|101|24.23|25.30|260.1|
|legacy|102|24.26|25.42|260.1|

含输入检查、Pixel、原Depth、Lift和精确支持计数；CPU读图/解码/SHA、序列化、反序列化和H2D另测。不包含CARLA控制周期，不称20Hz在线闭环。

## 交付与限制

所有模型/配置/数据及源码SHA、原始日志、失败恢复记录保留。标签来源仍为CARLA可靠可见首表面，非独立Mesh。空间事件分数不是物理Occupancy或安全Free。

复现见[实施说明](../docs/M5J_IMPLEMENTATION.md)，决策见[ADR](../docs/M5J_ARCHITECTURE_DECISION.md)。[PR曲线](m5j_figures/m5j_pr.png)和[错位射线图](m5j_figures/m5j_trace_B_target_depth_misprojected.png)展示实际结果。

## 七项最终回答

1. **Mobile独立迁移：Go，仅限显式离线研究Candidate。** 新Test Mobile P/R为81.10/86.08%、78.94/88.13%；原M5I同预算GT对照为40.79/45.23%、34.43/43.08%，补充只读回归确认相对收益。原M5G候选比较存在历史训练差异，不能单独解释为纯架构消融。默认Encoder不替换，绝对跨相机质量仍Partial。
2. **BEV错误来源不是单一累积。** 已确认背景高分格中，GT目标射线落到其真实端点以外的质量约51.18/41.81%，GT非目标源约30.43/38.47%，未知源约18.39/19.72%；低q<0.2质量仅2.40/6.09%。A包含低分非目标预测，不能把该份额全部叫高置信二维误检；B是粗格端点错位，不等于独立物理碰撞真值。概率扩散、重复bin和多射线累积可以叠加。
3. **匹配Recall后有小幅Precision收益，但聚合升级No-go。** Val匹配Recall的固定工作点在新Test上：gate101 P20.48/R46.25%，raw P17.40/R42.88%；gate102 P19.09/R45.44%，raw P12.50/R44.00%。正则化P20.60/15.64%、R45.75/46.00%。Test上实际Recall不必完全相等，因为匹配只在Val完成；没有使用Test插值重选阈值。两个seed未同时达到预声明5pp增益，不能借seed102单项通过改判。
4. **多位置有改善趋势，尚未稳定达到资格门槛。** 正则化在新Test四位置均提高Precision且Recall不降，但Val收益不足且Test多数增益不足5pp。gate101在Town01同层Recall下降2.3pp。没有逐Actor质心定位模型，当前只报告可见表面覆盖和Actor检出，不称精确对象定位。
5. **主要研究方向转向Depth空间落点和采样，但二维背景排除仍需保留。** matched Oracle干预下，Mobile+LearnedDepth raw P/R约17.96/39.50%、13.60/40.12%；Mobile+OracleDepth提高至30.52/69.88%、17.51/70.38%。Oracle2D+LearnedDepth原始Recall仍36.94%。S4 Oracle中心Recall80.88%，P8为68.94%，S16为48.44%；这是当前表面参考和采样相位下的诊断，不是严格信息论上界，也不能自动外推学习S4网络。
6. **有限BEV/Occupancy联合Pilot：本轮No-go。** 先补足可靠Actor与小目标支持、研究Depth概率几何与采样，保持有限且预声明的下一轮对照。本轮不继续叠加聚合校准模型，也不新增Depth微调。
7. **完整Runtime与纯视觉驾驶安全仍开放。** 完整离线双分支GPU P95约25.1–25.5ms、峰值约228MiB，但CPU读图/解码/SHA校验P95约76.6ms，序列化10.4ms、反序列化8.0ms、H2D3.6ms分别测量；不能相加这些分位数当端到端P95，更不能宣称20Hz闭环。本轮没有在线Shadow、没有修改Tick Owner、没有视觉控制。

新Test严格20–35m仅61条可关联Actor射线，全部为vehicle；person/cone此段无严格支持，泛化支持不足。Replay关联Actor远场MAE从10.11m降至7.82/7.58m，但同层1m质量从3.14%降至2.84/2.83%；近场也有MAE下降而空间质量下降。因此继续保持Replay不升级。

新Test小目标≤32像素仍仅6个Actor-view，33–128像素仍为0；采集预算没有达到小目标覆盖目标，这是一项明确失败，不能用“添加了锥桶”代替实际支持。Actor背景FP不能唯一归属某个GT Actor，因此逐Actor提供TP/FN/检出，背景FP在camera/scene/components报告。

原始raw事件阈值本轮按精确float 0.2计算；历史m5i部分统计使用floor(0.2×256)/256。公式数值等价已实测，但跨轮阈值边缘计数可有差异；本轮A/B/C全部同一精确阈值，不从历史数字拼接增益。PR直方图为256bin近似，最终工作点计数使用原始浮点分数。

完整测试最终记录在`m5j_logs/full_tests_final.log`；原始失败记录、旧源码快照和恢复方式保留。补充只读对照与绘图在冻结后增加，均不改变候选、参数、Mask、工作点和选择结果。
