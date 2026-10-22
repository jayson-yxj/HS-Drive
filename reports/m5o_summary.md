# M5-O — Perception Dataset & Localization Benchmark Reset

总体 **Partial**。工程接口：Passed；固定训练：Completed；视觉定位增益：NoGo；Offline Pilot：NotApproved。

基线 `2afe740`；独立分支 `feat/m5o-perception-benchmark-reset`；实验源码见逐报告 SHA 与最终 `m5o_artifact_manifest.json`。当前报告生成源码提交 `83a3a5da4093da3229995a8c1f066fdf9efe4db6`。不合并 main，不改变 Mobile、ResNet Depth、48 LID、HeightSplat 或 Oracle 控制；禁止视觉输出控制车辆。

## 实际数据与可靠支持

先完成失败径向布局试采，再改为采集前道路几何放置；旧数据和失败日志保留。第一组径向试采37次目标放置中23次无道路落点支持，可靠唯一候选仅6/7；道路几何版38/38有落点支持，候选30/31。此修订只使用Train通路审计。

Town02容量不足只分配2组，未缩小隔离距离；追加Town04不同road_id且距Train Ego≥150m的4个Calibration位置。Train20、Val6、Calibration6、Sealed6（实际全部完成）。相同布局家族、相机重叠及同位置目标相关性明确保留；位置/道路片区是独立分析单位，Actor-view不是独立场景。

|Split|seed|可靠唯一候选|独立位置|Vehicle|Person|Cone|
|---|---:|---:|---:|---:|---:|---:|
|train|101|642|20|511|63|68|
|train|102|658|20|532|59|67|
|val|101|198|6|156|21|21|
|val|102|202|6|160|20|22|
|calibration|101|199|6|157|25|17|
|calibration|102|205|6|171|21|13|
|sealed|101|269|6|226|23|20|
|sealed|102|279|6|227|27|25|

上述数量是预声明研究支持，不等于定位或安全合格。监督是CARLA渲染实例与OBB几何关联代理；Center是八顶点均值，不是Mesh质心或接触Gold。可见分母仅包括可靠实例关联且≥3原生像素的Actor-view，无法关联参考条目另计，不能宣称覆盖所有物理目标。冻结Encoder见过部分历史相邻道路，详见lineage audit。

## 1. 候选阶段造成多少失败？

历史M5-N回归 seed101：75个可见Actor-view中，候选漏检20，匹配后Direct误差>5m仍有14。
历史M5-N回归 seed102：75个可见Actor-view中，候选漏检21，匹配后Direct误差>5m仍有17。

历史未匹配候选172/142个不能全部解释为物理FP。Oracle实例区域将检测分母中的75个目标全部提供给冻结Head，但同目标定位尾部仍明显；这是输入分布变化诊断，不是重新训练后的检测上界。详见 `m5o_oracle_candidate_diagnostic.json` 和每组完整候选图。

新Sealed seed101：总候选4270，可见Actor-view 363，匹配307，漏检56，未匹配未知3963，分裂Actor-view 27，合并候选6。
新Sealed seed102：总候选3286，可见Actor-view 363，匹配311，漏检52，未匹配未知2975，分裂Actor-view 21，合并候选6。

封存失败分解（互不重叠）：seed101=漏检56 + 关联歧义30 + 唯一关联但Direct>5m 98 + 唯一关联且≤5m 179 =363；seed102=52+24+94+193=363。因此候选/关联造成约23.7%/20.9%的目标不能进入可靠定位，定位器本身还产生大量严重错误。未匹配候选3963/2975不在这些已知目标分母中，必须另行保留为未知负担。

## 2–3. 数据规模影响与Direct相对Geometry Prior

使用旧DirectCenterHead，Direct600维、Prior24维；两种子、25/50/100%嵌套位置组、每臂600步，相同候选/样本顺序/监督。只有Validation位置宏平均XY选择Checkpoint，无预算扩展。参数量不相等，因此是固定既有方法对比，不是参数量受控消融。

|seed|训练位置比例|方法|Validation覆盖|平均XY m|P95 m|>5m比例|
|---|---:|---|---:|---:|---:|---:|
|101|25%|geometry_prior|63.4%|3.13|9.36|16.1%|
|101|25%|direct|63.4%|4.95|14.88|38.4%|
|102|25%|geometry_prior|64.3%|3.08|10.23|18.2%|
|102|25%|direct|64.3%|4.31|13.88|27.6%|
|101|50%|geometry_prior|63.4%|2.52|7.17|11.8%|
|101|50%|direct|63.4%|4.00|11.49|27.5%|
|102|50%|geometry_prior|64.3%|2.77|8.04|15.4%|
|102|50%|direct|64.3%|4.05|12.28|26.2%|
|101|100%|geometry_prior|63.4%|2.59|9.59|12.3%|
|101|100%|direct|63.4%|3.61|9.98|23.2%|
|102|100%|geometry_prior|64.3%|2.52|7.94|11.2%|
|102|100%|direct|64.3%|3.66|12.47|21.0%|

自然覆盖与相同目标对比不可混称；`m5o_paired_method_analysis.json` 给出同目标、逐位置差异和整位置bootstrap。少量位置、共享地图和布局相关性使这些区间仅具描述意义。数量增加后是否改善均值、P95与严重错误必须分别判断。

从5个训练位置扩到20个，Direct平均XY误差由4.95→3.61m、4.31→3.66m；P95由14.88→9.98m、13.88→12.47m。第二种子10→20位置的P95略升，尾部改善不稳定。两种子在相同可定位目标上均逊于Geometry Prior。封存同目标、位置宏平均差为Direct比Prior多1.45m/1.33m；描述性位置bootstrap区间均为正，不能当成跨地图总体显著性证明。

Direct训练Loss降至约0.001，Validation性能仍差；数据支持不足是原瓶颈之一，扩大后仍不能解释掉特征/回归映射的泛化问题。固定600步试验不能证明任何预算下都不可能改善，但不支持继续当前路线自动增加训练。

真实RTX4090训练共12臂，累计时间 43.3s，最大峰值PyTorch分配 20.5MiB（缓存特征上的小Head训练，不含完整Backbone；634帧训练特征缓存190.5s、峰值237.0MiB）。日志记录每任务Loss、梯度、Checkpoint SHA与完整采样顺序；原视觉/Depth骨干冻结。

## 4. 独立风险校准

风险模型只用预测距离、框大小/区域、前景概率和边缘截断代理。Calibration与Validation位置分离；留一Calibration位置得到经验风险，阈值仅由Calibration选取，不能使用真实误差作推理字段。经验90%半径不是分布无关保证。无合格阈值时1.1拒绝全部准入，仍保留原始位置供研究。

calibration seed101：风险/实际误差Spearman=0.420，经验90%区间实际覆盖=90.0%；按风险排序事后诊断中，严重误差≤5%的最大可见目标覆盖=15.3%。Test曲线不用于重新选阈值。
calibration seed102：风险/实际误差Spearman=0.342，经验90%区间实际覆盖=90.0%；按风险排序事后诊断中，严重误差≤5%的最大可见目标覆盖=9.8%。Test曲线不用于重新选阈值。
sealed seed101：风险/实际误差Spearman=0.452，经验90%区间实际覆盖=90.6%；按风险排序事后诊断中，严重误差≤5%的最大可见目标覆盖=17.9%。Test曲线不用于重新选阈值。
sealed seed102：风险/实际误差Spearman=0.376，经验90%区间实际覆盖=95.5%；按风险排序事后诊断中，严重误差≤5%的最大可见目标覆盖=22.6%。Test曲线不用于重新选阈值。

## 5. 类别、相机和距离限制

可靠支持及逐类/相机/距离/位置/遮挡代理/坡道误差在 `m5o_dataset_support.json` 和 `m5o_eval_*.json`。低于20候选或3位置的类别/相机不作独立可靠结论。训练距离以0–40m为主，不具备40–80m泛化证据；不能外推远距离能力。

|seed|方法|Sealed覆盖|平均XY m|Median m|P90 m|P95 m|Max m|>5m比例|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|101|plane|41.6%|2.52|1.70|4.58|7.79|17.69|9.3%|
|101|dense_expected|76.3%|10.32|9.80|18.15|20.98|30.71|75.1%|
|101|dense_mode|76.3%|8.86|6.18|19.50|24.24|40.27|56.0%|
|101|geometry_prior|76.3%|2.84|2.07|6.43|8.69|14.72|14.4%|
|101|direct|76.3%|4.37|3.03|9.82|12.33|26.47|35.4%|
|101|direct_selected|0.0%|—|—|—|—|—|—|
|102|plane|46.0%|2.88|1.83|6.97|8.75|15.87|16.2%|
|102|dense_expected|79.1%|9.58|8.73|18.05|20.41|28.69|69.3%|
|102|dense_mode|79.1%|8.76|6.28|19.37|24.24|40.27|55.1%|
|102|geometry_prior|79.1%|2.76|1.82|6.02|8.28|14.07|15.0%|
|102|direct|79.1%|4.09|3.07|8.82|10.66|20.82|32.8%|
|102|direct_selected|0.0%|—|—|—|—|—|—|

封存分层Direct P95：Vehicle 12.47/10.71m，Person 6.99/7.37m，Cone 9.11/9.09m；0–10m 3.51/5.22m、10–20m 7.55/7.40m、20–40m 12.83/12.41m。没有类别证明可可靠定位。封存右前摄仅7/7、右后摄3/4个可靠候选且各只有1位置，不能解释其偶然较低误差为可靠。其他四摄P95大多仍8.6–12.4m。Calibration锥桶仅17/13，类别风险校准支持不足。

## 6–7. 架构决策与下一阶段

现有Direct Head路线：**Stop_current_route_no_automatic_head_or_loss_expansion**。限定Offline Object-Center Pilot：**NotApproved**。

本轮已达到预先计划的总体研究支持目标，但固定Direct路线仍未通过尾部、严重错误、独立Calibration与同目标基线门禁，因此停止当前冻结Mobile+DirectCenter Head的自动扩展，不追加层数、Loss或训练步数。总体支持达标不等于类别、全六摄或40–80m支持充分。没有在本轮验证替代3D表示方案，不能直接将替代架构标为合格；可复用此Benchmark进行后续预声明对照。

## 工程状态与复现

测试：324 passed, 6 skipped in 18.95s（交付复测；原18.20s日志也保留）；SDK日志 `runs/m5o_logs/sdk_tests_final.log`。工程接口、研究学习、泛化与安全状态独立；M5-M Runtime Partial继承，无新Runtime矩阵，无视觉控制。

Oracle状态、BEV Evidence、目标候选的Shadow合同和源时间戳已整理，Unknown不能变成Free；现有控制代码未改。主项目Oracle Planner/仿真工程可独立推进，其成绩不代表RGB闭环。

命令、参数与限制见 `docs/M5O_IMPLEMENTATION.md`。所有原始日志位于 `runs/m5o_logs`，数据位于 `/root/gpufree-data/hs-drive-data/{raw,processed}/m5o*`，模型位于 `runs/m5o_training`；源码/资产SHA见各报告和最终manifest。Git推送包含源码/协议/报告/图，未自动上传大数据与权重。

诊断图：`m5o_figures/m5o_data_scaling.png`、`m5o_training.png`、`m5o_calibration_risk_coverage.png`、`m5o_sealed_risk_coverage.png`及全部位置六摄候选拼图。
