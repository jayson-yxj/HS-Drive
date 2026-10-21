# M5-L — Sparse Geometry Feasibility & Object-Centric Localization

**总体Partial；不升级默认Depth/Splat，不启动Occupancy联合训练或视觉控制。**

## 实际工作与采样失败

先复用M5-K全部312帧，旧Test仅回归。Train/Val远场行人可靠内部像素335/326、固定P8命中0/0均复现。P4严格射线各9，第二类别覆盖门禁仍失败，未训练P4Depth。实际Mobile浅层S4+语义P8局部特征接口已GPU验证，不把插值当成新视觉信息。

## Vehicle-only残差

两个种子，各B/C600步；Val选择Checkpoint，不延长预算。旧M5-K两类别门禁保持原样。

|集合|seed|方法|20–35m Vehicle MAE|同层1m概率|BEV ΔP/ΔR1/ΔFPR|保护/BEV门禁|
|---|---:|---|---:|---:|---|---|
|val|101|no_mobile|2.497m|14.15%|+6.63%/+14.42%/+0.133%|False/False|
|val|101|mobile|2.551m|14.55%|+6.50%/+14.25%/+0.132%|False/False|
|val|102|no_mobile|2.806m|18.45%|+8.71%/+21.44%/+0.145%|False/False|
|val|102|mobile|2.800m|19.18%|+8.71%/+21.49%/+0.145%|False/False|
|regression|101|no_mobile|2.085m|15.54%|+2.36%/+33.24%/+0.194%|False/False|
|regression|101|mobile|2.134m|15.77%|+2.33%/+33.52%/+0.201%|False/False|
|regression|102|no_mobile|2.328m|20.78%|+2.62%/+40.90%/+0.158%|False/False|
|regression|102|mobile|2.323m|21.24%|+2.64%/+41.31%/+0.152%|False/False|
|test|101|no_mobile|2.024m|15.51%|+1.74%/+25.15%/+0.081%|False/True|
|test|101|mobile|2.050m|15.59%|+1.70%/+24.54%/+0.080%|False/True|
|test|102|no_mobile|2.502m|16.31%|+1.96%/+30.67%/+0.049%|False/True|
|test|102|mobile|2.512m|16.39%|+2.03%/+30.98%/+0.045%|False/True|

Mobile条件的独立增量见`m5l_vehicle_residual_ablation.json`；不能把B/C共同收益全归给Mobile。未验证Pedestrian/Cone泛化。

## 目标级2.5D

同一RGB候选及底部射线上比较，AABB XY中心只是参考代理；可见底部端点误差另报。不是体素Recall。

|集合|seed|有效/可见Actor-view|Plane XY中位|Dense Expected XY中位|Dense Mode XY中位|
|---|---:|---:|---:|---:|---:|
|val|101|32/70|1.890|2.624|5.586|
|val|102|35/70|2.057|3.355|5.403|
|regression|101|34/71|1.265|2.203|5.310|
|regression|102|37/71|1.912|2.759|6.124|
|test|101|15/36|1.302|2.796|3.498|
|test|102|19/36|1.725|2.855|3.499|

二值前景不能确认接地或遮挡；输出仅是平面条件假设，可信定位状态仍Unknown。坡道、边缘、合并、分裂和无匹配需单列，不能以条件子集成绩宣称整体通过。

## 自适应质量归因

新GT端点覆盖与已有概率质量跨0.2阈值同时统计，并有交集计数，不能相加当成互斥百分比。Val锁定同FPR/Recall工作点，增加同前景矩形预算对照。原公式、48bin和GT隔离未变。详见`m5l_adaptive_mass_diagnostics.json`。

## 在线Shadow

实际独立进程2Hz六摄RGB，无CARLA客户端、tick或控制调用。主实验唯一TickOwner运行baseline/shadow/recovery，原时间戳不改、250ms过期拒绝。

是否因tick长尾停止：**False**；发出30包，输出30，状态{'expired_output_discarded': 1, 'stale_before_compute': 1, 'accepted': 28}。

- baseline tick P95 70.78ms / Max 237.05ms；间隔P95 70.85ms。
- shadow tick P95 84.20ms / Max 88.51ms；间隔P95 84.57ms。
- recovery tick P95 61.48ms / Max 70.68ms；间隔P95 61.11ms。

静态传感器工程可运行不代表Oracle驾驶资源争用通过，更不是纯视觉安全认证。

## 七项回答

1. P8失败同时受细小轮廓、可靠内部过滤和固定相位影响；真实P4只改善取样机会，现有行人支持仍不足，未验证其Depth空间收益。
2. Vehicle-only残差存在限定学习信号，但背景/近场/场景门禁必须独立看；保留研究Checkpoint，不追加同结构预算。
3. Mobile前景条件只带来小幅概率增量，没有稳定额外BEV优势，不能宣称已解决目标Depth。
4. 额外射线既带来新Oracle表面格，也通过既有格质量累积跨阈值；报告交集与同FPR对照，不把两者做互斥因果拆分。
5. 2.5D在可匹配条件子集有位置研究信号；接地/平面/遮挡可信度仍未解决，不批准通用定位。
6. Shadow结果仅适用于本轮实际静态六摄并发试验，是否触发停止以上述日志为准，不能外推驾驶安全。
7. 下一阶段优先研究可拒绝预测的目标级几何与可靠高分辨率监督；保留Vehicle-only候选供对照，不直接进入正式SparseQuery、Temporal或Occupancy联合训练。

复现见[实施说明](../docs/M5L_IMPLEMENTATION.md)，预声明见[架构决策](../docs/M5L_ARCHITECTURE_DECISION.md)。所有失败和原始日志保留服务器，Git仅摘要/必要图片/SHA索引。

## 交付复核

工程接口通过、Vehicle-only训练完成、泛化质量Partial、纯视觉安全NotValidated分别成立，不能互相代替。

原生行人宽度中位约5像素。Train/Val最佳固定P8相位也仅11条，默认相位0条；P4各9条，说明不能仅靠更改相位或插值解决监督稀疏。

Val Adaptive新增Oracle目标端点格929/945，既有目标格跨阈值453/399，交集111/118；背景跨阈值1548/1549。计数按frame-cell累积，有重复/相关性；A/B非互斥。Val同FPR锁点Adaptive阈值0.5，Recall4.08/4.00%，低于Uniform@0.2的5.75/4.44%；因此暂停直接ROI优化升级。

Shadow 30包，28及时输出，1计算前过期、1输出过期；首次推理长尾约930ms，输出完整长尾约1065ms均保留并丢弃。完整P95约133ms；tickP95由70.8增至84.2ms（约19%），未达到预声明停止阈值，但本场景自身平均tick约61ms，不能称20Hz稳定，也不能声称完全没有资源影响。

逐位置近场/Bias检查发现46项未通过，不能用聚合MAE掩盖，见`m5l_scene_protection.json`。这些失败进一步阻止默认升级。

封存集严格支持：

|Stride|类别|20–35m射线|有支持位置|
|---:|---|---:|---:|
|8|vehicle|58|4|
|8|person|0|0|
|8|cone|0|0|
|4|vehicle|258|4|
|4|person|1|1|
|4|cone|0|0|

目标级方法始终输出contact-unverified条件假设；可信位置数量为0。`contact_supported`仅按GT可靠可见底部端点接近平面诊断，侵蚀后的底部并非轮胎接地点，不能将该诊断当作推理可用的接地证据。坡道采集采用Ego相对配置，地形变化可能导致Actor悬浮/穿地；不宣称精确物理接地真值。

下一阶段建议限定为：RGB可见接地点/接触可信度、可拒绝的目标几何、近场保护及已确认背景抑制。保留Dense分支作背景/表面上下文与车辆研究对照，尚无证据将其直接替换；不自动部署Object Query、SparseDrive或Temporal。

图：[行人采样](m5l_figures/m5l_train_pedestrian_support.png)、[目标XY](m5l_figures/m5l_object_xy.png)、[Adaptive同FPR](m5l_figures/m5l_adaptive_test.png)、[在线Shadow原始时序](m5l_figures/m5l_shadow_timing.png)。

Vehicle-only 的主要保护失败不是全近场MAE，而是全部20–35m表面的系统性Bias：Val无Mobile两个种子约−4.23/−4.25m，绝对Bias比冻结基线增约3.87/3.89m，远超0.5m增量上限。目标车辆子集改善与背景远场偏移必须同时保留。

最终全量测试298通过/5跳过，独立CARLA SDK14通过。历史4863个Git文件与41287个资产SHA无变化。
