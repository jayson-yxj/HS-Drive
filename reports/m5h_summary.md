# M5-H — Efficient Visual Distillation & Near-Preserving Depth

详细七项结论、低FPR锁定工作点和因果限制见[结果解释补充](m5h_findings.md)。

**总体 Partial。** 完成实际RTX4090匹配蒸馏/深度研究、CARLA六位置封存测试、空间Shadow和独立六摄计时。未开启大型Occupancy、Temporal、SparseQuery或视觉闭环。

基线`bf0c38d`；源码冻结提交`f0c636c`；独立分支`feat/m5h-distill-near-depth`，不合并main。复现和数学定义见[实现文档](../docs/M5H_IMPLEMENTATION.md)。原始模型/报告/数据不覆盖；完整SHA在`m5h_artifact_manifest.json`。

## 实际执行与实验控制

Train从192帧/12位置增加至208帧/14位置，Val96帧/6位置不变；新增两个训练位置补相机盲区。测试72六摄帧来自Town01/02/03/04/05/10HD六位置，每处12配置，另有对应ego-removed参考。不是72个独立场景，资产/布局模板仍重复。新测试采集前锁定全部模型/阈值，之后无重新训练或模型选择。

两个seed×GT/Distill各2400步；Depth Replay/Replay+KL各1200步。Student主干/48LID/HeightSplat/默认控制器不变。Teacher缓存304训练验证帧和72测试帧的RGB-only预测，非每step推理。Teacher支持车辆/行人，cone仅GT监督。官方权重保留本地，不分发。

实际Student每组约39.6–41.5秒，峰值allocated约6.04GiB；Depth每组约28.5–39.6秒，峰值约2.04GiB。含缓存驻留与周期评测，期间部分独立作业并发，不是从原图完整六摄训练吞吐。参数与逐任务Loss、梯度、实际选择step在原始训练JSON。

完整训练覆盖1248/1248视角；右后仅2个正视角，不能声称外观充分。seed101最终选1200步，尚48视角未见；seed102选1600步。最终曝光与所选Checkpoint分开。相较M5G同时改变了样本和采样，不能把差异单独归因于覆盖；教师损失A/B是严格配对因素。

## 蒸馏与低误报

|模型|seed|Test P|Test R|AP|整体背景FPR|每百万有效像素FP|Removed局部FPR|Actor-view漏检|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|gt|101|29.12%|28.96%|18.39%|0.16%|1561.5|10.63%|49.21%|
|gt|102|40.83%|37.71%|23.32%|0.12%|1210.4|8.56%|33.33%|
|distill|101|29.26%|26.04%|18.02%|0.14%|1394.6|8.86%|49.21%|
|distill|102|46.00%|48.65%|34.41%|0.13%|1265.0|2.31%|31.75%|
|M5G_fpn_bce|101|30.86%|39.74%|24.97%|0.20%|1973.0|7.38%|33.33%|
|M5G_fpn_bce|102|36.92%|44.24%|34.33%|0.17%|1674.8|3.79%|30.16%|
|COCO_teacher|0|98.65%|61.62%|91.75%|0.00186%|18.6|0.00%|7.02%|

Teacher主行仅vehicle/person，Student主行含cone，严格同域对比请读`m5h_pixel_quality.json`的`supported_metrics`。Actor指标为可见>=3像素Actor-view，误检组件不等于物理Actor级Precision。所有主阈值由Val冻结；测试PR曲线只描述，不能重新挑阈值。整体背景FPR分母是全部可靠非目标像素，不是Empty专用分母。每百万有效像素FP分母还包含正像素。

蒸馏Val macro AP分别增加2.34/5.51个百分点，但联合门禁violation两seed均变差；故不启动C困难负样本或Backbone升级。保守正区soft BCE已与硬GT高度重叠，缺少背景/边界/特征知识迁移；这限制了本次方法，不能证明所有蒸馏无效。教师在Train/Val表现明显有域差，具体漏检、小目标、天气与camera审计在`m5h_teacher_quality.json`。低置信/冲突不作为强负样本。

## 近场保护与空间概率

新Test原Head近场Expected MAE **1.1024m**。以下均为Val选中非零step，未用Test选择。

|模型|seed|Test近场MAE m|相对原Head|1.05门禁|
|---|---:|---:|---:|---|
|replay|101|0.8170|0.741|Passed|
|replay_kl|101|1.0469|0.950|Passed|
|replay|102|0.8433|0.765|Passed|
|replay_kl|102|1.0397|0.943|Passed|

|模型|seed|全局MAE/Bias m|10–20m目标MAE / n|20–35m目标MAE / n|20–35m同层1m质量 / BEV n|
|---|---:|---|---|---|---|
|original|0|3.046 / -0.698|6.376886123324198 / 630|8.781553387641907 / 16|2.26% / 12|
|replay|101|2.910 / -1.269|3.7596860017095293 / 630|4.940238833427429 / 16|2.85% / 12|
|replay_kl|101|3.024 / -0.917|4.945075857071649 / 630|6.737109184265137 / 16|2.29% / 12|
|replay|102|2.911 / -1.198|4.1154169035336325 / 630|4.74109947681427 / 16|2.74% / 12|
|replay_kl|102|3.021 / -0.923|4.819751443938603 / 630|6.570736765861511 / 16|2.34% / 12|

MAE为Softmax期望光轴深度，Mode另报。分层0.5/1/2m概率、GTBin、范围外质量、期望三维误差与严格Mask Oracle2D投影见`m5h_depth_spatial_quality.json`及逐帧原报告。原M4 S16 Val近场也通过，跨场景单独检查，不能只看池化均值。

与M5G相比同时降低LR、取消距离/Actor等权并加入Replay，因此不能单独证明Replay贡献；B/C只隔离KL。近/远GT梯度余弦及KL梯度冲突已实际记录，初始化KL梯度为零。固定KL10保留原分布也约束可改善的近场误差，不是强制旧分布等于GT。条件D有资格但本轮明确延期，先验证B/C泛化，无额外空间损失训练。

## 联合BEV空间结果

|seed|组合|同层1m Recall|可见表面Precision|确认背景FPR|
|---:|---|---:|---:|---:|
|101|distill_olddepth|6.11%|1.00%|1.15%|
|101|newpixel_newdepth|11.64%|1.64%|1.10%|
|101|newpixel_olddepth|6.07%|1.09%|1.01%|
|101|oldpixel_newdepth|15.92%|2.04%|1.60%|
|101|oldpixel_olddepth|10.14%|1.14%|1.56%|
|101|oracle2d_newdepth|31.63%|34.60%|0.12%|
|101|oracle2d_oracledepth|69.58%|100.00%|0.00%|
|101|teacher2d_oracledepth|67.66%|30.04%|0.36%|
|102|distill_olddepth|16.67%|2.46%|1.14%|
|102|newpixel_newdepth|14.84%|1.98%|1.51%|
|102|newpixel_olddepth|10.22%|1.23%|1.36%|
|102|oldpixel_newdepth|18.87%|4.18%|1.03%|
|102|oldpixel_olddepth|16.50%|3.45%|0.94%|
|102|oracle2d_newdepth|30.96%|34.50%|0.12%|
|102|oracle2d_oracledepth|69.58%|100.00%|0.00%|
|102|teacher2d_oracledepth|67.66%|30.04%|0.36%|

新Pixel是Val选出的GT组，Distill另列；旧Pixel是M5G FPN1200最佳Checkpoint。Oracle2D/Depth是同渲染器诊断，不是独立Mesh真值。正式学生输出无GTMask；Unknown不算安全负例。Teacher+OracleDepth对cone无支持，同域空间结果单列supported。粗高度层中心3D误差仅空间格定位代理，不是精确Actor三维误差。

## 推理、ROI与安全

|模型|六摄P50 ms|P95 ms|峰值allocated MiB|
|---|---:|---:|---:|
|gt_101|7.49|8.54|184.2|
|gt_102|7.52|7.80|183.3|
|distill_101|7.49|7.80|183.3|
|distill_102|7.49|7.97|183.3|
|M5G_fpn_bce_101|7.65|7.75|225.3|
|M5G_fpn_bce_102|7.62|7.77|224.3|
|joint_101|18.34|18.61|244.7|
|joint_102|18.57|18.68|244.7|
|COCO_MaskRCNN_sequential6|818.97|895.86|740.0|

独立GPU计时关闭CARLA和其他GPU作业；不含磁盘/渲染/tick/控制。Student结构不变，蒸馏无额外部署教师。不能由此宣称20Hz控制严格实时。ROI选取仅使用Student输出，OracleDepth离线测射线覆盖，未实现S4Depth网络或替换Splat。

|seed|采样|平均射线数|同层1m Recall|目标射线遗漏|
|---:|---|---:|---:|---:|
|101|full_P8|23040|69.58%|0.00%|
|101|full_S4|92160|85.29%|0.00%|
|101|student_ROI_S4|716|22.82%|71.82%|
|102|full_P8|23040|69.58%|0.00%|
|102|full_S4|92160|85.29%|0.00%|
|102|student_ROI_S4|354|29.93%|63.14%|

## 验收、失败记录和下一步

全套268 passed、5 skipped；CARLA环境相关23 passed（与全套有重叠）；实际CUDA隔离验证通过。合成单测只验证Mask/梯度/计数合同，不能替代CARLA质量实测。采集/训练/冻结/评测完整日志在`reports/m5h_logs`。

保留初版地图名/世界坐标混用造成的错误审计，最终隔离依据`m5h_split_audit_final.json`的Pixel距离和`m5h_replay_coordinate_audit.json`更正的M4距离；最小距当前Train/Val/M4 Replay为49.57m。教师名义resize记录独立更正；首轮Train自选阈值指标不用于模型选择，采用`teacher_train_locked`替代一致工作点。某些报告在结束时对源文件取SHA，期间仅增加分组输出字段；实际执行版本归档`runs/m5h_source_archive`，冻结后科学代码不变。

**七项结论：**

1. 本次正区软分数蒸馏尚未提供稳定联合质量增益：seed101 P29.12→29.26%、R28.96→26.04%，seed102 P40.83→46.00%、R37.71→48.65%。结构和推理成本保持不变。Teacher更强不等于这种KD目标足够。
2. 仅seed102同时改善P/R/Removed（局部FPR8.56→2.31%）；seed101局部FPR10.63→8.86%但Recall下降。两个seed不稳定，不能宣称新位置可靠。
3. 完整预算曝光确实提升到100%，但最佳早期Checkpoint仍可能未全见；数据增补与采样未单独消融。Student未更换视觉预训练权重，教师loss才是A/B单变量。
4. 本轮降低LR、均匀射线CE+原分布Replay建立了通过Val的近场保护候选；新Test严格结果见近场表，默认Head仍不覆盖。
5. BEV收益需同时看空间Recall/Precision/背景；近场MAE改善不能自动视为Actor空间定位通过，联合门禁继续Partial。
6. 高分辨率采样仍是独立瓶颈，但二维识别也未可靠，不能宣称瓶颈已经完全转移到投影。预测ROI遗漏会限制更密采样收益。
7. 继续受控视觉特征/可靠负例覆盖与Depth空间概率研究，暂不扩大正式BEV/Occupancy训练，也不自动启动Temporal或Sparse Query。

工程接口、研究执行、泛化质量、运行时与纯视觉安全分别验收；详见`m5h_acceptance.json`。
