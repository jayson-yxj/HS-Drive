# M8-J6：独立三步 FP32 Smoke 与跨进程恢复通过

实际执行日期：2026-10-27。基础提交 `2d75f38cd64804af4ea681bb547712ac2be6cfef`；独立分支 `feat/m8j6-fp32-resume-smoke-retry`。
运行源码提交 `20c9906356287b0e78c97c364057889c9dcd7449`，run_id `m8j6_retry_20261027_124102`。后续交付提交只增加汇总和证据，不替换实际执行源码。

**本次 Smoke Passed。** 新运行从原始权重与空 AdamW 状态开始，在首进程执行两步，新进程完整恢复后执行第三步；第三个进程仅验证最终检查点。新运行恰好三次 optimizer.step，未执行第四步或过拟合。旧失败运行保持独立：实际更新1次，仍为 FailedStopped，历史合计4次。

## 固定配置和原失败保留

沿用已固定八帧及数据合同，三步顺序为 index **0 → 4 → 1**（scene-0061、scene-0655、scene-0061）；token 见预检及每步日志。六摄512×1408、十类、900 Query、六层、batch1，RGB/标定输入与GT监督仍分开。未扩充数据或调整类别、尺寸、Loss。

原始 checkpoint SHA256 `16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad`；加载的冻结 safe tensor archive SHA256 `cf33dccb1d549c425a89cf346d6f05401c49f6cccbd5a579cb7d8937b5a6d591`。初始tensor-state hash与原基线相同，优化器初始state为空，未复用失败运行动量。

RTX4090，PyTorch2.0.0+cu118、MMCV1.7.2、MMDetection2.24.1、MMDetection3D0.17.1。FP32；autocast、TF32关闭，GradScaler未使用，未更改AMP/DCNv2历史结论。seed875606，cuDNN deterministic=true/benchmark=false，torch deterministic_algorithms=false，CUBLAS `:4096:8`。Backbone/Neck eval且无反向，Head train，原Head dropout保留；AdamW lr1e-5、betas(0.9,0.999)、weight_decay=0、clip35。实际未使用scheduler，checkpoint显式保存None状态。

旧运行没有成功保存实验checkpoint或首步详细参数差分，不能补造；保留原异常、日志、已有参数清单与证据缺口。运行后复核旧发布清单55文件、12项原模型/源码、旧私有参数清单、16个固定输入/监督包及manifest全部未变。详见 [保留核验](m8j6_retry_preservation.json)。

## 三步实际结果

分类/回归为全部六层Loss之和。梯度列为全局范数裁剪前→裁剪后；没有依靠weight decay产生参数变化。

| Step / 进程 | 帧 | 分类Loss | 回归Loss | 总Loss | 梯度范数 | 有梯度且改变的参数对象 |
|---|---:|---:|---:|---:|---:|---:|
|1 / first|0|1.479559|3.622027|5.101586|101.032425 → 34.999997|141|
|2 / first|4|1.421977|2.766336|4.188314|49.770954 → 35.000001|141|
|3 / resume|1|1.662471|3.658460|5.320931|41.002178 → 35.000001|141|

三步Loss/梯度有限，分类、回归和Transformer参数梯度均非零。每步均有分类与回归参数实际改变，Adam一阶矩非零，141个优化器state的step分别为1、2、3。35附近的浮点舍入如实记录，不更改裁剪阈值。三步Loss不要求单调，也不支持收敛结论。

每步178个冻结参数对象和159个buffer均精确不变（337项冻结state），包括所有BN统计量及计数。141个训练对象对应221个state别名，没有把额外80个别名误判为冻结。共319个Parameter对象，训练11,550,624个标量、冻结24,254,781个标量；训练/冻结互斥，优化器无重复。完整“规范名称—全部别名—训练资格—group/slot”见 [参数映射](m8j6_retry_sources/m8j6_retry_20261027_124102/first/parameter_mapping.json)。各进程映射一致，跨进程不比较对象ID数值。

| 帧 × Decoder Layer | P | I | N | 分类有效权重 | 分类/回归归一化因子 |
|---|---:|---:|---:|---:|---:|
|0 × 每层0–5|50|0|850|900|50 / 50|
|4 × 每层0–5|28|0|872|900|28 / 28|
|1 × 每层0–5|51|0|849|900|51 / 51|

这是三帧各六层，不是三个Decoder层。18条完整frame×layer记录保存在 [逐步汇总](m8j6_retry_updates.json)，包含各层分类正/负贡献及直接梯度。正分类、负分类、正回归梯度均有效，负Query回归梯度为0。Negative依据已核对的官方检测任务匹配合同，不表示Occupancy Free，不移植Bench2Drive帧级Ignore。

峰值allocated **1,258,114,048 bytes（1.172GiB）**，reserved **1,776,287,744 bytes（1.654GiB）**。这是每步重置后的PyTorch CUDA allocator口径，冻结图像骨干、仅检测头反向；不含其他进程/驱动，不能当作整网训练显存。三步各约2.565/2.029/2.304秒，含读取、前后向、更新与证据检查，不是纯推理耗时。三个进程总墙钟约38.72秒（含重建、恢复和检查），全部exit0并已结束。

## 保存与恢复

| 检查点 | 保存PID → 恢复PID | 恢复后动作 | 检查结果 |
|---|---|---|---|
|step2|335257 → 336165|仅step3，下一样本index1|Passed|
|step3|336165 → 337012|只读验证，0次更新|Passed|

两次均逐项核对模型参数/buffer、优化器tensor与group、global_step、下一样本、共享别名、冻结策略、模块模式和Python/NumPy/torchCPU/CUDA随机状态。snapshot为独立clone；加载保持原模型共享关系。保存前固定eval与新进程相同eval的分类/框输出最大绝对误差及相对L2均为0，预声明容差atol1e-6/rtol1e-5未更改。eval探针保存恢复随机状态，不消耗后续训练随机数。step3开始的模型hash、优化器状态与step2结束精确衔接。

私有checkpoint各242,068,340 bytes：

- step2 SHA256：`a6bd65439d6863cbf70752f357ed2ac3b5d64dc5100e6dbaf5200082125e2042`
- step3 SHA256：`7eaa5d3420965dbd6c81756a7a5ca554fab68352eee830bfb638932b7931cdf9`

文件位于 `/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/m8j6_retry_20261027_124102/`，未提交模型或优化器tensor。完整证据见 [恢复验证](m8j6_retry_restore_validation.json) 与三个进程原始日志。通过范围为本次保存、跨进程恢复、继续一次更新和最终重载；没有执行连续三步对照，不宣称与所有不间断训练逐位等价。

## 测试、验收和下一步

原36项测试复跑通过（20项选帧/读取，16项官方合同/别名/隔离），新增14项通过，共**50项**。新增覆盖真实冻结参数/buffer故意改动会拒绝、共享别名一致性、矛盾策略拒绝、重复/遗漏/冻结优化器成员拒绝、独立快照、状态损坏、RNG恢复及官方模型冻结范围。原始35条依赖弃用警告保留，不是测试失败；中间12项测试记录也保留，不重复计数。复现命令见 [实施说明](../docs/M8J6_RETRY_IMPLEMENTATION.md)。汇总脚本只读检查证据，不加载模型、不产生额外step。

- 本次三步及恢复：**SmokePassed**；见 [验收](m8j6_retry_acceptance.json)。
- FP32：仅本配置/样本/有限更新范围通过；AMP未复测，历史未通过状态保留。
- 小集合过拟合：**建议进入下一项独立、固定预算实验；本轮NotRun**。建议继续这八帧、官方十类、512×1408、bs1、FP32、同冻结策略；启动前固定步数上限、样本循环、评估频率与停止条件，不调新模型或扩数据。可以从本次step3作为明确记录的初始化，或原基线独立重启，须预先选定，不混合结果。
- 两场景属nuScenes full train，与预训练数据有已知重叠；本次不评估泛化。未覆盖类别、不间断训练对照、AMP、正式Occupancy和视觉驾驶安全没有被本次通过替代。

未启动CARLA、没有第4步或长训练，默认模型/控制系统和历史证据保持不变；只提交代码、统计、哈希及日志，不上传私有图像、逐对象GT或权重，不合并main。
