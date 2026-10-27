# M8-J7 固定八帧 FP32 过拟合

基础f4e44f9，独立分支feat/m8j7-eight-frame-fp32-overfit。本轮只验证八帧训练集上的学习，不是完整nuScenes验证。协议和固定token、文件SHA、每帧GT数见 configs/m8j7_protocol.json，原335条GT一条不改。原始权重、空优化器初始化，不从Smoke第3步续训。

## 冻结配置与评价合同

原PETR十类、六摄512×1408、bs1、FP32。沿用实际Smoke的AdamW lr1e-5 / betas(.9,.999) / wd0 / clip35。常数学习率、无scheduler；400步仅50次八帧循环，不套长周期warmup。样本顺序0到7循环，seed875606。图像无随机增强，固定已冻结预处理。Head训练Dropout保持官方0.1；Backbone/Neck eval和no_grad，141个训练Parameter对象、178个冻结Parameter及159个buffer。共享别名和冻结检查复用M8-J6。

在任何更新前先完成八帧eval和step0检查点。eval点0/80/160/240/320/400；固定全model.eval、相同GT过滤、正常get_bboxes解码。官方NMSFree top300 query-class pairs、原decoder post_center_range、无内部score阈值。诊断区域固定strict |x|/|y|<51.2，与训练GT XY范围一致；这项预测过滤不读GT。额外保留原始decoded总量与区域外预测数量。

主score>=0.2，按同类别分数降序、一对一最近尚未匹配GT，XY距离strict <2m；同分数按固定预测索引。0.5/1m辅助指标、固定score网格PR、全部分数的101点插值AP另报。AP仅对有GT类平均；无GT类AP/Recall为空但误检计数保留。匹配只生成诊断，不挑选模型输出；重复预测仍为FP。主诊断分母固定335，逐帧/类/层记录。

定位误差分别记录XY和重心Z；尺寸记录三轴平均绝对误差及对齐尺寸IoU误差；yaw directed类按2π、barrier按π、cone不适用。误差仅来自成功匹配，始终同时报告TP/FP/FN，避免把少量匹配当全量质量。

Loss使用同一eval模式的八帧算术平均，同时保留各帧六层分类/回归及P/I/N。训练随机单步loss仅供稳定性检查，不替代固定评估。评估前保存随机状态/各模块模式，结束后恢复，并断言eval未改参数/buffer。没有“最佳checkpoint”选择：最终400步必须报告。

## 保护与保存

复用 tools.m8j6_retry_gpu.optimizer_step，不改旧代码。每步参数/独立快照、梯度与有限性、冻结337项、优化器映射和共享别名均检查，step返回立即落私有ledger。每步简明统计公开，完整参数变化和梯度明细私有并公开SHA。未使用AMP、TF32或GradScaler。

各评估点保存模型、优化器、step、下一样本、RNG、模式、别名、协议SHA及固定eval输出。最终新进程重建原结构、验证checkpoint哈希、精确加载参数/buffer/优化器与共享策略，再核对固定eval输出atol1e-6/rtol1e-5。最终验证进程不执行optimizer.step。不宣称所有连续训练逐位等价。

异常立即停止并保留failure checkpoint及最后有效评估。一次运行上限400步/2400秒，不自动重试或延长、不因loss短期上升而停。目录存在拒绝覆盖。所有GPU子进程禁网，不读原始点云/Radar或GT推理特征。私有图像/逐对象标签/预测/权重均不提交。

## 复现命令

主环境合成测试：

```bash
.venv/bin/python -m pytest -q tests/test_m8j6_selection.py tests/test_m8j6_ingest.py tests/test_m8j7_metrics.py
.venv/bin/python -m tools.m8j6_check_official --output /tmp/m8j7_official16
```

原有冻结和共享检查14项，在既有PETR环境禁GPU、禁网执行，命令同M8J6_RETRY_IMPLEMENTATION.md，无新增依赖。训练代码、协议与测试提交后，干净工作区执行一次：

```bash
.venv/bin/python -m tools.m8j7_run
```

run_id固定m8j7_20261027_131635；已有目录时不会重复更新。私有目录 /root/gpufree-data/hs-drive-private/m8j7_20261027_131635，公开统计 reports/m8j7_sources/m8j7_20261027_131635。checkpoint完整保留，不仅保留中间最优。
