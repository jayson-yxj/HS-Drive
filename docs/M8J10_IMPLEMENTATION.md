# M8-J10：等400步多场景对照

基础 eb61246。只训练 C，从 A 原始 safe tensor archive 和空 AdamW 开始。B 只读已有缓存；A 复用 M8-J9 官方结果。原八帧实验及历史报告不改动。

训练场景严格按冻结 devkit 的 mini_train ∩ train_detect∪train_track 计算，排除 mini_val/full val。结果六场景242帧，包含原八帧。按场景名排序轮流抽取，场景内 Python random.Random(seed+1009*scene_index) 独立洗牌；耗尽时同一 RNG 再洗牌。400步顺序已在任何训练前写入 reports/m8j10_training_manifest.json。此 RNG 不影响模型 Dropout。

训练沿用 M8-J7：FP32，bs1，512×1408×6，10类，AdamW lr1e-5/betas(.9,.999)/wd0/clip35，无 scheduler、warmup、AMP、TF32。141个可训练Parameter、178个冻结Parameter、159个冻结buffer；backbone/neck eval，head train/Dropout0.1。共享参数和冻结保护复用 M8-J6。只有样本池与顺序变化。预处理复用已验证官方转换代码，无随机图像增强。准备阶段对原八帧输入和监督逐数组精确比较。

第0步复用A，不新增A前向。C第200、400步各评估同一81帧，eval保存/恢复RNG与模块模式、核验参数和buffer未变。400为唯一最终比较，无中间checkpoint择优。保存200与400检查点，新进程验证400的模型/优化器/共享关系/冻结状态及固定输出恢复。

官方指标使用M8-J9原导出器、devkit1.1.9、detection_cvpr_2019、Top300/score≥0.2/原XY及官方范围/原速度属性规则，无NMS。项目诊断沿用3559GT，官方独立加载3393GT；两个分母不混用。B缓存完整性先核验，不默认需要新前向。逐对象预测、监督、图像、权重均私有存储。

运行阶段显式分开，失败停止不自动重试。已提交的干净工作区：

```bash
.venv/bin/python -m tools.m8j10_run prepare
# 检查准备结果并冻结实际训练manifest SHA后：
.venv/bin/python -m tools.m8j10_run train
.venv/bin/python -m tools.m8j10_run verify
.venv/bin/python -m tools.m8j10_run official
.venv/bin/python -m tools.m8j10_run report
```

GPU阶段禁网，不读取原始传感器载荷，输入白名单与监督隔离沿用既有实现。每个步骤持久化真实optimizer.step返回凭据，保护失败时保留失败checkpoint。预算400步到达即停。不因验证集结果修改学习率、过滤、类别或预算。
