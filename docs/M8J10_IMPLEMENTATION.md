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

## 已执行准备与支持量

242帧，原始标注11317条，官方既有任务过滤后8200条，558个独立保留instance（连续关键帧记录不是独立目标）。零点数、范围、非任务类过滤可同时命中，原因计数不直接相加当唯一删除数。原八帧包含且转换数组精确一致，没有新增人工签核门禁。scene-0553、scene-0796虽属mini_train但不在完整版train交集，因此按固定官方规则排除。

固定顺序使每个场景获得66或67步；242个样本均至少出现一次，多数第二次。C计划目标记录曝光13668，B为335×50=16750；相同步数不等于相同GT数量、类别混合或样本重复率。此对照测试扩展样本池整体影响，不能隔离“独立场景数”这一单一因素。预训练原配置记录full train，C训练场景与其训练域已知重叠；mini_val既有开发集，不宣称新盲测。

准备阶段提交fb4fddf；完成准备后仅向配置补充已冻结原十类列表供导出使用，未改变任何训练、样本或评价规则（配置差异经preflight核验）。真正训练冻结于b6627ba，后续只增加离线汇总工具/测试，不改已运行训练源码或配置。

## 测试

```bash
.venv/bin/python -m pytest -q tests/test_m8j10_selection.py tests/test_m8j6_selection.py tests/test_m8j7_metrics.py
M8J10_RT=/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime
env M8J6_RUNTIME="$M8J10_RT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
 LD_LIBRARY_PATH="$M8J10_RT/env/lib/python3.8/site-packages/torch/lib:$M8J10_RT/cuda-11.8/lib" \
 PYTHONPATH="$PWD:$M8J10_RT/sources/PETR:$M8J10_RT/sources/mmdetection3d" \
 "$M8J10_RT/env/bin/python" -m pytest --noconftest -q \
 tests/test_m8j6_retry_state.py tests/test_m8j9_official_export.py tests/test_m8j10_selection.py
```

测试覆盖split交集、完整关键帧、验证隔离、场景轮换、不重复抽完前循环、独立随机状态、固定预算、原配置一致、参数别名/优化器唯一性、冻结检测及官方导出。初次runtime测试缺M8J6_RUNTIME导致一项官方模型合成检查skip；补明既有环境变量后38项全通过，无新环境安装或权重读取。随后增加两项配置/实际采样顺序检查，另保存最终测试记录，不覆盖初次日志。

## 可复用结果及退出

训练日志的逐步Loss对应不同样本，不作为整体拟合曲线；C200/C400固定mini_val eval Loss可比较，A基线本轮不追加Loss前向。官方A复用J9，B/C使用同一个原导出接口，官方TP不适用项原NaN保留。聚合JSON为易读将N/A映射null时明确注明，不改原devkit汇总。

项目诊断保留相同匹配、重复计数、垂直中心与框高分布、共同GT配对XY误差，整体/类别/两个场景分别报告。z/h超过1m和2m仅额外描述性计数，不作为训练选择或新增门禁；历史异常案例不重新开展逐对象调查。

私有根目录：/root/gpufree-data/hs-drive-private/m8j10_20261027_multiscene。公开报告只包含聚合数据、token级manifest、曝光计数和源码日志，不包含逐对象预测/标签或权重。全部阶段拒绝已存在的输出目录；复现不得原地覆盖或自动追加第401步。

## 实际结束状态

400步与保存恢复完成，训练源码提交b6627ba；恢复89f2d52、官方缓存评测9989bcb、离线汇总84239c7。全部阶段exit0。C400输出恢复逐位一致，optimizer141条状态的step均400。没有失败训练重跑或第401步。C验证162帧前向，保存probe2次、恢复probe1次，训练400次；A/B新增前向0。数据准备和官方/汇总均CPU禁网。

汇总命令（输出已存在时拒绝覆盖）：

```bash
.venv/bin/python -m tools.m8j10_verify
.venv/bin/python -m tools.m8j10_closeout
```

m8j10_closeout生成基础表格，m8j10_decision.md及验收补充字段为对真实结果的人工编写研究解释，不是新的自动门禁或统计显著性判断。所有官方原始metrics文件均原样保留。最终runtime43项、主环境33项测试通过，不把重复执行相同测试相加当独立测试数量。
