# M8-J6 独立 FP32 三步恢复重试

基线：`2d75f38cd64804af4ea681bb547712ac2be6cfef`，独立分支 `feat/m8j6-fp32-resume-smoke-retry`。
运行协议：`configs/m8j6_retry_protocol.json`；沿用原 `configs/m8j6_protocol.json` 的SHA，禁止修改原八帧、模型、类别、几何、优化器范围及超参数。

旧失败尝试完成过1次step，没有成功保存实验模型或优化器；本次不是补做2步。新run由原始safe tensor checkpoint初始化，优化器全新空状态。

## 监督与计算范围

- FP32、batch1、十类、六摄512×1408，输入和监督沿用已校验的私有NPZ。
- 样本索引固定 `[0,4,1]`，seed875606，AdamW lr1e-5、betas(.9,.999)、weight_decay0、clip35。
- Backbone/Neck eval，无图像骨干反向；仅原本可训练的检测头参数。固定 code_weights 仍冻结。
- 无scheduler：保存 `type=None / state=null`（JSON type字段为字符串`None`）。没有实例化AMP hook或GradScaler；TF32关闭。
- CUDA峰值allocated/reserved每步reset单独记录；该计量不包含驱动和其他进程，不等于整卡占用。

## 冻结和参数共享

复用旧 `partition_state`、`assert_frozen_unchanged`。新增 `parameter_mapping` 按实际 Parameter 对象聚合所有别名，先检查各别名的策略一致，再检查实际 requires_grad 和优化器归属。重复参数、冻结参数混入优化器、应训练参数缺失及矛盾别名策略均在step前拒绝。

公开映射记录规范名、全部别名、shape、dtype、训练资格和optimizer group/slot。仅进程内使用对象id去重，不保存或比较跨进程id。模型与优化器快照逐tensor `detach().cpu().clone()`，不持有可变状态引用。

每步对全部冻结state名称比较原始快照和本步更新前快照，并单独断言所有buffer未改变；可训练别名被正确识别，不是忽略别名检查或复制权重拆开共享。

## 2+1+只读验证的进程顺序

1. first进程：原模型+空optimizer，执行两次step，写入step2 checkpoint。
2. first退出后，resume新进程：重建同结构，核对原始权重、映射，加载step2。参数/buffer/optimizer逐项精确比较；共享关系、RNG、模式、下一样本位置、scheduler状态一致。固定eval输入比较通过后恢复训练随机状态，只执行第三步，保存step3。
3. verify新进程：独立实例恢复step3，以同样方式核对状态和固定eval输出，无optimizer.step。

固定eval检查会保存/恢复进入检查前的RNG和各子模块模式，不消耗后续训练随机数。输出容差预先固定atol1e-6、rtol1e-5；同时报告是否逐位相等。不能据此宣称和所有不间断训练逐位等价。

每次step前先落盘loss/梯度；step返回立即写入独立ledger，后续断言失败也不会遗失实际步数。异常立即保存私有失败checkpoint，supervisor停止，不自动重试。

## 命令

先运行原36项：

```bash
.venv/bin/python -m pytest -q tests/test_m8j6_selection.py tests/test_m8j6_ingest.py
.venv/bin/python -m tools.m8j6_check_official --output /tmp/m8j6_retry_reused16
```

新增CPU合成测试（使用现有PETR Python3.8，不访问私有数据）：

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  M8J6_RUNTIME=/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime \
  PYTHONPATH="$PWD:/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/sources/PETR:/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/sources/mmdetection3d" \
  LD_LIBRARY_PATH=/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/env/lib/python3.8/site-packages/torch/lib \
  unshare --net /root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/env/bin/python \
  -m pytest --noconftest -q tests/test_m8j6_retry_state.py
```

源码、协议、预检和测试结果提交后，干净工作区执行一次：

```bash
.venv/bin/python -m tools.m8j6_retry_run
```

运行器记录提交号和源码SHA，禁网，三个子进程各300秒上限。run_id目录已存在则拒绝重复运行。不应直接重跑GPU子脚本绕过一次性目录。

## 资产与结果

私有 checkpoint 保存在 `/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/<run_id>/`。只保存本次自生成checkpoint，验证哈希后在隔离环境加载。公开报告不含图像、逐对象标签或模型/优化器tensor。

新结果写 `reports/m8j6_retry_sources/<run_id>/`，旧失败报告不改写。最终验收必须同时满足新运行实际step=3、first/resume/verify调用数=2/1/0、step2跨进程恢复通过及step3独立重载通过。没有满足则Partial/Failed，不自动进入过拟合或长训练。
