# Coverage-Aware PETR：Loss、梯度与数值验收

2026-10-26 UTC，真实RTX4090执行；合成测试与真实缓存回放分开。固定seed850541、固定参数、同一前向图/GT，TF32关闭。官方版本及实际导入文件SHA见 `m8j5bv_sources/frozen_training_sources.json`。

## 原始行为及新接口

原始 Hungarian 对已给定GT独立匹配每层。未匹配Query是十类全负的背景，权重1；因此不完整GT直接进入原Loss会惩罚缺标目标。新机制只匹配可靠源Box；Positive优先，Ignore不删除匹配正样本；未匹配Query只有具备全类别、完整空间包络的独立证书才能变为Negative。图像未知掩码仅具有否决权。预测包络与掩码关系不是可靠性证据，真实帧用关闭负监督兜底。

六层分别输出最终 `loss_cls/loss_bbox` 和 `d0–d4`。分类分母为 `max(P+bg_cls_weight*N,1)`，回归分母为 `max(P,1)`。Ignore不进N；仅正样本归一化Box，避免占位尺寸log(0)。真实速度未知，vx/vy回归屏蔽。新增包装器显式锁定FP32监督，不改默认模型/推理。

## 明确容差与结果

| 对照 | 标准 | 实际 |
|---|---|---|
| Hungarian整数分配 | 完全相同 | batch1/2、六层相同 |
| FP32分类/回归Loss | atol=1e-6, rtol=1e-5 | 最大差0 |
| 直接分类/Box输出梯度 | 同上 | 最大差0 |
| 关闭Coverage后的完整模式 | 同上 | Loss和梯度差0 |
| 官方900Query Head的特征/参数梯度 | 同上 | 最大差0 |
| Ignore直接logits与无效Box梯度 | 精确0 | 所有边界用例满足 |
| FP16输出边界，scale512 | atol=5e-4, rtol=5e-4 | 分类梯度最大差7.96318e-5 |
| BF16输出边界，补充试验 | atol=4e-3, rtol=4e-3 | 最大差2.70680e-4 |
| 官方整网FP16特征/FP32 Head共享梯度 | 保留严格atol=1e-6, rtol=1e-5 | **失败**；最大约2.44e-4～3.05e-4 |

FP16/BF16边界容差在该测试首次运行时即明确设定；它们不用于放宽整网梯度验收。新旧整网Loss和直接输出梯度相同，但上游卷积/图像梯度不能通过严格标准。原版在同一图上重复反传也出现同量级波动，参数哈希未变化。最近复跑conv1最大差2.44140625e-4，原版自身重复也是2.44140625e-4；方向cosine约0.9999995。方向接近不替代数值验收。

实际官方路径：`petr3d.extract_feat`的auto_fp16只包图像特征提取且out_fp32=True，Head运行在外层FP32；配置为Fp16OptimizerHook loss_scale512。完整合成测试保留官方with_cp=True，通过backward而非不兼容的autograd.grad(inputs=...)穿过重入checkpoint。具体非确定性算子尚未隔离，不能断言已定位DCNv2缺陷。

## P/I/N梯度方向与归一化

代表合成用例（batch1/FP32，所有六层）：

| 用例 | P/I/N | 目标类logit梯度 | 负类logit梯度 | Positive回归梯度最大值 | 非正回归/Ignore直接梯度 |
|---|---|---:|---:|---:|---:|
| Positive与Ignore重叠 | 1/3/0 | -0.14914338 | 不适用 | 0.24999997 | 0 |
| 只有认证安全Negative | 0/3/1 | 不适用 | +0.44743013 | 不适用 | 0 |
| 全Ignore | 0/4/0 | 不适用 | 不适用 | 不适用 | 0 |

梯度下降会提高正确Positive类logit，降低Safe Negative全部类logit。共享参数梯度仍可非零，实测共享linear权重梯度范数6.8499；这与Ignore输出直接梯度为零不矛盾。

分类权重是P+N；与归一化分母不是同一个量。合成bg_cls_weight=0.5时P=1,N=1，分类分母1.5、回归分母1、分类权重和2，正负损失均非零。真实N=0只剩正损失，不是完整训练。

## 保留的失败

所有失败日志和启动源码SHA保留，不清洗失败历史。

1. run1：错误地对整个Head施加FP16 autocast，原版和新Loss均出现共享特征NaN。该执行方式不等于官方精度路径，不获批准。
2. run2：补充完整模型测试使用autograd.grad穿过官方重入checkpoint，触发不支持API错误。测试改用backward，未关闭官方checkpoint。
3. run3起：修复测试接口后梯度有限，但整网共享梯度不满足严格容差。加入原版重复反传对照后发现基线本身同样超限；失败仍保留。
4. recorded与recorded_repeat：最终53项用例都是52通过、1失败；51条核心数值记录逐项相同，完整模型的上游微小波动不同。全套GPU测试进程约10–11秒、peak allocated约849MB；这不是完整分辨率训练资源估算。

原31项CUDA回归仍通过，当前不新增真实前向或优化。A在严格清单下保持Partial，不能用Loss局部成功代替整网混合精度兼容性通过。
