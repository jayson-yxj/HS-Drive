# M8-J5C 数值归因

2026-10-26 UTC，真实RTX4090，固定基础4143011。历史A=Partial、B=No-Go、C=NotRun保持原样。本轮结论：**FP32在明确的有限合成配置内通过；当前官方AMP不通过。**

## 控制条件

复用J5BV的官方模型、GT构造、Loss调用、完整标注合同，不重写Loss。seed850541；每次都从同一参数与缓冲区state_dict重新载入，清空参数梯度，克隆相同输入，恢复Python、NumPy、CPU和全部CUDA随机状态。比较前后state SHA及buffer均相同。每种精度/模式运行原版2次、新版2次，分别比较原/原、新/新、原/新。

PyTorch2.0.0+cu118、MMCV1.7.2、mmdet2.24.1、mmdet3d0.17.1。TF32关闭、cuDNN benchmark关闭、cuDNN deterministic开启、CUBLAS_WORKSPACE_CONFIG=:4096:8。原始with_cp=True保留。eval和train分别测试；train下Head/GridMask训练态、Backbone训练态、BN因官方norm_eval保持eval。随机状态复位使dropout/GridMask一致。

FP32不启用autocast或Scaler。实际AMP是官方extract_feat内FP16，返回FP32特征，Head和Loss为FP32。没有实例化GradScaler；显式loss×512反传后将参数、输入及各hook梯度除512比较。它验证静态缩放的数学路径，不认证GradScaler溢出处理或optimizer.step。

唯一容差仍为atol=1e-6、rtol=1e-5，没有事后放宽。相对误差采用 ‖g₁−g₂‖₂/max(‖g₁‖₂,‖g₂‖₂,1e-30)，cosine及异常张量逐项保留。

## 严格确定性开关的独立失败

run1显式开启torch.use_deterministic_algorithms(True)，官方Head的 `bbox_weights[pos_inds] = 1.0` 触发Torch2.0 CUDA Indexing内部断言（10 vs 1）。日志原样保留，没有改官方源码绕过。

run2与final显式使用 `--legacy-determinism` 关闭这一开关，仍保留cuDNN确定性及全部原容差。运行记录写明覆盖原因。**通过的是容差内可重复配置，不是逐bit确定性配置。** 严格Torch确定性路径单独No-Go。

## 从Decoder到最小算子

1. 固定六层900 Query输出：完整标注、Ignore不生效。FP32及实际AMP的FP32 Head输出，三组匹配、12项Loss、直接分类/框梯度均完全相同。
2. 整网合成输入[1,6,3,64,96]，全部参数梯度及关键模块边界。FP32 eval/train三组都满足原容差，最大绝对差7.90e-7。当前AMP三组都失败，最大约9.77e-4；方向接近不替代误差标准。详见[12组对照](m8j5c_gradient_comparison.md)。
3. 沿反向传播，从Head输出、Head输入特征至最后一个 `img_backbone.layer4.2.conv2` 的输出梯度均相同；该DCNv2的输入梯度首次超限。FP16该边界最大差3.05176e-5、相对L2约2.39e-4，随后向较早Backbone累积。该模块权重/offset梯度在这个边界保持一致。
4. 冻结捕获的DCNv2输入、参数和上游梯度，去掉Decoder、Loss、checkpoint独立反传，仍复现FP16输入梯度差异。FP32对应差约3.73e-9，在容差内。
5. 完全随机合成的独立DCNv2最小复现也得到FP16差1.52588e-5、relative L2约2.5e-4；FP32约3.73e-9。无需私有数据或真实Checkpoint。

已安装MMCV源码显示：modulated_deform_conv以offset dtype决定input/weight dtype；反向grad_input使用zeros_like(input)，CUDA col2im对grad_im使用atomicAdd。结合一致上游梯度与独立算子复现，证据支持**半精度DCNv2输入梯度的并行累加顺序/舍入波动**，而不是新Loss的匹配或归一化差异。这里没有宣称所有AMP故障均已解释，也没有修改CUDA内核。

## 精度决策与范围

| 配置 | 状态 | 可证实范围 |
|---|---|---|
| 固定Decoder输出的FP32监督 | Passed | 六层Matching/Loss/直接梯度 |
| 端到端FP32，Torch deterministic=False | Passed（有限配置） | eval/train，单seed，6×64×96合成输入，随机固定参数，无更新 |
| 官方Backbone AMP | No-Go | 全部三组共享梯度一致性失败 |
| Torch严格确定性 | No-Go | 原版索引赋值内部错误 |
| 真实512×1408前向/反向、实际Checkpoint、优化步骤 | NotRun | 本轮未批准真实更新，也未认证该资源范围 |

final整网诊断约30.37秒，峰值allocated603,850,752 bytes，仅适用于小合成输入。不能当成实际训练显存估计。两次有界整网诊断方向一致，旧失败全部保留。

复现：

```bash
# 完整三组对照；name必须是新的，禁网执行，上限600秒
.venv/bin/python -m tools.m8j5c_run_numeric --name reproduce --legacy-determinism
# 最小DCNv2；无需数据或Checkpoint
.venv/bin/python -m tools.m8j5c_run_numeric --name dcn_reproduce --legacy-determinism --minimal-dcn
# 重现已记录的Torch严格确定性失败（预期非零）
.venv/bin/python -m tools.m8j5c_run_numeric --name strict_reproduce
```

诊断进程正常完成的退出码0只表示报告已生成，**不是AMP门禁通过**。JSON每组比较单独保存passed状态。真实更新数为0。
