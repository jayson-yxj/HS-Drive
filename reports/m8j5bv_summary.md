# M8-J5B Task 4 验证总结

日期：2026-10-26 UTC。独立分支 `feat/m8j5b-loss-validation`，基础 `92ee99a`。结论：**Partial；没有真实数据梯度更新，训练Smoke仍No-Go。**

| 独立门禁 | 状态 | 依据 |
|---|---|---|
| A — LossImplementationReady | **Partial / No-Go** | 核心Loss、匹配、掩码、直接梯度及FP32 Head通过；整网混合精度共享梯度严格一致性仍失败 |
| B — RealSupervisionReady | **No-Go** | 214帧无可信Safe Negative证书；N=0；人工抽检Pending |
| C — GradientSmokeReady | **No-Go / NotRun** | A/B未全部满足，未配置真实更新任务 |

历史32项CPU、31项PETR/CUDA、9项浏览器检查通过；新增15项CPU通过，53项CUDA中52通过、1失败。另有30视图像素一致性、214帧逐对象追溯和四帧24层缓存回放通过。未把这些不同粒度检查相加宣称全套通过。

完整标注条件下，六层Hungarian、分类/回归Loss和直接输出梯度与官方完全一致；关闭新机制可恢复完整数据原路径。Positive与Ignore重叠仍有正常分类/回归梯度；Unknown直接logits梯度严格0；合成Safe Negative正常产生分类梯度，回归0。FP32官方Head参数梯度差0。

额外整网混合精度试验发现：梯度有限、直接输出梯度差0，但上游共享梯度最大差约2.44e-4～3.05e-4，超过预先规定容差。原版自身重复反传也存在同量级差异，具体算子尚未定位。保留失败和非零退出码，未放宽标准。详见[Loss与数值报告](m8j5bv_loss_validation.md)。

实际数据复核：3741条非自车记录、1276条车辆/行人几何均保留；570条正样本诊断候选Box与源转换逐值一致。四个缺标区域（含00124行人）全部实例区域像素落入Ignore，没有制造3D Box。局部图像Ignore占原图0.1433%、crop0.2386%；它不是Query Ignore比例。见[覆盖及安全负监督](m8j5bv_safe_negative_coverage.md)。

每一层900 Query实际回归：

| 真实帧 | P | I | N |
|---|---:|---:|---:|
| 00107 | 4 | 896 | 0 |
| 00124 | 3 | 897 | 0 |
| 00196 | 2 | 898 | 0 |
| 00213 | 2 | 898 | 0 |

全部六层的正负Loss、权重和分母见[24行统计表](m8j5bv_decoder_statistics.md)。未匹配Query的全帧Ignore是缺少真实覆盖证书的退化策略，不能解释为负学习完成。

私有六摄页面：`/root/gpufree-data/hs-drive-private/m8j5bv-full-validation/web/index.html`。四个对照图复用历史本机资产，未上传。30个视图展示mask与Loss mask严格一致；真实人类签核为Pending，未生成Golden标签。

下一步仅建议：先隔离上游混合精度梯度波动；再判断能否为一小批真实样本提供全类别、3D范围及遮挡约束明确的负监督证据。无法认证则维持N=0并停止更新。无需继续补签36个候选来掩盖覆盖不足。复现、接口与边界见[实施文档](../docs/M8J5BV_VALIDATION.md)。

默认模型、原始数据、历史报告和检查点保持不变。本轮无数据下载、CARLA、真实模型反向、优化器更新或main合并。
