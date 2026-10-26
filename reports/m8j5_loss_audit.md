# M8-J5：官方 PETR Loss 审计与独立包装层

2026-10-26 UTC；本机已存在官方 PETR f7525f93467a33707ef401c587a52d5e7b34de74、Python 3.8 / Torch 2.0.0+cu118 隔离环境。逐文件 SHA 见 m8j5_sources/frozen_loss_sources.json。本轮没有网络研究、重新下载或修改该环境源码。

## 原始行为

PETRHead._get_target_single（453–508 行）调用 HungarianAssigner3D 与 PseudoSampler，将所有未匹配 Query 的标签设为 num_classes=10、分类权重设为 1。gt_bboxes_ignore 非空在 get_targets/loss 及 Assigner 中被拒绝，并不支持 Ignore。

匹配使用 FocalLossCost 权重 2.0，加归一化 Box 前八维 L1 代价权重 0.25；IoU 代价为零。归一化不是将 XYZ 缩到 [0,1]：实际为 cx,cy,log(w),log(l),cz,log(h),sin(yaw),cos(yaw),vx,vy。位置仍以米计。匹配不使用速度两维。

分类为十类独立 sigmoid Focal Loss，alpha=0.25、gamma=2、loss_weight=2。背景标签 10 表示十类均为负。分类归一化为 max(P+bg_cls_weight*N,1)；当前配置 bg_cls_weight=0，**不表示背景分类损失关闭**，只表示背景数不进入分母。

回归为权重 0.25 的 L1，只有匹配正样本贡献；code_weights 前八维为 1、速度为 0.2；分母 max(reduce_mean(P),1)。原版 normalize_bbox 会对未匹配占位 Box 的零尺寸取 log(0)，随后排除非有限行并 nan_to_num 最终 Loss。

六个 Decoder Layer 各自重新匹配并计算 Loss，最终层输出 loss_cls/loss_bbox，其他层输出 d0..d4 共十二项。当前单阶段配置没有 encoder proposal loss；新包装层遇到两阶段输出会明确拒绝。DDP 与混合精度尚未实测。

真实合成边界测试发现旧 PseudoSampler 在零 GT 时产生 [0,4]，无法填入 PETR [0,9] 目标。初次失败日志保留；新代码不修补官方类，而是保留其 Hungarian assigner、直接从 assignment 构造正目标和权重。

## 新监督规则

training/m8j5_ignore_loss.py 为显式调用的无参数包装层，不注册新 Head、不 monkey patch forward、不修改默认配置。datasets/m8j5_loss_adapter.py 将经过实际审核的对象转换为监督，RGB 输入合同不变。

1. 只有审核通过且类别/几何可靠、可表达的目标进入 Hungarian Matching。拒绝畸形尺寸、非有限数值、非法类别。实际目标速度没有证据，速度两维回归权重为零；完整合成参考测试保留速度权重以验证原版一致性。
2. Positive 优先。其分类、可靠回归不受 Ignore 空间掩码误屏蔽。
3. 未匹配 Query 的预测几何 detach，构造 yaw 无关的 XY 外接圆 AABB 与完整高度，加 0.5m 边距。只有**整个包络落在一个已审核安全负区域**，且不与 Unknown/已知 Positive 包络相交，才成为 Negative。边界接触也拒绝。完整包络规则比中心点或置信阈值保守。
4. 其他未匹配 Query 为 Ignore，分类权重零；未匹配回归也为零。没有安全负区域的帧禁用负监督；这是证据缺失时的退化规则，不是将所有训练样本长期设为 Ignore 的方案。
5. 安全负区域证书必须覆盖全部十个模型类别，注明审核来源和范围；只有 car/person 的可见支持不能证明另外八类不存在。当前没有实际证书。Full-annotation 分支仅在完整标注证书及没有 Unknown 的条件下启用，本轮只用合成标注验证它。

空间包络是预测几何，不是物理真值；错误位置仍可能影响监督选择。0.5m 是预声明的保守合成策略，不是已校准真实不确定性。真实负区域仍需独立人工证据和后续误差审核，不能凭测试通过认证场景背景。

归一化保持原配置及 reduce_mean 方式，Ignore 不计为 Negative。只归一化正样本 Box，避免空目标 log(0)。空选取仍连到输入图，使全 Ignore 的梯度明确为零而非 None。非有限预测直接报错，不用 nan_to_num 掩盖。

## 实际测试结果

16 项隔离 PETR 环境测试通过，其中包含实际私有坐标合同检查。合成 GPU 梯度证据见 m8j5_loss_validation.json；测试日志保留 run1–run4 与 final，包括失败记录。

| 合成情况（4 Query） | P / N / I | 分类 Loss | 回归 Loss | 梯度证据 |
|---|---|---:|---:|---|
| 完整标注 | 1 / 3 / 0 | 10.223920 | 0.062500 | 与官方 Loss 及所有梯度 atol=1e-6 一致 |
| 部分标注 | 1 / 1 / 2 | 5.025317 | 0.062500 | 正类 logit 梯度 −0.149143；安全负类 +0.447430；Ignore 为 0 |
| 全 Ignore | 0 / 0 / 4 | 0 | 0 | 分类及回归梯度全部有限且为 0 |
| 零 Positive、有安全背景 | 0 / 1 / 3 | 2.599302 | 0 | 安全负分类有梯度，回归全零 |

另外验证 Positive 位于 Unknown 中仍生效、Unknown 与安全区重叠时 veto、只包含中心不包含车身时拒绝、接触边界拒绝、旋转包络稳定、未知速度梯度零、多帧混合批次、畸形标签/证书失败、所有六层辅助 Loss 完整一致。极高分数的 Query 可能改变官方 Hungarian 正匹配，这是预期行为；修正测试后，以零 Positive 隔离验证 Negative 判定不依赖分数，没有更改匹配代价。

24 项 CPU 合同测试通过。追加浏览器 Z 时间戳兼容检查后，6 项审核测试再次通过。浏览器合成 36 候选、14 项交互断言通过，未产生真实签核。

一次真实 00107 冻结前向与 M8-J4 完全一致：300 个 Box 最大差 0、分数差 0、类别完全相同。包装层没有破坏推理。当前真实规则为六层各 P=0/N=0/I=900、十二项 Loss 为零；这是 unsigned 数据不能启动训练的证据，不是训练成绩。没有真实 backward 或 optimizer.step。
