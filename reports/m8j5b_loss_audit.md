# M8-J5B 实际 PETR 训练路径与覆盖感知 Loss

检查日期：2026-10-26 UTC。官方 PETR 固定 `f7525f93467a33707ef401c587a52d5e7b34de74`，已安装隔离环境 Python3.8 / Torch2.0+cu118。对实际导入的 Head、Assigner、Focal、Box 编码和配置重新读源码并核对 SHA，与 M8-J5 冻结记录一致，见 `m8j5b_sources/frozen_loss_sources_verified.json`。

## 原版路径与风险

HungarianAssigner3D 只匹配提供的 GT：sigmoid Focal 分类代价权重2，加归一化 Box 前8维 L1 代价权重0.25；IoU不参与。未匹配 gt_inds=0，Head 将其 label 设为 num_classes=10、label_weight=1。sigmoid背景表示十个类别都是负。原版 `gt_bboxes_ignore` 必须为 None，没有图像覆盖 Ignore。

归一化 Box 是米制 cx,cy,log(w),log(l),cz,log(h),sin(yaw),cos(yaw),vx,vy，不是所有坐标缩为[0,1]。分类分母 max(P+bg_cls_weight*N,1)，当前 bg_cls_weight=0，仅指 N 不进分母，绝不等于关闭背景损失。回归只有匹配 Positive 有权重，分母 max(reduce_mean(P),1)。六个 Decoder Layer 分别重新匹配、计算分类与回归，输出最终两项及 d0–d4 共12项。

因此直接以不完整 Sidecar 调用官方 `head.loss()` 会把未标注目标对应的未匹配 Query 当背景，风险仍存在。M8-J5 老包装层有空间证书门禁，但只接人工签核且没有图像覆盖掩码。本轮提供独立显式 `CoverageAwarePETRLoss`，不注册/替换原 Head，不修改默认调用链。未来训练入口必须显式调用该包装层；未迁移入口仍不安全。

## 新规则

1. Positive来自现存原始 Box，经已验证几何、car/pedestrian类别、2° yaw-only表达限制、模型范围及实际crop内语义投影支持筛选。**不从缺标像素生成新 Box**。源Actor身份与render token不做数值连接；Identity Unknown不等于几何无效。未可靠目标不进入matching/regression。资格仅供本轮Loss诊断，不是Golden或训练批准。
2. 每层调用同一个官方 Hungarian Assigner，仅使用上述Positive。Positive优先，不受图像Ignore或未知包络误屏蔽。速度没有可靠监督，实际样本回归速度权重为零。
3. 对未匹配 Query，把detach的预测完整几何构造成yaw无关外接AABB（XY外接圆、完整高度、0.5m预声明边距）。投影到全部相机的实际模型crop。投影矩形触及任意Ignore像素则禁止背景分类；与近面相交、没有crop内可投影视图、非有限几何也拒绝负监督。不用置信度阈值，掩码选择无梯度。
4. 局部掩码只是**否决条件**。其外不能直接称为安全背景。Negative还必须满足完整包络处于一块覆盖全部十类的已验证3D安全区域，且不碰到未知/已知Positive包络，并且该帧允许负监督。未匹配的其他Query为Ignore，分类/回归权重为零。
5. 真正完整标注证书才允许完整模式；要求无Unknown与图像Ignore。此模式用于合成完整真值验证与原Loss/梯度等价。本片段没有该证书。
6. Ignore不进归一化N；仅Positive目标做Box归一化，避免空占位尺寸log(0)。全Ignore仍连接输入计算图，梯度严格为零。非有限预测/GT直接报错。零Query输入显式拒绝。两阶段encoder proposal不在已审计配置内，明确拒绝。
7. 每样本每层Ignore比例≥95%发出HighIgnoreFraction；N=0发出NoNegativeSupervision，写入机器报告。不能将该退化悄悄当成成功训练。

数学上，对未匹配query q，负权重为 `certified_3d_empty(q) AND NOT image_unknown(q) AND frame_negative_enabled`；Positive的分类与可靠回归权重始终为1。未知二维区域不沿深度反投影为正Box或安全Free。

## 真实局限与帧级退化

分割只描述可见表面；缺少原始时间戳，未验证遮挡后空间，且当前只支持car/person源标注，不能认证全部十类不存在。214帧没有安全负区域证书，所以**本片段全部采用帧级负监督关闭**。这不是通用算法“所有未匹配一律Ignore”：合成具有安全区域时，正常Negative梯度已实测。但真实样本N=0意味着负学习尚未成立，必须保留训练No-Go。

预测位置本身可能有错，未知图像投影veto可能漏掉错位的Query，或者遮住实际无关Query；仅凭局部掩码不保证安全。本轮真实策略以帧级退化兜底。0.5m/4px边距是受控保守规则，不是经物理误差标定的安全范围。

## 梯度证据

最终隔离GPU测试31通过，其中15项M8-J5B、16项历史回归。完整模式逐损失和全分类/回归梯度与官方一致（atol=1e-6），六层全部验证。

| 合成4 Query | P / I / N | 分类Loss | 回归Loss | 梯度结果 |
|---|---|---:|---:|---|
| 完整标注 | 1 / 0 / 3 | 10.223920 | 0.062500 | 与官方所有梯度一致 |
| 图像未知＋安全负区域 | 1 / 1 / 2 | 7.624618 | 0.062500 | Unknown分类/回归0；Positive目标logit负梯度；背景logit正梯度 |
| 全Ignore | 0 / 4 / 0 | 0 | 0 | 所有梯度有限且为0，发出两类警告 |
| 零Positive＋安全区域 | 0 / 3 / 1 | 2.599302 | 0 | 负分类有梯度，回归全0 |

Positive处于Ignore内仍有分类与回归梯度；高分未知Query仍被屏蔽；第二摄像头未知可否决第一摄像头；背景归一化权重非零的合成测试确认Ignore不计入N。全部记录见 `m8j5b_sources/gradients_final.json`。测试标签是合成真值，不冒充真实安全区域证书。
