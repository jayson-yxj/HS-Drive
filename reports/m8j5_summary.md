# M8-J5 — GT 审核闭环与 Ignore-aware PETR Loss

2026-10-26 UTC；基于 f8e9f9ca5f57976cac77ae3477efc224dcfdd281。分支 feat/m8j5-ignore-aware-petr-supervision。

**Partial：监督工程和合成梯度验证通过；36 个 Golden GT 候选已交付交互审核页，但实际人工签核为 0，因此真实参数更新 Smoke 为 NotRun，TrainingReadinessNoGo 保持。**

本轮将全部 214 组、3741 条非自车记录拆分为独立的 Geometry、Class、Identity、Visibility 与 Supervision Eligibility。1276 条车辆/行人 Box 具有代数几何和源类别支持；其中 712 条关联有自动证据，未视为人工确认。其余 2465 条不支持对象保留 Unknown。Identity Unknown 不再导致 Geometry Invalid：23 条地下行人的 Box 代数仍可自洽，但领域/可见性不成立，必须 Ignore。

712 与 711 的差额恰为一条车辆记录，因倾斜 2.308° 超过既有 2° yaw-only 条件而退出正监督候选。完整几何没有丢弃；该记录又处于 PETR 范围之外。逐条原因和对象证据保存在私有 filter_difference.json，公开只发布聚合原因。545 条满足自动证据与模型范围/裁剪的候选仍没有真实人工签核，不授权训练。

首批 36 个审核候选来自 27 帧，覆盖近/远车、行人、遮挡、相机边界、撤回身份冲突、地下异常与 PETR 范围边界；类别和阶段有相关性，不能视为独立场景。候选包包含每对象完整 Box、八角点、六摄投影、证据、内容 SHA 和 Pending 状态。此前 39 条撤回关联没有恢复。

按用户追加要求，原审核页已升级为**离线交互 HTML**，原静态版保留为 review_index_static.html。每候选单独绘制投影，提供“标注正确，审核通过／标注错误，拒绝／无法确定，暂不使用”。通过需确认几何、类别、身份、可见性四项；不符合模型表示/范围的对象不能批准为 Positive。审核者和证据来源本人填写，时间取实际点击的浏览器时钟并注明未平台校验。全部 36 项完成后生成可导出的模板 JSON，不连接服务器、不使用 localStorage、不自动落盘。刷新会丢失尚未导出的选择。

私有页面：/root/gpufree-data/hs-drive-private/m8j5-supervision/review_index.html。约 21.9MB，图像已嵌入，复制这一个 HTML 到本机即可离线审核。导出 m8j5_review_responses.json 后，由用户保存到同一服务器私有目录；浏览器无法自动写入远程服务器。JSON 校验器绑定候选内容 SHA、来源及四项确认，但不能认证真实审核者身份，必须保留人实际提交的来源。不生成或补造签核。

Ignore-aware Loss 保留官方 Hungarian、Focal/L1 及归一化，通过独立包装层筛选监督，不改模型或推理。安全负 Query 需完整预测包络落入审核区域并避开 Unknown/Positive；不使用置信分数决定 Ignore。该几何策略有合成测试支持，尚未证实真实空间准确性或真实负区域质量。详见 [Loss 审计](m8j5_loss_audit.md)。

| 验收项 | 实际结果 |
|---|---|
| 官方完整标注 Loss/梯度一致性 | 通过，atol=1e-6，含六层十二项 Loss |
| Ignore、Safe Negative、Positive 梯度 | 通过；Ignore 零梯度、负背景正常梯度、正目标正常分类/回归 |
| 全 Ignore / 零 Positive | Loss 与梯度有限；修复包装层对旧 PseudoSampler 空目标不兼容 |
| 隔离环境测试 | 16 项通过，0 跳过；合成梯度为主，含实际坐标审核 |
| CPU 合同回归 | 24 项通过；审核时间戳兼容修复后另复跑 6 项通过 |
| 离线审核 UI | Chrome 实测 36 个合成候选、14 项交互断言；真实页面截图已查看，无实际签核 |
| 冻结模型真实前向 | 00107 一帧成功，与 M8-J4 的 Box/分数/类别完全一致 |
| 当前真实 Query 监督 | 每层 P=0 / N=0 / I=900；无签核和安全负证书 |
| 人工 Golden GT | 0；Pending |
| 真实更新 Smoke | NotRun，用户指定的人工签核前置条件未满足 |

GPU 前向+解码单次冷运行约 977.35ms，峰值 PyTorch allocated 1,097,678,336 bytes、reserved 1,344,274,432 bytes；不是性能基准。合成 Head 测试峰值约 44.6MiB，不代表完整模型训练显存。本轮真实 backward=0、optimizer.step=0，无新数据下载、CARLA、模型替换或历史报告修改。

失败记录保留：普通 CPU 环境缺少已有审计模块所需 PIL，改用现有 .venv-gpu；PETR Python3.8 无法加载全项目 Python3.10 conftest，隔离测试用 --noconftest；旧空 GT sampler 形状不兼容；高置信度改变 Hungarian 正匹配的测试混淆已纠正；首次冻结回归被文件访问 canary 拒绝 Matplotlib 缓存，改将缓存放在允许输出目录，未放宽 GT 路径；浏览器 ISO Z 与 Python3.10 解析不兼容已修复并复验。没有删除失败来制造全通过。

进入 M8-J6 前至少需要实际审核者提交且通过校验的少量 Golden GT、确认其坐标和有效区域、验证安全负监督或明确限定为正样本学习可行性检查，并在此之后完成最多 2 帧/3 步的真实更新链路 Smoke（≤8GiB allocated、≤180s，失败即停）。该 Smoke 尚未执行，尚不能宣布完整训练链路通过。正式训练、泛化、Occupancy 与视觉控制均 No-Go/未验证。

交付索引：[验收](m8j5_acceptance.json)、[可信状态](m8j5_gt_trust.json)、[人工审核状态](m8j5_golden_review_status.json)、[Loss 验证](m8j5_loss_validation.json)、[Query 计数](m8j5_query_supervision.json)、[冻结前向](m8j5_frozen_forward_regression.json)、[UI 测试](m8j5_review_ui_validation.json)、[实现与命令](../docs/M8J5_IMPLEMENTATION.md)。私有图像、逐对象标签、审核及预测均不提交 Git；只提交代码、汇总和必要实验日志。
