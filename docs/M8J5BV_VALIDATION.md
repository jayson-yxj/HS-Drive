# M8-J5B Task 4：完整验证与复现

执行日期：2026-10-26 UTC。基础提交 `92ee99ab68d33f3839fe29e7cf85d581ae6b4ec1`，分支 `feat/m8j5b-loss-validation`。本轮以 `m8j5bv_` 为新产物前缀；历史 J5/J5A/J5B 文件未修改。

本次新增显式调用的 `ValidatedCoveragePETRLoss`，保持官方 Head、默认推理、历史 Loss 不变。它在几何、匹配和损失入口关闭 autocast，并转换为 FP32；增加每层 P/I/N、分类权重、正/负损失分解、匹配结果和归一化分母。统计再次调用 Hungarian，仅用于离线可审计性，未声称可用于实时路径。只有完整、可信的全类别标注合同才允许 `coverage_enabled=False` 转回官方 Loss；部分覆盖拒绝关闭保护。

## 实测范围与失败

- 历史 CPU 32、历史官方 PETR/CUDA 31、浏览器 9 项通过。
- 新增无私有数据 CPU 15 项通过。
- 新增官方 PETR/CUDA 53 项：**52 通过、1 失败**。失败是完整合成模型混合精度共享梯度的严格一致性；没有 xfail、跳过或放宽阈值。
- 完整标注、batch 1/2、六层 Hungarian、12 项 Loss、直接输出梯度，以及 FP32 官方 Head 特征/参数梯度均一致。关闭覆盖机制的完整模式与官方一致。
- 9 类边界 × batch 1/2 × FP32/FP16，共36项；全部六层检查有限值、计数、分母、分类/回归梯度。另有 BF16 Loss 边界补充试验；BF16未批准为实际训练配置。
- 合成输入没有私有数据、Checkpoint或真实GT。真实四帧只回放原缓存输出，无真实前向/反传/更新。没有启动 CARLA。

GPU环境为独立 M8-A Python3.8 / PyTorch2.0+cu118 / MMCV / RTX4090。完整合成检测器输入仅 `[1,6,3,64,96]`、随机参数、eval模式、官方 `with_cp=True`、静态 scale512。它证明一条有限合成数值路径，**不证明完整分辨率、训练增强、训练模式或真实样本优化可用**。所有测得模型参数SHA在前后相同。

## 不依赖私有数据的复现

在仓库根目录执行。CPU测试只需现有CPU环境：

```bash
.venv/bin/python -m pytest --noconftest tests/test_m8j5bv_eligibility.py -q
```

官方CUDA合成测试需要现有隔离的官方PETR环境和GPU，但不读取私有数据或权重。Runner禁网，单次上限300秒，返回真实pytest退出码。当前预期为**退出码1（52 passed / 1 failed）**，不能包装成成功。

```bash
M8J5BV_PRIVATE_ROOT=/tmp/hs-m8j5bv-repro \
M8J5BV_REPORT_ROOT=/tmp/hs-m8j5bv-repro/reports \
.venv/bin/python -m tools.m8j5bv_run synthetic --suffix local
```

入口审计钩子拒绝访问 J3/J4/J5/J5A/J5B 私有数据路径（包括带后缀目录），并用不存在的canary路径检查拒绝行为。官方代码与已安装训练依赖SHA记录于 `reports/m8j5bv_sources/frozen_training_sources.json`。

## 需要服务器私有资产的复现

输出目录必须是新的，工具以排他创建方式保留旧证据。以下命令已经用独立 `m8j5bv-reproduction` 目录验证；再次执行应选另一个目录。

```bash
export M8J5BV_PRIVATE_ROOT=/root/gpufree-data/hs-drive-private/m8j5bv-repro-next
export M8J5BV_REPORT_ROOT="$M8J5BV_PRIVATE_ROOT/reports"
.venv/bin/python -m tools.m8j5bv_real_audit
.venv/bin/python -m tools.m8j5bv_run replay --suffix local
.venv/bin/python -m tools.m8j5bv_ui_regression
```

历史回归命令：

```bash
.venv/bin/python -m pytest tests/test_m8j5b_coverage.py tests/test_m8j5a_annotation_audit.py tests/test_m8j4_gt.py tests/test_m8j4_contract.py tests/test_m8j5_supervision.py -q
.venv/bin/python -m tools.m8j5bv_run baseline --suffix local
```

私有视图入口：
`/root/gpufree-data/hs-drive-private/m8j5bv-full-validation/web/index.html`。
复用已生成的五帧六摄图层：原始GT、转换GT、正样本诊断候选、缺标候选和局部Ignore。全帧负监督禁用在页面明确标注，不能把局部掩码外的图像当成安全负样本。30个视图的alpha掩码与Loss实际crop掩码严格一致。人工签核仍为Pending。

全部214帧的逐对象追溯、逐帧统计、mask SHA、真实Query匹配保存在私有目录，公开仓库只含源码、聚合指标与合成结果。不同相机投影/连续帧不是独立目标。

## 训练入口约束

1. 正样本仅来自可靠原始Box；Identity Unknown不能单独否决几何和类别可靠的Box。
2. 对未匹配Query，真实可见像素只能否决负监督，不能证明3D空间全类别无目标。局部预测投影错误不能授权Negative。
3. 没有独立安全负证书的真实214帧，一律 `frame_negative_enabled=False`。实际adapter拒绝载入声称已开启真实负监督的sidecar。
4. 图像缺标区域不会生成3D Box、Free、时间戳或Actor身份。速度监督保持关闭。
5. 全部6层执行相同规则。分类分母 `max(P + bg_cls_weight*N,1)`，回归分母 `max(P,1)`（单GPU）；当前背景权重参数0仅表示N不进入分母，不代表N没有分类损失。
6. 高Ignore及N=0发出诊断警告。完整空GT在官方历史PseudoSampler存在shape缺陷，不能宣称禁用新机制后该边界可恢复为有限Loss；新机制的空GT路径另行验证。

## 最小下一步

先用本轮现成合成脚本隔离混合精度上游梯度波动的具体算子，保持原阈值，禁止以改大容差直接关门禁。原版重复反传也超限，所以尚不能归因于Coverage Loss；也不能未经定位就断言一定是DCNv2。

其次只做少量真实样本的覆盖证据审查：能否取得覆盖目标类别、3D范围、遮挡和全部相机条件的独立负监督证据。若做不到，继续保持N=0和No-Go，停止训练准备；不靠语义背景像素造证书。只有证书、权重平衡、坐标一致性和真实审核者的定向抽检齐备，才能另行制定有限更新配置。本轮不请求补签36个Golden候选，也不执行优化。
