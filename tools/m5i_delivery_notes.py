"""Postfreeze reporting clarification only: no models, thresholds or metrics changed."""
import json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5i_delivery_review.json')
    if out.exists():raise FileExistsError(out)
    files=['reports/m5i_summary.md','docs/M5I_ARCHITECTURE_DECISION.md','docs/M5I_IMPLEMENTATION.md'];before={p:file_hash(p)for p in files}
    p=Path(files[0]);s=p.read_text();s=s.replace('所有A/B/架构/C使用相同共享数据','实际运行的A/B/架构组使用相同共享数据').replace('真实训练的是FPN/PixelHead及C训练适配层','实际只训练FPN/PixelHead，C未执行').replace('教师特征256通道经实际resize映射，Student64->256适配层仅训练使用；通道L2归一化余弦，可靠前景/背景/边界均值等权。教师缓存无GT输入。','C的特征对齐与训练适配层已实现并通过合成梯度测试，但B在两seed均满足预声明验证收益条件，因此未生成教师特征缓存、未执行C训练，也不声称C实际有效。教师输出缓存不使用GT。')
    s += '''
## 最终决策与关键限制

- **视觉架构：建议下一阶段对 MobileNetV3 感知分支做独立迁移验收，暂不改默认 Encoder。** 在固定Test的6个位置，Mobile两个seed的AP均6/6改善，P/R同时改善分别6/6、5/6；但六摄P/R同时改善仅5/6、3/6，左侧Precision不足，不能宣称全六摄稳定通过。
- **输出蒸馏：有输出校准/排序收益，没有全面检测收益。** 两seed AP、Precision、Removed局部FPR改善，但Recall下降、Actor-view漏检升高；满足预声明B验证条件所以C不执行。没有测试后追加模型或Loss。
- **Depth：Replay不具备升级资格。** 所有逐场景近场和原S16回归通过，但全局/20–35m负Bias超限，新Test中远场同层1m质量下降。严格可关联Actor射线只有87条（6位置、vehicle/person），低于100门槛；语义前景135条不能充作135条关联Actor证据。
- **MAE与空间质量并不一致。** 新Test关联Actor的20–35m MAE由10.13m改善至7.74/7.49m，但同层1m质量由3.09%降至2.80%/2.78%；近场关联Actor MAE仍3.56/3.78m，同层1m质量也降。总体近场MAE受背景数量影响，不能代替障碍定位质量。
- **下一阶段优先视觉分支迁移与几何/Lift研究。** 新视觉显著提高BEV表面Recall，但精确表面Precision仍低；Replay附加收益小。先补足关联Actor的可靠中远场支持、评估真实采样密度和概率空间落点，不自动追加KD或Depth训练。正式Occupancy规模化、Temporal、Sparse Query和视觉控制均不启动。

本轮未使用新的独立Mesh真值。CARLA可见渲染首表面、实例关联及粗网格参考有共同来源和离散化限制。极小目标3–32像素仅6个Actor-view，全部模型该子集Recall均5/6，支持不足以判定稳定小目标优势。类别分项的Precision是二值目标头在“该类别正类+共享非目标背景”下的条件Precision，其他目标类别忽略；它不是多类分类混淆矩阵。

Oracle2D+OracleDepth同层1m Recall仍只有72.06%，因为参考来自全分辨率有效表面，而Lift使用固定P8采样。这是采样/观测覆盖限制，不能把所有未命中都归因于Depth学习。Teacher+OracleDepth使用原始软分数及固定0.2 BEV事件阈值，并非二维0.9工作点先二值化，低分背景累积可导致大量BEV误检；不能把该17.98% Precision解释为Teacher二维识别质量骤降。所有这些结果都是表面事件质量，不能解释为可通行Free或语义Occupancy概率。

实际训练6组×2400更新，两seed；每组约46–49s（含周期评测和Checkpoint写入，不含初次缓存加载），共享训练缓存下峰值约10.7GiB。单次推理峰值、预处理/Teacher成本另列，不将缓存训练耗时当成端到端训练总耗时。测试274通过、5跳过；CARLA SDK环境23通过为重叠子集，不能相加。跳过项及完整原始日志保留。Runtime继承Partial，Oracle驾驶未重验，纯视觉驾驶未验证。

图像见 [学习曲线](m5i_figures/m5i_learning.png)、[同类PR](m5i_figures/m5i_precision_recall.png)、[Depth偏置](m5i_figures/m5i_depth_bias.png)、[六组合BEV](m5i_figures/Town05_itest10_m5i_bev.png) 和 [后向视觉](m5i_figures/Town05_itest10_m5i_rear.png)。分层证据见 [补充诊断](m5i_findings.md)。
''';p.write_text(s)
    p=Path(files[1]);s=p.read_text().replace('输出/特征迁移是否在两seed池化P/R/Removed均改善','输出迁移是否在两seed池化P/R/Removed均改善').replace('C独立GT+Feature，不叠加B。','C独立GT+Feature的合同已实现，但B满足预声明验证收益条件，C本轮未执行，不得声称其训练有效。');s+='''

## 可执行的下一阶段方向

优先为Mobile视觉分支建立独立迁移版本，继续保留原Encoder/Depth对照，先审查左右侧背景误检及BEV接口；本轮不是默认替换授权。相比继续叠加KD，它在当前相同预算下提供更强的可复现证据。小目标支持不足、左侧Precision不足仍需独立门禁。

几何研究与视觉迁移分开：P8 Oracle采样仅72.06%表面Recall，LearnedDepth附加空间扩散明显；Replay降低MAE却降低新Test远场正确空间质量。先扩展可靠Actor评价支持，再预声明采样密度/概率投影对照；不更改48LID或默认Splat。当前不启动正式Occupancy规模化，只能在后续新决策中评估有限联合Pilot。

工程接口通过、二维学习相对增益、二维绝对门禁、Depth资格、BEV泛化、GPU算时、完整Runtime及驾驶安全分别判断；不存在单项收益自动关闭全系统门禁。
''';p.write_text(s)
    p=Path(files[2]);s=p.read_text();s+='''

## 本次执行结果索引

封存源码提交`01d5cf4`，冻结UTC与完整选择SHA在`m5i_pretest_freeze.json`；报告澄清脚本是冻结后的文字整理，不改变任何权重、阈值、协议或指标。实际A/B/架构共6组，C条件为False，未运行教师FPN缓存或特征蒸馏训练。数据新增6个Train/Val位置与6个封存位置，共192个训练/验证/测试新增场景样本及192帧参考。完整Test解析96帧，Train/Val共享400帧（272/128）。CARLA0.9.15 Epic六摄640×384，实际同步/位姿/重复图像审计通过。

M5-I模型与原始数据保留服务器，不将官方或训练权重上传Git。所有资产SHA在`m5i_artifact_manifest.json`，最终版本与推送结果另记。正式默认和旧Runtime未改，继承Partial，不以本轮GPU计时重新关闭控制实时性门禁。
''';p.write_text(s)
    atomic_json(out,dict(kind='postfreeze_documentation_only_review',before=before,after={p:file_hash(p)for p in files},scientific_assets_modified=False,source_sha256=file_hash(__file__),notes='Explicit C notrun, classconditional precision, P8Oracle samplingceiling, Actor support, architecture recommendation and independentgates. No testbased tuning.'))
if __name__=='__main__':main()
