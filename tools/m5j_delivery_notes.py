"""Reporting-only final interpretation; frozen scientific assets remain immutable."""
import json
from pathlib import Path
from tools.m5j_common import write
from tools.m5j_metrics import aggregate
from simulator.recorder import file_hash

def main():
    files=['reports/m5j_summary.md','docs/M5J_ARCHITECTURE_DECISION.md','docs/M5J_IMPLEMENTATION.md'];before={p:file_hash(p)for p in files};p=Path(files[0]);s=p.read_text();s+='''
## 七项最终回答

1. **Mobile独立迁移：Go，仅限显式离线研究Candidate。** 新Test Mobile P/R为81.10/86.08%、78.94/88.13%；原M5I同预算GT对照为40.79/45.23%、34.43/43.08%，补充只读回归确认相对收益。原M5G候选比较存在历史训练差异，不能单独解释为纯架构消融。默认Encoder不替换，绝对跨相机质量仍Partial。
2. **BEV错误来源不是单一累积。** 已确认背景高分格中，GT目标射线落到其真实端点以外的质量约51.18/41.81%，GT非目标源约30.43/38.47%，未知源约18.39/19.72%；低q<0.2质量仅2.40/6.09%。A包含低分非目标预测，不能把该份额全部叫高置信二维误检；B是粗格端点错位，不等于独立物理碰撞真值。概率扩散、重复bin和多射线累积可以叠加。
3. **匹配Recall后有小幅Precision收益，但聚合升级No-go。** Val匹配Recall的固定工作点在新Test上：gate101 P20.48/R46.25%，raw P17.40/R42.88%；gate102 P19.09/R45.44%，raw P12.50/R44.00%。正则化P20.60/15.64%、R45.75/46.00%。Test上实际Recall不必完全相等，因为匹配只在Val完成；没有使用Test插值重选阈值。两个seed未同时达到预声明5pp增益，不能借seed102单项通过改判。
4. **多位置有改善趋势，尚未稳定达到资格门槛。** 正则化在新Test四位置均提高Precision且Recall不降，但Val收益不足且Test多数增益不足5pp。gate101在Town01同层Recall下降2.3pp。没有逐Actor质心定位模型，当前只报告可见表面覆盖和Actor检出，不称精确对象定位。
5. **主要研究方向转向Depth空间落点和采样，但二维背景排除仍需保留。** matched Oracle干预下，Mobile+LearnedDepth raw P/R约17.96/39.50%、13.60/40.12%；Mobile+OracleDepth提高至30.52/69.88%、17.51/70.38%。Oracle2D+LearnedDepth原始Recall仍36.94%。S4 Oracle中心Recall80.88%，P8为68.94%，S16为48.44%；这是当前表面参考和采样相位下的诊断，不是严格信息论上界，也不能自动外推学习S4网络。
6. **有限BEV/Occupancy联合Pilot：本轮No-go。** 先补足可靠Actor与小目标支持、研究Depth概率几何与采样，保持有限且预声明的下一轮对照。本轮不继续叠加聚合校准模型，也不新增Depth微调。
7. **完整Runtime与纯视觉驾驶安全仍开放。** 完整离线双分支GPU P95约25.1–25.5ms、峰值约228MiB，但CPU读图/解码/SHA校验P95约76.6ms，序列化10.4ms、反序列化8.0ms、H2D3.6ms分别测量；不能相加这些分位数当端到端P95，更不能宣称20Hz闭环。本轮没有在线Shadow、没有修改Tick Owner、没有视觉控制。

新Test严格20–35m仅61条可关联Actor射线，全部为vehicle；person/cone此段无严格支持，泛化支持不足。Replay关联Actor远场MAE从10.11m降至7.82/7.58m，但同层1m质量从3.14%降至2.84/2.83%；近场也有MAE下降而空间质量下降。因此继续保持Replay不升级。

新Test小目标≤32像素仍仅6个Actor-view，33–128像素仍为0；采集预算没有达到小目标覆盖目标，这是一项明确失败，不能用“添加了锥桶”代替实际支持。Actor背景FP不能唯一归属某个GT Actor，因此逐Actor提供TP/FN/检出，背景FP在camera/scene/components报告。

原始raw事件阈值本轮按精确float 0.2计算；历史m5i部分统计使用floor(0.2×256)/256。公式数值等价已实测，但跨轮阈值边缘计数可有差异；本轮A/B/C全部同一精确阈值，不从历史数字拼接增益。PR直方图为256bin近似，最终工作点计数使用原始浮点分数。

完整测试最终记录在`m5j_logs/full_tests_final.log`；原始失败记录、旧源码快照和恢复方式保留。补充只读对照与绘图在冻结后增加，均不改变候选、参数、Mask、工作点和选择结果。
''';p.write_text(s)
    p=Path(files[1]);s=p.read_text();s+='''

## 具体执行边界

Mobile接口Go并非默认系统Go：只批准已实现的显式离线候选，原Depth/Context继续固定。新增Test相对同预算M5I GT基线也显著改善，见`m5j_matched_resnet_regression.json`。但4位置、6个小目标、零中等像素目标不足以关闭跨域泛化。

聚合No-go是预声明研究判据未满足，不是“数学上不能改善”。门控和正则化有有限收益，Oracle分解显示真实Depth替换收益更大，继续增加校准复杂度没有本轮证据。保留原raw主对照，候选代码只供诊断。

下一阶段先补可靠关联Actor中远场与目标面积覆盖，再对采样密度和深度概率空间质量建立受控实验；仍不擅自改变48LID、默认Splat或启动大规模Occupancy。CPU输入路径本轮为离线校验读取，不能自动成为在线控制链。
''';p.write_text(s)
    p=Path(files[2]);s=p.read_text();s+='''

## 本次实际执行补充

科学冻结提交`f26c0e4`，完整源SHA在`m5j_pretest_freeze.json`。新增4位置×20配置=80个六摄样本，另80帧自车移除参考；三处为8m内近路口。采集、同步、位姿与跨集合RGB重复审计通过。无训练，原Checkpoint全部只读。

附加历史同预算M5I GT回归与描述性绘图只读，无选择影响：

```bash
.venv-gpu/bin/python -m tools.m5j_matched_regression
.venv-gpu/bin/python -m tools.m5j_more_figures
.venv-gpu/bin/python -m tools.m5j_delivery_notes
.venv-gpu/bin/python -m pytest -q -rs > reports/m5j_logs/full_tests_final.log 2>&1
```

主报告和ADR在生成后经delivery_notes补充口径与具体Go/No-go解释；该脚本不修改冻结的模型/协议/阈值或任何原始实验JSON。错误输入附加合成测试覆盖NaN、缺相机、非刚体外参。
''';p.write_text(s)
    write('reports/m5j_delivery_review.json',dict(kind='postfreeze_documentation_only',before=before,after={p:file_hash(p)for p in files},source_sha256=file_hash(__file__),scientific_changes=False))
if __name__=='__main__':main()
