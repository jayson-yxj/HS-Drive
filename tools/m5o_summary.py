"""Write measured final research narrative from immutable report artifacts."""

import json
from pathlib import Path
from tools.m5o_common import provenance


def load(name):
    p = Path(f"reports/{name}.json")
    return json.loads(p.read_text()) if p.exists() else None


def main():
    acceptance = load("m5o_acceptance")
    support = load("m5o_dataset_support")
    val = load("m5o_eval_val")
    cal = load("m5o_eval_calibration")
    test = load("m5o_eval_sealed")
    training = load("m5o_training")
    paired = load("m5o_paired_method_analysis")
    oracle = load("m5o_oracle_candidate_diagnostic")
    lines = [
        "# M5-O — Perception Dataset & Localization Benchmark Reset",
        "",
        f"总体 **{acceptance['overall']}**。工程接口：{acceptance['engineering_interfaces']}；固定训练：{acceptance['finite_training']}；视觉定位增益：{acceptance['visual_localization_gain']}；Offline Pilot：{acceptance['offline_object_center_pilot']}。",
        "",
        f"基线 `2afe740`；独立分支 `feat/m5o-perception-benchmark-reset`；实验源码见逐报告 SHA 与最终 `m5o_artifact_manifest.json`。当前报告生成源码提交 `{provenance()['git_head']}`。不合并 main，不改变 Mobile、ResNet Depth、48 LID、HeightSplat 或 Oracle 控制；禁止视觉输出控制车辆。",
        "",
        "## 实际数据与可靠支持",
        "",
        "先完成失败径向布局试采，再改为采集前道路几何放置；旧数据和失败日志保留。第一组径向试采37次目标放置中23次无道路落点支持，可靠唯一候选仅6/7；道路几何版38/38有落点支持，候选30/31。此修订只使用Train通路审计。",
        "",
        "Town02容量不足只分配2组，未缩小隔离距离；追加Town04不同road_id且距Train Ego≥150m的4个Calibration位置。Train20、Val6、Calibration6、Sealed计划6。相同布局家族、相机重叠及同位置目标相关性明确保留；位置/道路片区是独立分析单位，Actor-view不是独立场景。",
        "",
        "|Split|seed|可靠唯一候选|独立位置|Vehicle|Person|Cone|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in (support or {}).get("records", []):
        if "seed" in r:
            lines.append(
                f"|{r['split']}|{r['seed']}|{r['unique_candidates']}|{r['groups']}|{r['by_class'].get('vehicle',0)}|{r['by_class'].get('person',0)}|{r['by_class'].get('cone',0)}|"
            )
    for r in acceptance.get("sealed_support", []):
        lines.append(
            f"|sealed|{r['seed']}|{r['unique_candidates']}|{r['groups']}|{r['by_class'].get('vehicle',0)}|{r['by_class'].get('person',0)}|{r['by_class'].get('cone',0)}|"
        )
    lines += [
        "",
        "上述数量是预声明研究支持，不等于定位或安全合格。监督是CARLA渲染实例与OBB几何关联代理；Center是八顶点均值，不是Mesh质心或接触Gold。可见分母仅包括可靠实例关联且≥3原生像素的Actor-view，无法关联参考条目另计，不能宣称覆盖所有物理目标。冻结Encoder见过部分历史相邻道路，详见lineage audit。",
        "",
        "## 1. 候选阶段造成多少失败？",
        "",
    ]
    for r in oracle["records"]:
        lines.append(
            f"历史M5-N回归 seed{r['seed']}：{r['visible']}个可见Actor-view中，候选漏检{r['candidate_stage_missed']}，匹配后Direct误差>5m仍有{r['matched_direct_over5']}。"
        )
    lines += [
        "",
        "历史未匹配候选172/142个不能全部解释为物理FP。Oracle实例区域将检测分母中的75个目标全部提供给冻结Head，但同目标定位尾部仍明显；这是输入分布变化诊断，不是重新训练后的检测上界。详见 `m5o_oracle_candidate_diagnostic.json` 和每组完整候选图。",
        "",
    ]
    if test:
        q = load("m5o_candidate_quality_sealed")
        for r in q["aggregate"]:
            lines.append(
                f"新Sealed seed{r['seed']}：总候选{r['candidates']}，可见Actor-view {r['visible_actor_views']}，匹配{r['matched']}，漏检{r['missed']}，未匹配未知{r['unmatched_unknown']}，分裂Actor-view {r['split_actor_views']}，合并候选{r['merged_candidates']}。"
            )
    lines += [
        "",
        "## 2–3. 数据规模影响与Direct相对Geometry Prior",
        "",
        "使用旧DirectCenterHead，Direct600维、Prior24维；两种子、25/50/100%嵌套位置组、每臂600步，相同候选/样本顺序/监督。只有Validation位置宏平均XY选择Checkpoint，无预算扩展。参数量不相等，因此是固定既有方法对比，不是参数量受控消融。",
        "",
        "|seed|训练位置比例|方法|Validation覆盖|平均XY m|P95 m|>5m比例|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in (val or {}).get("records", []):
        if r["method"] in ["direct", "geometry_prior"]:
            e = r["xy"]
            lines.append(
                f"|{r['seed']}|{r['fraction']:.0%}|{r['method']}|{r['coverage']:.1%}|{e['mean']:.2f}|{e['p95']:.2f}|{e['severe5m_rate']:.1%}|"
            )
    lines += [
        "",
        "自然覆盖与相同目标对比不可混称；`m5o_paired_method_analysis.json` 给出同目标、逐位置差异和整位置bootstrap。少量位置、共享地图和布局相关性使这些区间仅具描述意义。数量增加后是否改善均值、P95与严重错误必须分别判断。",
        "",
    ]
    if training and training["status"] == "Completed":
        lines.append(
            "真实RTX4090训练共12臂，累计时间 %.1fs，最大峰值PyTorch分配 %.1fMiB。日志记录每任务Loss、梯度、Checkpoint SHA与完整采样顺序；原视觉/Depth骨干冻结。"
            % (
                sum(r["elapsed_s"] for r in training["records"]),
                max(r["peak_mib"] for r in training["records"]),
            )
        )
    lines += [
        "",
        "## 4. 独立风险校准",
        "",
        "风险模型只用预测距离、框大小/区域、前景概率和边缘截断代理。Calibration与Validation位置分离；留一Calibration位置得到经验风险，阈值仅由Calibration选取，不能使用真实误差作推理字段。经验90%半径不是分布无关保证。无合格阈值时1.1拒绝全部准入，仍保留原始位置供研究。",
        "",
    ]
    for split, d in [("calibration", cal), ("sealed", test)]:
        for r in (d or {}).get("risks", []):
            lines.append(
                f"{split} seed{r['seed']}：风险/实际误差Spearman={r['spearman']:.3f}，经验90%区间实际覆盖={r['interval_coverage']:.1%}；按风险排序事后诊断中，严重误差≤5%的最大可见目标覆盖={r['max_coverage_at_empirical_severe5m_le05']:.1%}。Test曲线不用于重新选阈值。"
            )
    lines += [
        "",
        "## 5. 类别、相机和距离限制",
        "",
        "可靠支持及逐类/相机/距离/位置/遮挡代理/坡道误差在 `m5o_dataset_support.json` 和 `m5o_eval_*.json`。低于20候选或3位置的类别/相机不作独立可靠结论。训练距离以0–40m为主，不具备40–80m泛化证据；不能外推远距离能力。",
        "",
    ]
    if test:
        lines += [
            "|seed|方法|Sealed覆盖|平均XY m|Median m|P90 m|P95 m|Max m|>5m比例|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in test["records"]:
            e = r["xy"]
            if not e["n"]:
                lines.append(
                    f"|{r['seed']}|{r['method']}|{r['coverage']:.1%}|—|—|—|—|—|—|"
                )
                continue
            lines.append(
                f"|{r['seed']}|{r['method']}|{r['coverage']:.1%}|{e['mean']:.2f}|{e['median']:.2f}|{e['p90']:.2f}|{e['p95']:.2f}|{e['max']:.2f}|{e['severe5m_rate']:.1%}|"
            )
    lines += [
        "",
        "## 6–7. 架构决策与下一阶段",
        "",
        f"现有Direct Head路线：**{acceptance['direct_head_route']}**。限定Offline Object-Center Pilot：**{acceptance['offline_object_center_pilot']}**。",
        "",
        "本轮不以Loss或平均误差替代尾部、严重错误、独立Calibration和全候选负担门禁。若新数据支持不足，不能声称充分数据下Head失败；若支持充分但固定方法仍不达标，停止自动加层/调Loss/延长预算。没有在本轮验证替代3D表示方案，不能直接将替代架构标为合格；可复用此Benchmark进行后续预声明对照。",
        "",
        "## 工程状态与复现",
        "",
        f"测试：{acceptance['tests']}；SDK日志 `runs/m5o_logs/sdk_tests_final.log`。工程接口、研究学习、泛化与安全状态独立；M5-M Runtime Partial继承，无新Runtime矩阵，无视觉控制。",
        "",
        "Oracle状态、BEV Evidence、目标候选的Shadow合同和源时间戳已整理，Unknown不能变成Free；现有控制代码未改。主项目Oracle Planner/仿真工程可独立推进，其成绩不代表RGB闭环。",
        "",
        "命令、参数与限制见 `docs/M5O_IMPLEMENTATION.md`。所有原始日志位于 `runs/m5o_logs`，数据位于 `/root/gpufree-data/hs-drive-data/{raw,processed}/m5o*`，模型位于 `runs/m5o_training`；源码/资产SHA见各报告和最终manifest。Git推送包含源码/协议/报告/图，未自动上传大数据与权重。",
        "",
        "诊断图：`m5o_figures/m5o_data_scaling.png`、`m5o_training.png`、`m5o_calibration_risk_coverage.png`、`m5o_sealed_risk_coverage.png`及全部位置六摄候选拼图。",
    ]
    p = Path("reports/m5o_summary.md")
    assert not p.exists()
    p.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
