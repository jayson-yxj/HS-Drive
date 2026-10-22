"""Conservative research decisions built from actual frozen experiment outputs."""

import json
from pathlib import Path
import numpy as np
from tools.m5n_common import write, asset, stats
from diagnostics.m5n_metrics import error, same_target


def main():
    splits = ["train", "val", "regression", "sealed"]
    evals = {
        s: json.loads(Path(f"reports/m5n_eval_{s}.json").read_text()) for s in splits
    }
    lock = json.loads(Path("reports/m5n_validation_lock.json").read_text())
    all_agg = [dict(split=s, **r) for s, d in evals.items() for r in d["aggregate"]]
    write(
        "reports/m5n_localization_comparison.json",
        dict(
            kind="actual_RTX4090_inference_offline_simulator_center_evaluation",
            aggregate=all_agg,
            strata=[dict(split=s, **r) for s, d in evals.items() for r in d["strata"]],
            same_target_and_fixed_coverage=asset("reports/m5n_risk_coverage.json"),
            sources=[asset(f"reports/m5n_eval_{s}.json") for s in splits],
            limits="Different methods have different natural coverage; same-target analysis is required for comparative claims. Dense region distribution refers visible surfaces, not necessarily OBB center; output not contactGold. Train metrics diagnostic only. All known visible actor denominator explicitly retained; unmatched candidate cannot be asserted falseobject.",
        ),
    )
    matching = []
    refs = []
    for split in splits:
        root = Path(f"runs/m5n_cache_{split}")
        s = json.loads((root / "supervision.json").read_text())
        cc = json.loads((root / "candidates.json").read_text())
        for seed in [101, 102]:
            ll = [r for r in s["labels"] if cc[r["feature_index"]]["seed"] == seed]
            aa = [r for r in s["actors"] if r["seed"] == seed]
            matching.append(
                dict(
                    split=split,
                    seed=seed,
                    known_visible_actor_views=len(aa),
                    detected=sum(r["detected"] for r in aa),
                    candidates=sum(c["seed"] == seed for c in cc),
                    unmatched_candidates=sum(not r["matched"] for r in ll),
                    matched_candidates=sum(r["matched"] for r in ll),
                    merged_candidates=sum(
                        r.get("merged_actor_count", 0) > 1 for r in ll
                    ),
                    split_candidates=sum(r.get("split_components", 0) > 1 for r in ll),
                    unique_supervision=sum(r["train_eligible"] for r in ll),
                    physical_falseobjects=None,
                )
            )
        if split == "sealed":
            rr = s["references"]
            refs.append(
                dict(
                    frames=len(s["frames"]),
                    positions=len({r["group"] for r in s["frames"]}),
                    references=len(rr),
                    max_world_closure=max(r["world_closure_residual_m"] for r in rr),
                    max_vertex_reconstruction_m=max(
                        r["vertex_reconstruction_residual_m"] for r in rr
                    ),
                    max_orthogonality=max(r["orthogonality_residual"] for r in rr),
                )
            )
    write(
        "reports/m5n_matching_complete.json",
        dict(
            kind="offline_all_set_matching_audit",
            records=matching,
            sealed_reference=refs,
            limits="Merged/split candidate counts are not independent actors; unique candidate supervision only. Full denominator limited by renderer-Actor association.",
        ),
    )
    acceptance = dict(
        stage="M5-N",
        overall="Partial",
        reference_engineering="Passed_simulator_OBB_center",
        RGB_input_isolation="Passed",
        finite_training="Completed",
        visual_localization_gain="NoGo",
        uncertainty_selectivity="NoGo",
        generalization="NoGo",
        offline_object_center_query_pilot="NoGo",
        contactGold="Unchanged_NotValidated",
        runtime="Inherited_M5M_Partial",
        pure_visual_control="NotValidated_Prohibited",
        default_modules_changed=False,
        main_merged=False,
        reason="Fixed Validation study failed coverage/tail/baseline superiority in both seeds. Freeze and stop600step; sealed cannot override failed Validation. Train fit successful but no stable cross-position feature gain over geometry prior. No additional training permitted by this protocol.",
        sealed_capture=asset("reports/m5n_sealed_capture_quality.json"),
        validation=asset("reports/m5n_validation_lock.json"),
    )
    assert not lock[
        "promoted"
    ], "Decision report must be reconsidered for measured qualification"
    write("reports/m5n_acceptance.json", acceptance)
    train = json.loads(Path("reports/m5n_direct_range_training.json").read_text())
    runtime = json.loads(Path("reports/m5n_runtime_budget.json").read_text())
    risk = json.loads(Path("reports/m5n_risk_coverage.json").read_text())
    lines = [
        "# M5-N — Direct Actor-Center Localization & Uncertainty Calibration",
        "",
        "**总体 Partial：仿真 Box Center监督合同和训练管线通过；Direct Head跨位置定位与选择性可靠性未通过。暂停当前训练路线，不申请位置型Object-Center Query Pilot。**",
        "",
        "基线`29ad67a`；独立分支`feat/m5n-direct-actor-localization`。原Mobile、ResNet Depth、48 LID、HeightSplat、Controller与Runtime默认实现未改。无视觉驾驶、无接触Gold升级。",
        "",
        "## 参考与数据",
        "",
        "八个已变换Box顶点均值定义仿真Box Center；反变换恢复局部中心、旋转与尺寸，Actor原点、可见表面均值、轮廓底部、物理接触分别保存。历史静态帧时间戳与标定检查通过；新采集显式记录Actor Snapshot帧号/时间及bbox局部参数。Box Center不是Mesh质心或物理接触Gold。",
        "",
        "复用Train/Val各122帧、6位置；每种子可靠唯一监督Train64/65、Val60/59个候选。Vehicle32/33训练，Pedestrian各13、Cone各19，后两类不具备泛化支持。K/L/M旧测试全部回归，不用于训练或阈值选择。M5-M两帧配对失败独立审计：单帧静态几何相容，但旧sidecar依赖配对自车Mask，仍不纳入主评价；没有自动认定单帧无效或有效。",
        "",
        "## 固定小预算训练",
        "",
        "冻结Mobile P8，候选框3×3采样共576维视觉特征加24维合法标定/区域特征；小MLP预测log光轴Z与候选中心相对Box中心投影偏移，再用K/T重建XY。两种子各600步；几何特征对照使用相同候选、样本顺序、预算。只有回归Head更新，没有NLL或新Depth训练。Checkpoint由Validation位置宏平均XY选择。",
        "",
        "|集合|seed|方法|定位/可见Actor-view|MAE|Median|P95|>5m比例|",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in all_agg:
        if r["split"] in ["val", "sealed"]:
            e = r["xy"]
            lines.append(
                f"|{r['split']}|{r['seed']}|{r['method']}|{r['localized']}/{r['visible']}|{e.get('mean',float('nan')):.2f}|{e.get('p50',float('nan')):.2f}|{e.get('p95',float('nan')):.2f}|{(r['severe5m_rate']or 0):.1%}|"
            )
    lines += [
        "",
        "表中是各方法自然覆盖，不能直接当成严格同目标消融。同目标比较见`m5n_risk_coverage.json`：Fixed Plane会拒绝困难目标；Dense摘要是同一RGB候选内原Depth概率均值的Expected/Mode，可能定位可见表面而非Box中心。没有在Test挑选摘要。",
        "",
        "Direct训练Loss接近零而Validation尾部未改善，符合小样本过拟合表现；几何先验在Validation的均值更低，未证明视觉区域特征带来稳定增量。未因短训练失败追加步数或调Loss。",
        "",
        "## 不确定性与拒绝",
        "",
        "用Validation留一位置拟合logXY误差的岭回归，仅输入预测位置/候选形状/前景概率。经验90%半径和置信分数不是真实误差上界。Validation只有6位置，且Checkpoint已使用同一Val选择，故不能称严格独立Conformal校准或真实道路保证。",
        "",
        "预声明30%覆盖、20匹配、至少3位置、P95≤5m、>5m≤5%，并在相同目标上优于Fixed Plane及Dense两摘要；两个种子没有共同通过的阈值。冻结阈值0仅保留研究估计，不代表准入。无法定位候选保留DetectedNoLocation；本轮任何输出都不得为VerifiedContact。",
        "",
        "## 七项决策",
        "",
        "1. **可以不依赖接触Gold训练Actor中心。** 仿真定义中心与接触合同分离，真实GPU有限训练已完成。",
        "2. **没有稳定优于全部冻结基线。** 相对区域Dense有收益，但未稳定优于Fixed Plane和合法区域几何先验。",
        "3. **置信度尚未满足选择性定位要求。** 经验校准与风险曲线已完成，不能同时满足覆盖与尾部门禁。",
        "4. **行人、遮挡、坡道尚无充分收益证据。** 分层保留，样本和位置支持不足，坡道新测试范围有限。",
        "5. **候选实例质量是限制之一。** 合并/分裂与漏检单列；唯一关联子集仍有跨位置误差，不能把全部失败归给候选实例。",
        "6. **停止当前600步Head路线的自动扩展。** 保留模型和参考接口；若另立研究，应先补独立目标尺度/车型/遮挡支持并与几何先验公平比较。",
        "7. **不批准Offline Object-Center Query Pilot。** 当前输出合同可作离线诊断，但位置准入与可信度未满足预声明门禁；更不涉及正式SparseDrive/Occupancy/Temporal/控制。",
        "",
        "## 工程与运行预算",
        "",
        f"实际RTX4090离线六摄：冷帧 {runtime['cold']['ms']:.1f}ms，稳态P95 {runtime['steady']['p95']:.1f}ms，峰值PyTorch分配 {runtime['peak_mib']:.1f}MiB。输入预加载GPU，不含磁盘/H2D/IPC。无CARLA竞争，不外推在线20Hz。",
        "",
        "M5-M Runtime Partial继承，本轮未重复静态Shadow压力矩阵。完整命令见`docs/M5N_IMPLEMENTATION.md`，原始日志/Checkpoint/SHA保留。",
    ]
    Path("reports/m5n_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
