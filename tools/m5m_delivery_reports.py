"""Final reports from immutable measured artifacts, no retuning or promotion."""

import json
from pathlib import Path
import numpy as np
from tools.m5l_common import write, asset, summary
from tools.m5m_selective_eval import evaluate


def main():
    data = {
        s: json.loads(Path(f"runs/m5m_contact_{s}.json").read_text())
        for s in ["train", "val", "regression", "test", "dev", "sealed"]
    }
    dense = json.loads(Path("runs/m5m_dense_objects.json").read_text())
    di = {
        (r["split"], r["sample_id"], r["seed"], r["camera"], r["actor_id"]): r
        for r in dense
    }
    agg = []
    detection = []
    strata = []
    risk = []
    threshold = json.loads(Path("reports/m5m_selective_lock.json").read_text())[
        "selected_threshold"
    ]
    for split, d in data.items():
        for r in d["actors"]:
            dd = di.get((split, r["sample_id"], r["seed"], r["camera"], r["actor_id"]))
            if dd:
                r.update({k: v for k, v in dd.items() if k.startswith("dense_")})
        for seed in [101, 102]:
            rr = [r for r in d["actors"] if r["seed"] == seed]
            for name, rule in [("A_frozen_m5l_plane", False), ("B_rgb_rules", True)]:
                agg.append(
                    dict(
                        split=split,
                        seed=seed,
                        method=name,
                        **evaluate(rr, threshold, rule),
                    )
                )
            dd = [r for r in d["detections"] if r["seed"] == seed]
            detection.append(
                dict(
                    split=split,
                    seed=seed,
                    candidates=sum(r["candidates"] for r in dd),
                    unmatched_candidates=sum(r["unmatched_candidates"] for r in dd),
                    known_visible=sum(r["visible"] for r in dd),
                    matched=sum(r["matched"] for r in dd),
                )
            )
            for field, fn in {
                "type": lambda r: r["type"],
                "group": lambda r: r["group"],
                "camera": lambda r: r["camera"],
                "range": lambda r: (
                    "near"
                    if r["distance_m"] < 10
                    else "10_20" if r["distance_m"] < 20 else "20plus"
                ),
                "road": lambda r: "slope" if abs(r["ego_pitch_deg"]) > 1 else "flat",
                "visibility_proxy": lambda r: (
                    "truncated"
                    if r["native_truncated"]
                    else (
                        "possible_occluded"
                        if (r["bbox_bottom_visibility_gap_px"] or 0) > 3
                        else "no_aabb_occlusion_flag"
                    )
                ),
            }.items():
                for v in sorted({fn(r) for r in rr}):
                    strata.append(
                        dict(
                            split=split,
                            seed=seed,
                            field=field,
                            value=v,
                            **evaluate([r for r in rr if fn(r) == v], threshold),
                        )
                    )
            for t in [0, 0.2, 0.4, 0.6, 0.8]:
                risk.append(
                    dict(split=split, seed=seed, threshold=t, **evaluate(rr, t))
                )
    write(
        "reports/m5m_selective_complete.json",
        dict(
            kind="actual_GPU_RGB_and_CARLA_surface_proxy_evaluation",
            aggregate=agg,
            detection=detection,
            strata=strata,
            risk_curves=risk,
            limits="Sealed reported at validation lock; curve is diagnostic only and never used to choose a test threshold. Dense comparisons use same matched RGB components and exact same rays. AABB center != visible surface != Gold contact. Native association can miss unassociated visible objects, so denominator is known visible actor-views, not exhaustive physical actors.",
        ),
    )
    refs = []
    for split, d in data.items():
        for typ in ["vehicle", "person", "cone"]:
            rr = [r for r in d["references"] if r["type"] == typ]
            refs.append(
                dict(
                    split=split,
                    type=typ,
                    visible=len(rr),
                    gold=sum(r["status"] == "Gold" for r in rr),
                    candidate=sum(r["status"] == "Candidate" for r in rr),
                    unknown=sum(r["status"] == "Unknown" for r in rr),
                    positions=len({r["group"] for r in rr}),
                )
            )
    write(
        "reports/m5m_contact_support_complete.json",
        dict(
            kind="offline_contact_support_audit",
            records=refs,
            contact_head=dict(
                trained=False,
                gate="NoGo",
                reason="Gold0 and no reliable Gold negative labels; new dev only two positions per split; no automatic anatomy-to-contact promotion",
            ),
            anatomy=[asset(f"reports/m5m_anatomy_{p}.json") for p in ["dev", "sealed"]],
        ),
    )
    gates = {
        s: json.loads(Path(f"reports/m5m_gating_{s}.json").read_text())
        for s in ["val", "regression", "test"]
    }
    write(
        "reports/m5m_vehicle_residual_gating.json",
        dict(
            kind="actual_RTX4090_frozen_four_residuals_two_seeds_no_training",
            validation_lock=json.loads(
                Path("reports/m5m_residual_lock.json").read_text()
            ),
            splits={s: asset(f"reports/m5m_gating_{s}.json") for s in gates},
            all_depth=[
                dict(split=s, **v) for s, r in gates.items() for v in r["depth"]
            ],
            bev=[dict(split=s, **v) for s, r in gates.items() for v in r["bev"]],
            empty_removed=[
                dict(split=s, **v) for s, r in gates.items() for v in r["empty_removed"]
            ],
            status="NoGo",
            reason="RGB gate protects aggregate20–35m but far35–80m background and per-position Bias protection fail. No new residual optimization. Binary foreground is not vehicle classifier. Unknown diagnostics not supervised negatives.",
        ),
    )
    shadow = json.loads(Path("reports/m5m_shadow_runtime.json").read_text())
    accept = dict(
        stage="M5-M",
        overall="Partial",
        engineering_interface="Passed",
        contact_gold_reference="NoGo",
        contact_head_training="NotRun_GoldGateFailed",
        selective_geometry_quality="NoGo",
        vehicle_residual_training="NoGo",
        object_query_pilot="NoGo_for_position_queries",
        shadow_process_isolation="Passed_static_architecture",
        shadow_resource_isolation="Partial",
        strict_20hz="NotPassed",
        formal_occupancy="Open",
        pure_visual_driving_safety="NotValidated",
        visual_control=False,
        main_merged=False,
        shadow_stop=shadow["stopped_for_tick_tail"],
        rationale={
            "contact": "Gold references absent; RGB selective tail gate cannot pass minimum coverage in both seeds",
            "vehicle": "All validation gates including per-position background protection required, not only target gain",
            "query": "Contract is usable for noncontrol diagnostics but reliable locations/confidence not demonstrated; defer target-position Query Pilot",
            "shadow": "Independent worker/latest packet/expiry tested; shared-GPU runtime costs remain, static trial not driving certification",
        },
    )
    write("reports/m5m_acceptance.json", accept)
    lines = [
        "# M5-M — 可信接触几何与选择性目标定位",
        "",
        "**结论：Partial。接触定位、Vehicle 残差继续训练、具有可信位置的 Object Query Pilot 均为 No-go；Shadow 仅通过静态进程隔离工程检查。**",
        "",
        "基于 `f2f4e1a`，独立分支 `feat/m5m-selective-object-geometry`。旧模型、48 Depth Bin、HeightSplat、控制器和 Runtime 默认行为保持不变。没有视觉控制、没有训练新残差或 Contact Head。",
        "",
        "## 接触参考与定位",
        "",
        "历史 Train/Val/M5-K Test/M5-L Test 均重新审计；旧测试仅作回归。补采4个开发位置、4个冻结后封存位置，每位置9配置。原生实例轮廓、可靠侵蚀像素分开统计。多点碰撞道路查询和渲染道路对照不是精确 Mesh 接触真值。",
        "",
        "所有参考仍无 Gold：轮胎物理圆半径和脚部骨骼原点尚未校准到可见轮胎接触斑/鞋底；遮挡、坡度和姿态稳定性也不足以认证。故 Contact Head 不训练，`VerifiedPosition=0`。候选保留 `Unknown` 和可选位置假设，Unknown 不表示不存在或 Free。",
        "",
        "|集合|seed|已知可见 Actor-view|检出|规则条件定位|Coverage|XY median / P95 / Max (m)|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in agg:
        if r["method"] == "B_rgb_rules" and r["split"] in ["val", "test", "sealed"]:
            e = r["xy"]
            lines.append(
                f"|{r['split']}|{r['seed']}|{r['visible']}|{r['detected']}|{r['conditional']}|{r['coverage']:.1%}|{e.get('p50',float('nan')):.2f} / {e.get('p95',float('nan')):.2f} / {e.get('max',float('nan')):.2f}|"
            )
    lines += [
        "",
        "预声明要求至少30%可见目标覆盖、20个匹配、3个位置、P95≤5m且不差于基线，>5m错误率≤5%。Validation置信度0.6仅留下8/76=10.5%；未通过覆盖门禁，不通过拒绝大多数目标换取成功。完整同目标Dense Expected/Mode、类别/相机/位置/距离分层和风险曲线见完整JSON。",
        "",
        "## 固定残差与道路平面",
        "",
        "无Mobile残差 seed101 的全部20–35m Bias：原始−0.357m、无条件−4.226m、RGB q≥0.9门控−0.493m；但35–80m仍由−9.833m恶化为−12.518m。seed102同方向。背景和逐场景门禁失败，停止这类残差训练。二值前景不能替代Vehicle分类；Pedestrian/Cone/Unknown分别记录，不宣称小样本类别泛化。",
        "",
        "固定地面与历史平路渲染路面约有0.25m高度偏差；局部Oracle平面和GT底部像素分别干预。Oracle道路只作诊断。原Depth估计的RGB局部平面不具备认证法向，不新增道路网络、不启用推理替换。",
        "",
        "## Shadow 实测",
        "",
        "唯一Tick Owner、独立Worker、2Hz最新完整六摄包、250ms原时间戳/墙钟过期拒绝；6秒Safety和控制代码未改。每条件两次，第二次交换idle/shadow顺序。首次推理和CUDA初始化独立记录。",
        "",
        "|重复|条件|Tick P95 / Max (ms)|",
        "|---:|---|---|",
    ]
    for r in shadow["phases"]:
        lines.append(
            f"|{r['repeat']}|{r['phase']}|{r['tick_ms']['p95']:.2f} / {r['tick_ms']['max']:.2f}|"
        )
    lines += [
        "",
        f"严重退化停止触发：{shadow['stopped_for_tick_tail']}。静态六摄资源试验不能外推驾驶实时性。队列put时间不包含feeder线程完整序列化，queue age包含等待/调度/传输，不能虚称已精确拆分。",
        "",
        "## 下一阶段决策",
        "",
        "1. **接触几何：No-go升级。** 先获取可核验轮胎/鞋底接触与可见性标注，保留条件假设。",
        "2. **Vehicle Depth：No-go继续同类训练。** RGB门控有研究增益，但远场背景保护失败。",
        "3. **Object Query：暂缓位置型Pilot。** 输出合同已就绪；可信位置与置信度尚未满足最低覆盖/尾部门禁，不等于SparseDrive或规划许可。",
        "4. **Shadow：工程隔离通过、资源隔离Partial。** 冷启动/稳态/顺序效应分别报告，进一步测试需维持纯观察及停止条件。",
        "",
        "完整复现、资产SHA、失败日志见 `docs/M5M_IMPLEMENTATION.md`。来源类型在每份JSON中明确标注；测试不等价于泛化或安全门禁。",
    ]
    Path("reports/m5m_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
