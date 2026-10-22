"""Scene-aware selective localization, same matched targets and honest denominators."""

import json
from pathlib import Path
import numpy as np
from tools.m5l_common import write, asset, summary


def stats(values):
    r = summary(values)
    if values:
        r["p90"] = float(np.quantile(values, 0.9))
    return r


def evaluate(rr, threshold, rule=True):
    matched = [
        r
        for r in rr
        if r["baseline_valid"]
        and (not rule or r["rule_eligible"] and r["confidence"] >= threshold)
    ]
    det = [r for r in rr if r["detected"]]
    comparisons = {}
    for ref in ["center", "bottom"]:
        for axis in ["xy_m", "radial_error_m", "bearing_error_deg"]:
            same = [
                r
                for r in matched
                if all(
                    k + "_" + ref in r
                    for k in ["plane", "dense_expected", "dense_mode"]
                )
            ]
            comparisons[ref + "_" + axis] = {
                k: stats([r[k + "_" + ref][axis] for r in same])
                for k in ["plane", "dense_expected", "dense_mode"]
            }
    vals = [r["plane_center"]["xy_m"] for r in matched]
    bad = [r for r in matched if r["status"] != "Gold"]
    return dict(
        visible=len(rr),
        detected=len(det),
        conditional=len(matched),
        verified=0,
        positions=len({r["group"] for r in matched}),
        detection_rate=len(det) / max(len(rr), 1),
        coverage=len(matched) / max(len(rr), 1),
        undetected=len(rr) - len(det),
        geometry_rejected_given_detected=len(det) - len(matched),
        xy=stats(vals),
        severe5m_rate=sum(v > 5 for v in vals) / len(vals) if vals else None,
        unverified_accepted=len(bad),
        unverified_accepted_fraction=len(bad) / len(matched) if matched else None,
        false_accept_gold_unreliable=None,
        false_accept_note="No binary Gold reliability reference: unverified fraction is not measured contact error rate",
        matched_comparison=comparisons,
    )


def main():
    datasets = {
        s: json.loads(Path(f"runs/m5m_contact_{s}.json").read_text())
        for s in ["train", "val", "regression", "test"]
    }
    protocol = json.loads(Path("configs/m5m_protocol.json").read_text())
    curves = []
    for seed in [101, 102]:
        rr = [r for r in datasets["val"]["actors"] if r["seed"] == seed]
        base = evaluate(rr, 0, False)
        for threshold in protocol["selective"]["thresholds"]:
            ev = evaluate(rr, threshold)
            ok = (
                ev["coverage"] >= 0.3
                and ev["conditional"] >= 20
                and ev["positions"] >= 3
                and ev["xy"].get("p95", 1e9) <= min(5, base["xy"].get("p95", 1e9))
                and ev["severe5m_rate"] <= 0.05
            )
            curves.append(dict(seed=seed, threshold=threshold, passed=ok, **ev))
    admissible = [
        t
        for t in protocol["selective"]["thresholds"]
        if all(
            next(r for r in curves if r["seed"] == s and r["threshold"] == t)["passed"]
            for s in [101, 102]
        )
    ]
    selected = min(admissible) if admissible else 0.0
    write(
        "reports/m5m_selective_lock.json",
        dict(
            kind="validation_only_selection",
            selected_threshold=selected,
            promoted=bool(admissible),
            gold_qualification=False,
            curves=curves,
            limits="Threshold0 retained diagnostic if no admissible point; not deployment permission",
        ),
    )
    out = []
    risk = []
    calibration = []
    strata = []
    for split, data in datasets.items():
        for seed in [101, 102]:
            rr = [r for r in data["actors"] if r["seed"] == seed]
            for method, rule in [("A_frozen_m5l_plane", False), ("B_rgb_rules", True)]:
                out.append(
                    dict(
                        split=split,
                        seed=seed,
                        method=method,
                        **evaluate(rr, selected, rule),
                    )
                )
            for t in protocol["selective"]["thresholds"]:
                risk.append(
                    dict(split=split, seed=seed, threshold=t, **evaluate(rr, t))
                )
            groups = {
                "type": lambda r: r["type"],
                "group": lambda r: r["group"],
                "camera": lambda r: r["camera"],
                "road": lambda r: "slope" if abs(r["ego_pitch_deg"]) > 1 else "flat",
                "range": lambda r: (
                    "near"
                    if r["distance_m"] < 10
                    else "10_20" if r["distance_m"] < 20 else "20plus"
                ),
                "visibility_proxy": lambda r: (
                    "truncated"
                    if r["native_truncated"]
                    else (
                        "possible_occluded"
                        if (r["bbox_bottom_visibility_gap_px"] or 0) > 3
                        else "no_aabb_occlusion_flag"
                    )
                ),
            }
            for field, fn in groups.items():
                for v in sorted({fn(r) for r in rr}):
                    strata.append(
                        dict(
                            split=split,
                            seed=seed,
                            stratum=field,
                            value=v,
                            **evaluate([r for r in rr if fn(r) == v], selected),
                        )
                    )
            for lo, hi in zip([0, 0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8, 1.01]):
                sub = [
                    r for r in rr if r["baseline_valid"] and lo <= r["confidence"] < hi
                ]
                calibration.append(
                    dict(
                        split=split,
                        seed=seed,
                        lo=lo,
                        hi=hi,
                        n=len(sub),
                        mean_confidence=(
                            float(np.mean([r["confidence"] for r in sub]))
                            if sub
                            else None
                        ),
                        xy_within2m_fraction=(
                            float(
                                np.mean([r["plane_center"]["xy_m"] <= 2 for r in sub])
                            )
                            if sub
                            else None
                        ),
                        meaning="XY AABBcenter proxy empirical reliability, not Gold contact probability",
                    )
                )
    write(
        "reports/m5m_selective_localization.json",
        dict(
            kind="offline_actual_RGB_cache_and_renderer_reference",
            aggregate=out,
            strata=strata,
            lock=asset("reports/m5m_selective_lock.json"),
            limits="Native associated visible denominator, not all physical actors. A is identical frozen component/geometry with new denominator. Historical original M5L reports untouched. Unknown candidates retained; all trusted statuses Unknown. AABB center != contact. Same-target dense comparison requires available frozen historical rays; missing support explicit n.",
        ),
    )
    write(
        "reports/m5m_risk_coverage.json",
        dict(
            kind="offline_proxy_risk_not_Gold_certification",
            curves=risk,
            calibration=calibration,
            scene_stability=[r for r in strata if r["stratum"] == "group"],
            gate="NoGo" if not admissible else "ConditionalOnly",
            limits="No Gold contact calibration or independent scene significance claim; correlated actor-views counted, not iid frames.",
        ),
    )


if __name__ == "__main__":
    main()
