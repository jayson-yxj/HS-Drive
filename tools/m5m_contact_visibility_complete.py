"""All-set native visibility/detection strata, with audit identifiers for each refusal."""

import json
from pathlib import Path
from tools.m5l_common import write, asset, summary


def main():
    out = []
    reasons = []
    for split in ["train", "val", "regression", "test", "dev", "sealed"]:
        data = json.loads(Path(f"runs/m5m_contact_{split}.json").read_text())
        for seed in [101, 102]:
            for typ in ["vehicle", "person", "cone"]:
                rr = [
                    r for r in data["actors"] if r["seed"] == seed and r["type"] == typ
                ]
                det = [r for r in rr if r["detected"]]
                out.append(
                    dict(
                        split=split,
                        seed=seed,
                        type=typ,
                        visible=len(rr),
                        detected=len(det),
                        undetected=len(rr) - len(det),
                        baseline_conditional=sum(r["baseline_valid"] for r in rr),
                        geometry_refusal=sum(
                            r["detected"] and not r["baseline_valid"] for r in rr
                        ),
                        native_truncated=sum(r["native_truncated"] for r in rr),
                        merged=sum(r.get("merged_actor_count", 0) > 1 for r in rr),
                        incomplete50=sum(
                            r.get("native_coverage", 1) < 0.5 for r in det
                        ),
                        vertical_bottom_error_px=summary(
                            [abs(r["pred_bottom_minus_visible_px"][1]) for r in det]
                        ),
                        aabb_occlusion_proxy=sum(
                            (r["bbox_bottom_visibility_gap_px"] or 0) > 3 for r in rr
                        ),
                    )
                )
        for r in data["actors"]:
            reasons.append(
                dict(
                    split=split,
                    **{
                        k: r[k]
                        for k in [
                            "sample_id",
                            "camera",
                            "actor_id",
                            "type",
                            "seed",
                            "detected",
                        ]
                    },
                    reasons=(
                        r["output"]["rejection_reasons"]
                        if r["detected"]
                        else ["undetected"]
                    ),
                )
            )
    write("runs/m5m_refusals_complete.json", reasons)
    write(
        "reports/m5m_visibility_complete.json",
        dict(
            kind="offline_actual_native_surface_GT_visibility_proxies",
            aggregate=out,
            refusals=asset("runs/m5m_refusals_complete.json"),
            limits="No exact tire visibility truth, AABB projected bottom is proxy. All GT metrics audit-only. RGB-only rule reasons are reproducible without GT.",
        ),
    )


if __name__ == "__main__":
    main()
