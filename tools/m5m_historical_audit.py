"""Explicitly retain frozen M5-L results and per-class coverage for comparison."""

import json
from pathlib import Path
from tools.m5l_common import write, asset, summary


def main():
    records = []
    for split in ["train", "val", "regression", "test"]:
        data = json.loads(Path(f"runs/m5l_objects_{split}.json").read_text())
        for seed in [101, 102]:
            for typ in ["vehicle", "person", "cone"]:
                rr = [
                    r for r in data["actors"] if r["seed"] == seed and r["type"] == typ
                ]
                v = [r for r in rr if r["valid"]]
                records.append(
                    dict(
                        split=split,
                        seed=seed,
                        type=typ,
                        visible_strict=len(rr),
                        detected=sum(r["detected"] for r in rr),
                        conditional=len(v),
                        geometry_refusal=sum(
                            r["detected"] and not r["valid"] for r in rr
                        ),
                        undetected=sum(not r["detected"] for r in rr),
                        plane_xy=summary([r["plane_center"]["xy_m"] for r in v]),
                        plane_visible_bottom=summary(
                            [r["plane_bottom"]["xy_m"] for r in v]
                        ),
                        dense_expected=summary(
                            [r["dense_expected_center"]["xy_m"] for r in v]
                        ),
                        dense_mode=summary([r["dense_mode_center"]["xy_m"] for r in v]),
                        trusted_unknown=len(rr),
                        fixed_plane_contact_supported=sum(
                            r["contact_matches_plane"] for r in rr
                        ),
                    )
                )
    paths = [
        "reports/m5l_summary.md",
        "reports/m5l_acceptance.json",
        "reports/m5l_object_2p5d_baseline.json",
        "reports/m5l_contact_contract.json",
        "reports/m5l_scene_protection.json",
        "reports/m5l_online_shadow.json",
        "diagnostics/m5l_objects.py",
        "tools/m5l_object_eval.py",
        "tools/m5l_vehicle_train.py",
    ]
    write(
        "reports/m5m_m5l_audit.json",
        dict(
            kind="offline_frozen_M5L_result_review",
            sources=[asset(p) for p in paths],
            by_class=records,
            findings=[
                "Fixed plane z=-.25 is engineering assumption, old bottom uses eroded renderer interior and is not tire/foot contact",
                "Dense baselines use identical RGB bottom rays and AABB center proxy; correct conditional median is not global localization",
                "No validated road normal, contact anatomy, occlusion confidence or multiclass RGB, therefore trusted positions remain Unknown",
                "New M5M native silhouette denominator exposes additional thin/truncated actor views; do not attribute denominator change to model improvement",
                "Vehicle-only residual target improvement coexists with far background bias, no residual budget extension",
            ],
        ),
    )


if __name__ == "__main__":
    main()
