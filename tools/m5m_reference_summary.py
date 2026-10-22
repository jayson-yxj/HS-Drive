"""Reference status and explicit visibility-proxy audit; never auto-promote Gold."""

import json
from pathlib import Path
import numpy as np
from tools.m5l_common import write, asset, summary


def main():
    splits = ["train", "val", "regression", "test"]
    data = {
        s: json.loads(Path(f"runs/m5m_contact_{s}.json").read_text()) for s in splits
    }
    queries = json.loads(Path("runs/m5m_collision_queries.json").read_text())["records"]
    aggregate = []
    agreement = []
    vis = []
    for split, d in data.items():
        for typ in ["vehicle", "person", "cone"]:
            rr = [r for r in d["references"] if r["type"] == typ]
            aggregate.append(
                dict(
                    split=split,
                    type=typ,
                    actor_views=len(rr),
                    positions=len({r["group"] for r in rr}),
                    gold=sum(r["status"] == "Gold" for r in rr),
                    candidate=sum(r["status"] == "Candidate" for r in rr),
                    unknown=sum(r["status"] == "Unknown" for r in rr),
                    placement_flags={
                        k: sum(r["placement_flag"] == k for r in rr)
                        for k in [
                            "possible_floating",
                            "possible_penetration",
                            "box_near_road",
                            "unresolved",
                        ]
                    },
                    fixed_plane_height_error=summary(
                        [
                            r["fixed_plane_height_error_m"]
                            for r in rr
                            if r["fixed_plane_height_error_m"] is not None
                        ]
                    ),
                    aabb_road_gap=summary(
                        [
                            r["aabb_bottom_road_gap_m"]
                            for r in rr
                            if r["aabb_bottom_road_gap_m"] is not None
                        ]
                    ),
                )
            )
            for seed in [101, 102]:
                ar = [r for r in d["actors"] if r["seed"] == seed and r["type"] == typ]
                vs = [r for r in ar if r["detected"]]
                vis.append(
                    dict(
                        split=split,
                        seed=seed,
                        type=typ,
                        visible=len(ar),
                        detected=len(vs),
                        undetected=len(ar) - len(vs),
                        baseline_conditional=sum(r["baseline_valid"] for r in ar),
                        geometry_refused=sum(
                            r["detected"] and not r["baseline_valid"] for r in ar
                        ),
                        native_truncated=sum(r["native_truncated"] for r in ar),
                        merged=sum(r.get("merged_actor_count", 0) > 1 for r in vs),
                        incomplete_50pct=sum(
                            r.get("native_coverage", 1) < 0.5 for r in vs
                        ),
                        bottom_vertical_error_px=summary(
                            [abs(r["pred_bottom_minus_visible_px"][1]) for r in vs]
                        ),
                        aabb_occlusion_proxy=sum(
                            (r["bbox_bottom_visibility_gap_px"] or 0) > 3 for r in ar
                        ),
                        gold_contact=0,
                    )
                )
        for r in d["references"]:
            qq = next(
                (
                    q
                    for q in queries
                    if q["group"] == r["group"]
                    and q["type_id"] == r["type_id"]
                    and q["configuration"] == r["configuration"]
                ),
                None,
            )
            if qq and qq["collision_road_plane"] and r["rendered_local_road"]:
                cp = qq["collision_road_plane"]
                rp = r["rendered_local_road"]
                xy = r["center"][:2]
                delta = float(
                    np.dot(np.array(cp["coefficients"]) - rp["coefficients"], [*xy, 1])
                )
                agreement.append(
                    dict(
                        split=split,
                        sample_id=r["sample_id"],
                        actor_id=r["actor_id"],
                        camera=r["camera"],
                        type=r["type"],
                        group=r["group"],
                        ego_pitch_deg=r["ego_pitch_deg"],
                        collision_minus_render_road_m=delta,
                        aabb_collision_gap_m=qq["aabb_bottom_gap_m"],
                        aabb_render_gap_m=r["aabb_bottom_road_gap_m"],
                        candidate_only=True,
                    )
                )
    write("runs/m5m_reference_agreement.json", agreement)
    write(
        "reports/m5m_contact_reference_quality.json",
        dict(
            kind="recorded_renderer_audit_plus_actual_CARLA_static_collision_queries",
            aggregate=aggregate,
            collision_render_agreement=summary(
                [abs(r["collision_minus_render_road_m"]) for r in agreement]
            ),
            agreement_raw=asset("runs/m5m_reference_agreement.json"),
            queries=asset("runs/m5m_collision_queries.json"),
            gold=0,
            contact_head_training=False,
            reason="No independently identified and visibility-verified wheel/foot contacts in historical data; multiple road probes do not supply missing anatomy. New feasibility data reported separately.",
            contract=json.loads(Path("configs/m5m_protocol.json").read_text())[
                "contact"
            ],
            limits="Original capture freezes actors in ego-relative plane. AABB gap flags possible floating/penetration, not proof of exact rendered mesh contact. Native rendered bottom may be body not tire; slope magnifies placement error.",
        ),
    )
    write(
        "reports/m5m_contact_visibility.json",
        dict(
            kind="offline_GT_visibility_proxies_and_RGB_detection",
            aggregate=vis,
            candidate_counts=[
                dict(
                    split=s,
                    seed=seed,
                    candidates=sum(
                        r["candidates"] for r in d["detections"] if r["seed"] == seed
                    ),
                    unmatched_candidates=sum(
                        r["unmatched_candidates"]
                        for r in d["detections"]
                        if r["seed"] == seed
                    ),
                )
                for s, d in data.items()
                for seed in [101, 102]
            ],
            limits="GT-only diagnostics separated from inference. AABB projected lower edge is an occlusion proxy, not exact wheel/foot visibility. Unmatched candidates may be real unannotated objects; not guaranteed physical FP.",
        ),
    )


if __name__ == "__main__":
    main()
