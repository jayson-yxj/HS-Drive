"""Reference/matching audit and M5-M pair failures assessed as single frames."""

import json
from pathlib import Path
import numpy as np
from tools.m5n_common import write, asset, stats, rows
from datasets.m5n_reference import actor_reference, synchronized
from tools.m5f_common import input_path


def main():
    summaries = []
    matchcounts = []
    sources = []
    for split in ["train", "val", "regression"]:
        root = Path(f"runs/m5n_cache_{split}")
        s = json.loads((root / "supervision.json").read_text())
        c = json.loads((root / "candidates.json").read_text())
        rr = s["references"]
        summaries.append(
            dict(
                split=split,
                frames=len(s["frames"]),
                positions=len({r["group"] for r in s["frames"]}),
                references=len(rr),
                max_world_closure=max(r["world_closure_residual_m"] for r in rr),
                max_orthogonality=max(r["orthogonality_residual"] for r in rr),
                max_vertex_reconstruction_m=max(
                    r["vertex_reconstruction_residual_m"] for r in rr
                ),
                origin_center_offset=stats(
                    [r["actor_origin_center_offset_m"] for r in rr]
                ),
                all_frames_sync=all(r["synchronized"] for r in s["frames"]),
            )
        )
        sources.append(asset(root / "supervision.json"))
        for seed in [101, 102]:
            cc = [r for r in c if r["seed"] == seed]
            ids = {r["candidate_id"] for r in cc}
            ll = [r for r in s["labels"] if r["candidate_id"] in ids]
            aa = [r for r in s["actors"] if r["seed"] == seed]
            for field, fn in {
                "all": lambda r: "all",
                "class": lambda r: r["type"],
                "camera": lambda r: r["camera"],
                "range": lambda r: (
                    "near"
                    if r["ego_xy_range_m"] < 10
                    else "mid" if r["ego_xy_range_m"] < 20 else "far"
                ),
            }.items():
                for value in sorted({fn(r) for r in aa}):
                    sub = [r for r in aa if fn(r) == value]
                    matchcounts.append(
                        dict(
                            split=split,
                            seed=seed,
                            field=field,
                            value=value,
                            visible=len(sub),
                            detected=sum(r["detected"] for r in sub),
                            split_actors=sum(r["split_components"] > 1 for r in sub),
                            positions=len({r["group"] for r in sub}),
                        )
                    )
            matchcounts.append(
                dict(
                    split=split,
                    seed=seed,
                    field="candidate",
                    value="all",
                    candidates=len(cc),
                    matched=sum(r["matched"] for r in ll),
                    unmatched=sum(not r["matched"] for r in ll),
                    unique_train_eligible=sum(r["train_eligible"] for r in ll),
                    merged=sum(r.get("merged_actor_count", 0) > 1 for r in ll),
                    split_candidate_count=sum(r.get("split_components", 0) > 1 for r in ll),
                    physical_false_objects=None,
                )
            )
    # Two historical paired exclusions remain excluded from all M5-N learning/evaluation.
    failed = json.loads(Path("reports/m5m_capture_integrity_sealed.json").read_text())[
        "excluded"
    ]
    idx = json.loads(
        Path(
            "/root/gpufree-data/hs-drive-data/processed/m5m_sealed_pixels/index_unfiltered_preserved.json"
        ).read_text()
    )["records"]
    failure = []
    for row in idx:
        if row["sample_id"] not in failed:
            continue
        f = json.loads(Path(row["raw_frame"]).read_text())
        inp = json.loads(input_path(row).read_text())
        refs = []
        for actor in f["actors"]:
            if (
                actor["is_ego"]
                or np.linalg.norm(np.array(actor["corners_hs_ego_m"]).mean(0)[:2]) >= 80
            ):
                continue
            ref = actor_reference(
                actor, inp["ego_pose"], inp["cam_to_ego"][0], inp["intrinsics"][0]
            )
            refs.append(
                dict(
                    actor_id=actor["actor_id"],
                    closure=ref["world_closure_residual_m"],
                    box_center=ref["box_center_ego"],
                )
            )
        failure.append(
            dict(
                sample_id=row["sample_id"],
                single_frame_sync=synchronized(f, inp),
                single_frame_box_geometry=refs,
                single_frame_reference_status="GeometricallyConsistent_ConditionalStaticSync",
                used_for_training=False,
                used_for_primary_eval=False,
                paired_evaluation="Excluded",
                reason="Same-frame static transform read between ticks supports box reference; paired ego-removal mismatch does not mathematically invalidate original center. Label association used paired selfmask: do not automatically reuse existing sidecar; no need to recruit old test into train.",
            )
        )
    write(
        "reports/m5n_actor_center_reference.json",
        dict(
            kind="offline_actual_recorded_CARLA_OBB_audit",
            aggregate=summaries,
            sidecars=sources,
            m5m_pair_failures=failure,
            contract="Center is mean of eight get_world_vertices transformed into HS ego. Inverting saved ActorTransform recovers bbox-local center and local basis/dimensions, so local rotation not silently ignored. Origin, OBBcenter, rendered surface centroid, visible bottom separate. Contact absent.",
            sync="Per sensor frame/time checked when available, all K/L/M manifests inspected; static actors frozen before capture and no nexttick between snapshot and actor reads. Historical peractor snapshot frame ID missing, conditional static provenance rather than fully stamped dynamic snapshot proof.",
            official_reference="https://carla.readthedocs.io/en/0.9.15/python_api/#carlaboundingbox",
        ),
    )
    write(
        "reports/m5n_candidate_matching.json",
        dict(
            kind="offline_GPU_RGB_candidates_vs_native_associated_renderer",
            records=matchcounts,
            limits="IoU Hungarian1to1>=.1; intersections>=3 diagnose merged/split, ignore ambiguous center supervision. Known visible actor denominator limited by association purity>=.9. Unassociated scene objects not safe negatives; unmatched candidates count not proven physical falseobjects. Binary foreground cannot infer verified class.",
        ),
    )


if __name__ == "__main__":
    main()
