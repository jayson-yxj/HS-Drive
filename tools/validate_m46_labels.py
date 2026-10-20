"""Explicit label quality gates; limits declared independently of model fitting."""

import argparse, json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/m46_label_acceptance.json")
    p.add_argument("--source", default="reports/m46_occupancy_validation.json")
    p.add_argument("--dense", default="reports/m46_occupancy_dense.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    raw = json.loads(Path(a.source).read_text())
    dense = json.loads(Path(a.dense).read_text())
    records = []
    for row in dense["records"]:
        with np.load(row["path"]) as arr:
            d = {k: arr[k] for k in arr.files}
        s = d["occupancy_state"]
        unknown = s == 0
        check = dict(
            schema_shape=s.shape == (4, 160, 120),
            valid_exact=bool(np.array_equal(d["occupancy_valid"], ~unknown)),
            no_ego_free=not bool(np.any((s == 1) & d["ego_mask"])),
            no_unknown_semantic=not bool(np.any(d["surface_semantic_valid"] & unknown)),
            no_obstacle_unknown_negative=not bool(
                np.any(d["obstacle_valid"] & unknown)
            ),
            ground_not_obstacle=not bool(
                np.any((d["surface_semantic"] == 1) & d["obstacle_mask"])
            ),
            conflicted_semantics_ignored=not bool(
                np.any(d["semantic_conflict_mask"] & d["surface_semantic_valid"])
            ),
        )
        records.append(
            dict(
                scene=row["scene"],
                frame=row["frame_id"],
                checks=check,
                heldout_surface_false_free_rate=row["audit"][
                    "heldout_surface_marked_free"
                ]
                / row["audit"]["heldout_surface_voxels"],
            )
        )
    total = sum(r["audit"]["heldout_surface_voxels"] for r in dense["records"])
    ff = sum(r["audit"]["heldout_surface_marked_free"] for r in dense["records"])
    checks = dict(
        all_contracts_pass=all(all(r["checks"].values()) for r in records),
        self_visibility_check_pass=all(
            raw["checks"][k]
            for k in [
                "no_accepted_self_pixels",
                "every_recovered_ray_visibility_certified",
                "ego_voxels_not_free",
            ]
        ),
        heldout_false_free_is_zero=ff == 0,
        all_six_categories=raw["checks"]["all_six_scene_categories_present"],
    )
    atomic_json(
        a.output,
        dict(
            schema_version="hsd-m46-label-acceptance-v1",
            checks=checks,
            records=records,
            observed_heldout_false_free_voxels=ff,
            observed_heldout_surface_voxels=total,
            observed_heldout_false_free_rate=ff / total,
            engineering_status=(
                "Passed"
                if checks["all_contracts_pass"] and checks["self_visibility_check_pass"]
                else "Failed"
            ),
            strict_label_quality_status="Passed" if all(checks.values()) else "Partial",
            small_training_scope="Only explicitly masked observational-label plumbing/overfit research; not unrestricted occupancy or safety training. Zero-false-free quality gate remains open.",
            unresolved=[
                "Residual heldout surface-vs-Free conflicts in coarse cells",
                "Heldout pixels are same renderer, not independent geometry truth",
                "All data Town03, 36 static frames and 3 scripted-dynamic diagnostic frames",
                "Counterfactual capture cannot be assumed available for arbitrary driving sequences",
            ],
            source_sha256={p: file_hash(p) for p in [a.source, a.dense]},
        ),
    )


if __name__ == "__main__":
    main()
