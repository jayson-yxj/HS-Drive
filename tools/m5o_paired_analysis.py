"""Location-level paired method/scaling diagnostics; never test-driven selection."""

import json
from pathlib import Path
import numpy as np
from tools.m5o_common import write, provenance
from datasets.m5o_benchmark import METHODS, summary


def main():
    records = []
    scaling = []
    for split in ["val", "calibration", "sealed"]:
        p = Path(f"runs/m5o_eval_{split}/predictions.json")
        if not p.exists():
            continue
        predictions = json.loads(p.read_text())
        actors = json.loads(
            Path(f"runs/m5o_cache_{split}/supervision.json").read_text()
        )["actors"]
        for seed in [101, 102]:
            for fraction in [0.25, 0.5, 1.0] if split == "val" else [1.0]:
                preds = predictions[f"{seed}_{fraction}"]
                aa = [
                    a
                    for a in actors
                    if a["seed"] == seed
                    and a["detected"]
                    and a.get("unique_association", True)
                ]
                for baseline in [
                    "plane",
                    "geometry_prior",
                    "dense_expected",
                    "dense_mode",
                ]:
                    subset = [
                        a
                        for a in aa
                        if preds[a["candidate_id"]]["positions"][baseline] is not None
                    ]
                    groups = sorted({a["group"] for a in subset})
                    pergroup = []
                    direct = []
                    base = []
                    for g in groups:
                        rr = [a for a in subset if a["group"] == g]
                        errs = {
                            m: [
                                float(
                                    np.linalg.norm(
                                        np.array(
                                            preds[a["candidate_id"]]["positions"][m]
                                        )[:2]
                                        - a["box_center_ego"][:2]
                                    )
                                )
                                for a in rr
                            ]
                            for m in ["direct", baseline]
                        }
                        direct.extend(errs["direct"])
                        base.extend(errs[baseline])
                        pergroup.append(
                            dict(
                                group=g,
                                n=len(rr),
                                direct=summary(errs["direct"]),
                                baseline=summary(errs[baseline]),
                                delta_mean_m=float(
                                    np.mean(errs["direct"]) - np.mean(errs[baseline])
                                ),
                            )
                        )
                    deltas = [r["delta_mean_m"] for r in pergroup]
                    rng = np.random.default_rng(5199)
                    boot = (
                        [
                            float(
                                np.mean(rng.choice(deltas, len(deltas), replace=True))
                            )
                            for _ in range(1000)
                        ]
                        if deltas
                        else []
                    )
                    records.append(
                        dict(
                            split=split,
                            seed=seed,
                            fraction=fraction,
                            baseline=baseline,
                            same_target_n=len(subset),
                            direct=summary(direct),
                            baseline_metrics=summary(base),
                            group_delta_mean_m=(
                                float(np.mean(deltas)) if deltas else None
                            ),
                            group_delta_bootstrap95=(
                                np.quantile(boot, [0.025, 0.975]).tolist()
                                if boot
                                else None
                            ),
                            per_location=pergroup,
                            limits="Same-target descriptive paired location bootstrap; 6 locations/map correlation, not calibrated hypothesis test or safety interval",
                        )
                    )
    write(
        "reports/m5o_paired_method_analysis.json",
        dict(
            kind="offline_actual_GPU_predictions",
            records=records,
            provenance=provenance(),
            selection=False,
        ),
    )


if __name__ == "__main__":
    main()
