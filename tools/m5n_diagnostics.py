"""Fixed coverage, actual-error correlation and same-position matched comparisons."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from tools.m5n_common import write, stats
from diagnostics.m5n_metrics import error, aggregate, same_target


def main():
    reports = []
    for split in ["val", "regression", "sealed"]:
        p = Path(f"runs/m5n_eval_{split}/predictions.json")
        if not p.exists():
            continue
        pred = json.loads(p.read_text())
        actors = json.loads(
            Path(f"runs/m5n_cache_{split}/supervision.json").read_text()
        )["actors"]
        for seed in [101, 102]:
            aa = [r for r in actors if r["seed"] == seed]
            matched = [r for r in aa if r["detected"]]
            sorted_actors = sorted(
                matched,
                key=lambda r: pred[r["candidate_id"]]["confidence"],
                reverse=True,
            )
            fixed = []
            for coverage in [0.3, 0.5, 0.7]:
                n = int(np.ceil(len(aa) * coverage))
                available = len(sorted_actors) >= n
                sel = sorted_actors[:n]
                ee = [
                    error(
                        pred[r["candidate_id"]]["positions"]["direct"],
                        r["box_center_ego"],
                    )["xy"]
                    for r in sel
                ]
                fixed.append(
                    dict(
                        requested_coverage=coverage,
                        available=available,
                        actual_coverage=len(sel) / len(aa),
                        xy=stats(ee),
                        limits="Ranking diagnostic at prescribed coverage, no test threshold deployment",
                    )
                )
            yy = np.array(
                [
                    error(
                        pred[r["candidate_id"]]["positions"]["direct"],
                        r["box_center_ego"],
                    )["xy"]
                    for r in matched
                ]
            )
            risk = np.array([pred[r["candidate_id"]]["proxy_error_m"] for r in matched])
            corr = spearmanr(risk, yy).statistic if len(yy) > 2 else None
            acceptable = []
            for n in range(1, len(sorted_actors) + 1):
                ee = [
                    error(
                        pred[r["candidate_id"]]["positions"]["direct"],
                        r["box_center_ego"],
                    )["xy"]
                    for r in sorted_actors[:n]
                ]
                if np.quantile(ee, 0.95) <= 5 and np.mean(np.array(ee) > 5) <= 0.05:
                    acceptable.append(n)
            groups = []
            for group in sorted({r["group"] for r in aa}):
                groups.append(
                    dict(
                        group=group,
                        comparison=same_target(
                            [r for r in aa if r["group"] == group], pred
                        ),
                    )
                )
            reports.append(
                dict(
                    split=split,
                    seed=seed,
                    fixed_coverage=fixed,
                    risk_error_spearman=(
                        float(corr) if corr is not None and np.isfinite(corr) else None
                    ),
                    max_posthoc_ranking_coverage_at5m=max(acceptable, default=0)
                    / len(aa),
                    posthoc_limit="Evaluation only, not selected test threshold; use frozen Validation operating point",
                    same_targets=same_target(aa, pred),
                    by_position_same_targets=groups,
                )
            )
    write(
        "reports/m5n_risk_coverage.json",
        dict(
            kind="offline_actual_model_empirical_risk_diagnostics",
            records=reports,
            validation_selection=json.loads(
                Path("reports/m5n_validation_lock.json").read_text()
            ),
            limits="Six Validation positions, checkpoint selected on same Val. Leave-position-out calibrator does not undo checkpoint selection leakage into calibration estimate. Not independent conformal or real-road error guarantee.",
        ),
    )


if __name__ == "__main__":
    main()
