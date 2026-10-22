"""Freeze historical diagnostic conclusions; no new model selection from regression."""

import json
from pathlib import Path
import numpy as np
from tools.m5o_common import write, asset, provenance
from datasets.m5o_benchmark import summary


def main():
    mobile = json.loads(Path("reports/m5o_frozen_regression.json").read_text())
    oracle = json.loads(Path("reports/m5o_frozen_regression_oracle.json").read_text())
    p = json.loads(Path(mobile["predictions"]["path"]).read_text())
    q = json.loads(Path(oracle["predictions"]["path"]).read_text())
    sup = json.loads(Path("runs/m5o_cache_regression/supervision.json").read_text())
    osup = json.loads(
        Path("runs/m5o_cache_regression_oracle/supervision.json").read_text()
    )
    records = []
    key = lambda a: (a["seed"], a["sample_id"], a["camera"], a["actor_id"])
    ob = {key(a): a for a in osup["actors"]}
    for seed in [101, 102]:
        aa = [a for a in sup["actors"] if a["seed"] == seed]
        matched = [a for a in aa if a["detected"]]
        errors = {}
        same = {}
        for method in [
            "plane",
            "dense_expected",
            "dense_mode",
            "geometry_prior",
            "direct",
        ]:
            subset = [
                a
                for a in matched
                if p[a["candidate_id"]]["positions"][method] is not None
                and q[ob[key(a)]["candidate_id"]]["positions"][method] is not None
            ]
            same[method] = {}
            for mode, preds, lookup in [
                ("mobile", p, lambda a: a["candidate_id"]),
                ("oracle_region", q, lambda a: ob[key(a)]["candidate_id"]),
            ]:
                vals = [
                    np.linalg.norm(
                        np.array(preds[lookup(a)]["positions"][method])[:2]
                        - a["box_center_ego"][:2]
                    )
                    for a in subset
                ]
                same[method][mode] = summary(vals)
        for a in matched:
            errors[a["candidate_id"]] = float(
                np.linalg.norm(
                    np.array(p[a["candidate_id"]]["positions"]["direct"])[:2]
                    - a["box_center_ego"][:2]
                )
            )
        records.append(
            dict(
                seed=seed,
                visible=len(aa),
                candidate_stage_missed=len(aa) - len(matched),
                matched_direct_over5=sum(v > 5 for v in errors.values()),
                matched_direct_within5=sum(v <= 5 for v in errors.values()),
                same_actor_view_comparison=same,
            )
        )
    write(
        "reports/m5o_oracle_candidate_diagnostic.json",
        dict(
            kind="actual_RTX4090_frozen_historical_regression_diagnostic",
            records=records,
            mobile=asset("reports/m5o_frozen_regression.json"),
            oracle=asset("reports/m5o_frozen_regression_oracle.json"),
            candidate=asset("reports/m5o_candidate_quality_regression.json"),
            provenance=provenance(),
            limits="M5-N historical sealed now regression only. Frozen region heads see changed candidate shape/context with Oracle regions; cannot isolate pure detector causal effect or retrained upper bound. Report same-target subsets, not only different natural coverage. Native instance masks never fed to formal RGB inference. Merged/split detections kept in historical diagnostic, excluded from new unique-association localization.",
        ),
    )
    print(records)


if __name__ == "__main__":
    main()
