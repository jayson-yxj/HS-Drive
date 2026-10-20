"""Aggregate frozen evaluations, preserving scene units and unsupported strata."""

import json
from pathlib import Path
from simulator.recorder import atomic_json, file_hash


def read(name):
    return json.loads(Path("reports", name + ".json").read_text())


def totals(rows):
    n = sum(r["n"] for r in rows)
    p = sum(r["presence_hits"] for r in rows)
    a = sum(r["absence_false_positives"] for r in rows)
    return dict(
        n=n,
        presence_recall=p / n if n else None,
        absence_fpr=a / n if n else None,
        both_correct_fraction=(
            sum(
                r.get("both_correct", r.get("both_correct_fraction", 0) * r["n"])
                for r in rows
            )
            / n
            if n
            else None
        ),
    )


def main():
    dest = Path("reports/m5d_evaluation_digest.json")
    if dest.exists():
        raise FileExistsError(dest)
    source = read("m5d_presence_absence")
    records = []
    for r in source["records"]:
        frames = r["frames"]
        actors = [a for f in frames for a in f["metrics"]["actors"]]
        eligible = [a for a in actors if a["eligible"]]
        witness = [v for p in r["witnesses"] for v in p["witness"]]
        by_actor = {}
        for kind in sorted({a["type_id"] for a in actors}):
            aa = [a for a in actors if a["type_id"] == kind]
            ee = [a for a in aa if a["eligible"]]
            n = sum(a["visible_cells"] for a in aa)
            by_actor[kind] = dict(
                placed=len(aa),
                zero_support=sum(a["visible_cells"] == 0 for a in aa),
                eligible=len(ee),
                detected=sum(a["detected"] for a in ee),
                surface_cells=n,
                exact_recall=sum(a["exact_hits"] for a in aa) / n if n else None,
            )
        within = {}
        for t in ["0.5", "1.0"]:
            matches = [f["metrics"]["spatial"]["tolerances"][t] for f in frames]
            p = sum(x["pred_count"] for x in matches)
            g = sum(x["gt_count"] for x in matches)
            within[t] = dict(
                gt=g,
                pred=p,
                unmatched_gt=sum(x["unmatched_gt"] for x in matches),
                extra_pred=sum(x["extra_pred"] for x in matches),
                recall=sum(x["matched_gt"] for x in matches) / g if g else None,
                precision=sum(x["matched_pred"] for x in matches) / p if p else None,
            )
        vals = list(r["presence_absence"]["by_scene"].values())
        records.append(
            dict(
                name=r["name"],
                seed=r["seed"],
                threshold=r["threshold"],
                witness_micro=totals(vals),
                by_scene=r["presence_absence"]["by_scene"],
                legacy_micro=totals(
                    list(r["legacy_same_definition"]["by_scene"].values())
                ),
                witness_by_distance={
                    b: totals([v["by_distance"][b] for v in witness])
                    for b in ["0_10", "10_15", "15_20", "20_35", "35_80"]
                },
                witness_by_height={
                    str(z): totals([v["by_layer"][str(z)] for v in witness])
                    for z in range(4)
                },
                direction_correct=sum(
                    v["direction_correct"] * v["n"] for v in witness if v["n"]
                )
                / sum(v["n"] for v in witness),
                probability_amplitude=sum(
                    (v["mean_presence"] - v["mean_absence"]) * v["n"]
                    for v in witness
                    if v["n"]
                )
                / sum(v["n"] for v in witness),
                actor_exact_recall=r["absolute"]["placed_actor_only"]["recall"],
                actor_detected=sum(a["detected"] for a in eligible),
                actor_eligible=len(eligible),
                actors=by_actor,
                spatial=within,
                known_surface=r["absolute"]["overall"],
                corridor=r["absolute"]["corridor"],
                unknown_high_fraction=sum(
                    f["metrics"]["unknown_predicted"] for f in frames
                )
                / sum(f["metrics"]["unknown_cells"] for f in frames),
                mean_scene_change_iou=r["change"]["mean_scene_iou"],
                empty_returns=r["empty_returns"],
            )
        )
    atomic_json(
        dest,
        dict(
            kind="offline_summary_frozen_actual_GPU_test_results",
            records=records,
            source_report_sha256=file_hash("reports/m5d_presence_absence.json"),
            source_sha256=file_hash(__file__),
            limits="Micro counts supplement scene metrics; do not treat voxels/configurations as independent trials. Actor weak criterion inherited, not object volume localization. Unknown high scores not proven false positives.",
        ),
    )


if __name__ == "__main__":
    main()
