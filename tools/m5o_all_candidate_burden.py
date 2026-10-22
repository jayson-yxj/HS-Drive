"""Full detection and ambiguity ledger, including Empty/Removed and accepted unknowns."""

import argparse, json
from pathlib import Path
from collections import Counter
from tools.m5o_common import rows, write, asset, provenance
from tools.m5o_candidate_audit import category


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    a = p.parse_args()
    root = Path(f"runs/m5o_cache_{a.split}")
    cc = json.loads((root / "candidates.json").read_text())
    sup = json.loads((root / "supervision.json").read_text())
    byid = {l["candidate_id"]: l for l in sup["labels"]}
    rmap = {r["sample_id"]: r for r in rows(a.split)}
    configs = {}
    preds = {}
    ev = Path(f"runs/m5o_eval_{a.split}/predictions.json")
    if ev.exists():
        data = json.loads(ev.read_text())
        for seed in [101, 102]:
            preds.update(data[f"{seed}_1.0"])
    for r in rmap.values():
        raw = Path(r["raw_frame"]).parent.parent
        if raw not in configs:
            configs[raw] = json.loads((raw / "manifest.json").read_text())[
                "configurations"
            ]
    records = []
    for group in sorted({r["group"] for r in rmap.values()}):
        for seed in [101, 102]:
            for phase in ["present", "empty", "removed"]:
                selected = []
                for c in cc:
                    if c["group"] != group or c["seed"] != seed:
                        continue
                    r = rmap[c["sample_id"]]
                    cfg = configs[Path(r["raw_frame"]).parent.parent]
                    idx = r["configuration"]
                    actual = (
                        "present" if cfg[idx] else "empty" if idx == 0 else "removed"
                    )
                    if actual == phase:
                        selected.append(c)
                counts = Counter(category(byid[c["candidate_id"]]) for c in selected)
                matched = sum(byid[c["candidate_id"]]["matched"] for c in selected)
                accepted = [
                    c
                    for c in selected
                    if preds.get(c["candidate_id"], {}).get("output", {}).get("status")
                    == "DetectedWithEstimate"
                ]
                records.append(
                    dict(
                        group=group,
                        seed=seed,
                        phase=phase,
                        candidates=len(selected),
                        matched=matched,
                        unmatched=len(selected) - matched,
                        semantic_proxy=dict(counts),
                        accepted=len(accepted) if preds else None,
                        accepted_unmatched_unknown=(
                            sum(
                                not byid[c["candidate_id"]]["matched"] for c in accepted
                            )
                            if preds
                            else None
                        ),
                    )
                )
    write(
        f"reports/m5o_all_candidate_burden_{a.split}.json",
        dict(
            kind="offline_actual_CARLA_RGB_all_candidate_ledger",
            records=records,
            unmatched_candidates=[
                dict(candidate=c, supervision=byid[c["candidate_id"]])
                for c in cc
                if not byid[c["candidate_id"]]["matched"]
            ],
            missed_actor_views=[x for x in sup["actors"] if not x["detected"]],
            ambiguous_actor_views=[
                x
                for x in sup["actors"]
                if x["detected"] and not x.get("unique_association", True)
            ],
            unassociated_actor_camera_entries=sum(
                r["unassociated_actor_camera_entries"] for r in sup["frames"]
            ),
            provenance=provenance(),
            limits="Visible denominator includes only reliably geometrically associated render identity >=3 native pixels; unassociated actor-camera entries include invisible/out-of-FOV and unresolved identity, not counted as proven visible misses. Empty means inserted actors absent; existing physical static geometry persists. All output candidates retained; unknown cannot be safe Free.",
        ),
    )


if __name__ == "__main__":
    main()
