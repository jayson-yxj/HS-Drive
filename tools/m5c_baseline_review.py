"""Freeze interpretation and re-evaluate historical opened M5-B change localization."""

import json
from pathlib import Path
import numpy as np
from diagnostics.m5c_localization import matching
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5c_spatial_localization.json")
    if dest.exists():
        raise FileExistsError(dest)
    report = json.loads(Path("reports/m5b_counterfactual_detection.json").read_text())
    index = Path("/root/gpufree-data/hs-drive-data/processed/m5b_index")
    rows = json.loads((index / "index.json").read_text())["samples"]
    rows = [r for r in rows if r["split"] == "test"]
    lookup = {r["sample_id"]: i for i, r in enumerate(rows)}
    pairs = [
        r
        for r in json.loads((index / "m5b_pairs.json").read_text())["pairs"]
        if r["split"] == "test"
    ]
    records = []
    for record in report["records"]:
        with np.load(record["predictions"]) as z:
            score = z["score"]
        for r in pairs:
            with np.load(r["path"]) as z:
                pair = {k: z[k] for k in z.files}
            delta = score[lookup[r["b"]]] - score[lookup[r["a"]]]
            domain = pair["appeared"] | pair["vanished"] | pair["stable"]
            th = record["threshold"]
            records.append(
                dict(
                    model=record["name"],
                    seed=record["seed"],
                    group=r["group"],
                    a=r["a_configuration"],
                    b=r["b_configuration"],
                    appearance=matching((delta > th) & domain, pair["appeared"]),
                    disappearance=matching((delta < -th) & domain, pair["vanished"]),
                )
            )
    atomic_json(
        dest,
        dict(
            kind="offline_historical_opened_M5B_predictions_no_model_or_threshold_selection",
            records=records,
            source_sha256=file_hash(__file__),
            old_metric_issue="appearance_localization ignored height; unmatched counted all GT only if zero predictions. Replaced by bidirectional same-height nearest-neighbor distances and per-GT tolerance miss counts. Many-to-one coverage, no one-to-one actor matching.",
            metric_taxonomy=dict(
                A="conditional Ground/Obstacle semantics only on reliable known surfaces",
                B="paired signed change on supported witness and unchanged known surfaces",
                C="singleframe visible obstacle surface score on heldout first-hit support; primary M5C target; still not entire unknown BEV or occupied volume",
            ),
            baseline_limitations=[
                "M5A108frames are12posegroups;3testgroups, no pixel independence",
                "M5B known test opened, historical regression only",
                "Free-ray evidence is not wholecell safe free",
                "Surface×conditional Obstacle is not calibrated physical occupancy probability",
                "AP improvement does not guarantee thresholded localization or missed-actor recall",
            ],
            new_absolute_results="reports/m5c_absolute_obstacle_detection.json",
            no_history_overwrite=True,
        ),
    )


if __name__ == "__main__":
    main()
