"""Secondary absolute score audit on existing paired point witnesses; no selection.

Absence is at a rendered physical point, not certification of the whole coarse
voxel. Opposite-cell known obstacle veto is inherited from frozen pair protocol.
"""

import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m5c_local_witness_scores.json")
    if dest.exists():
        raise FileExistsError(dest)
    report = json.loads(
        Path("reports/m5c_absolute_obstacle_detection.json").read_text()
    )
    root = Path("/root/gpufree-data/hs-drive-data/processed/m5c_test_index")
    rows = json.loads((root / "index.json").read_text())["samples"]
    lookup = {r["sample_id"]: i for i, r in enumerate(rows)}
    pairs = json.loads((root / "m5c_pairs.json").read_text())["pairs"]
    records = []
    for model in report["records"]:
        with np.load(model["predictions"]) as z:
            score = z["score"]
        groups = {}
        details = []
        for pair in pairs:
            ia, ib = lookup[pair["a"]], lookup[pair["b"]]
            with np.load(pair["path"]) as z:
                for name, current, other in [
                    ("appeared", ib, ia),
                    ("vanished", ia, ib),
                ]:
                    mask = z[name]
                    p = score[current][mask]
                    a = score[other][mask]
                    th = model["threshold"]
                    n = len(p)
                    if not n:
                        continue
                    d = dict(
                        group=pair["group"],
                        a=pair["a_configuration"],
                        b=pair["b_configuration"],
                        event=name,
                        n=n,
                        presence_hits=int((p > th).sum()),
                        absence_residual=int((a > th).sum()),
                        both_correct=int(((p > th) & (a <= th)).sum()),
                        mean_present=float(p.mean()),
                        mean_absent=float(a.mean()),
                        direction_correct=int((p > a).sum()),
                    )
                    details.append(d)
        for group in sorted(set(d["group"] for d in details)):
            ds = [d for d in details if d["group"] == group]
            n = sum(d["n"] for d in ds)
            tp = sum(d["presence_hits"] for d in ds)
            fp = sum(d["absence_residual"] for d in ds)
            fn = n - tp
            groups[group] = dict(
                n=n,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=tp / max(1, tp + fp),
                recall=tp / max(1, n),
                iou=tp / max(1, tp + fp + fn),
                both_correct_fraction=sum(d["both_correct"] for d in ds) / n,
                absence_residual_fraction=fp / n,
                mean_present=sum(d["mean_present"] * d["n"] for d in ds) / n,
                mean_absent=sum(d["mean_absent"] * d["n"] for d in ds) / n,
            )
        records.append(
            dict(
                name=model["name"],
                seed=model["seed"],
                threshold=model["threshold"],
                by_scene=groups,
                details=details,
            )
        )
    atomic_json(
        dest,
        dict(
            kind="offline_fixed_checkpoint_and_threshold_secondary_point_witness_audit",
            records=records,
            source_sha256=file_hash(__file__),
            scope="Same frozen pair masks and singleframe score/threshold. point-presence/absence probes, not certified wholecell negatives. No tuning, no inference GT, no new model selection. Presence and disappearance directions duplicate physical configurations; report bygroup, not independent point trials.",
        ),
    )


if __name__ == "__main__":
    main()
