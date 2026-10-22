"""Check opt-in deployable estimator against independent cached offline evaluation."""

import json
from pathlib import Path
import numpy as np, torch
from tools.m5n_common import rows, write, stats
from tools.m5f_common import inputs
from models.m5n_candidate import DirectActorCandidate
from training.runtime import prepare_cuda


def main():
    prepare_cuda()
    torch.set_num_threads(2)
    pred = json.loads(Path("runs/m5n_eval_val/predictions.json").read_text())
    rs = rows("val")
    result = []
    for seed in [101, 102]:
        model = DirectActorCandidate(enable=True, seed=seed)
        for row in [rs[1], rs[27], rs[105]]:
            x = inputs(row)
            out = model(now_s=float(x["timestamps_s"][0]), **x)
            diff = []
            for r in out:
                cam, cid = r["candidate_id"].split(":")
                key = f"{row['sample_id']}:{seed}:{cam}:{cid}"
                p = pred[key]
                diff.append(
                    float(
                        np.linalg.norm(
                            np.array(r["position_ego_xy"])
                            - np.array(p["positions"]["direct"])[:2]
                        )
                    )
                )
            assert max(diff, default=0) < 0.002, diff
            result.append(
                dict(
                    seed=seed,
                    sample_id=row["sample_id"],
                    candidates=len(out),
                    xy_difference_m=stats(diff),
                )
            )
    write(
        "reports/m5n_inference_equivalence.json",
        dict(
            kind="actual_GPU_full_RGB_candidate_vs_cached_evaluation",
            records=result,
            passed=True,
            limits="Runtime uses fullVal calibrator; Val risk metric uses leave-group-out calibrator deliberately. Compared center positions, not expected-to-differ uncertainty. Test batch/candidate numerical differences bounded2mm.",
        ),
    )


if __name__ == "__main__":
    main()
