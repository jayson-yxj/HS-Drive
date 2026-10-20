"""Generate paired sidecars for reserved shifted test only after decision freeze."""

import json
from pathlib import Path
import numpy as np
from datasets.m5b_pairs import pair_witness
from simulator.recorder import atomic_json, file_hash


def main():
    decision = json.loads(Path("reports/m5b_pretest_decision.json").read_text())
    if decision["test_opened"]:
        raise ValueError("Test already open")
    root = Path("/root/gpufree-data/hs-drive-data/processed/m5b_sealed_index")
    idx = json.loads((root / "index.json").read_text())
    pairs = []
    for group in sorted(set(r["group"] for r in idx["samples"])):
        rows = sorted(
            [r for r in idx["samples"] if r["group"] == group],
            key=lambda x: x["configuration"],
        )
        raw = Path(rows[0]["raw_frame"]).parent.parent
        man = json.loads((raw / "manifest.json").read_text())
        frames = [json.loads(Path(r["raw_frame"]).read_text()) for r in rows]
        labels = []
        for r in rows:
            with np.load(r["label"]) as z:
                labels.append({k: z[k] for k in z.files})
        cache = {}
        for ai, bi in (
            [(0, i) for i in range(1, 11)]
            + [(i, 11) for i in range(1, 11)]
            + [(5, 6), (0, 11)]
        ):
            d, info = pair_witness(
                raw, man, frames[ai], frames[bi], labels[ai], labels[bi], cache
            )
            path = root / group / f"m5b_pair_{ai}_{bi}.npz"
            np.savez_compressed(path, **d)
            pairs.append(
                dict(
                    group=group,
                    split="test",
                    a=rows[ai]["sample_id"],
                    b=rows[bi]["sample_id"],
                    a_configuration=ai,
                    b_configuration=bi,
                    path=str(path),
                    sha256=file_hash(path),
                    appeared=int(d["appeared"].sum()),
                    vanished=int(d["vanished"].sum()),
                    stable=int(d["stable"].sum()),
                )
            )
            print(group, ai, bi, flush=True)
    atomic_json(
        root / "m5b_pairs.json",
        dict(
            pairs=pairs,
            decision_sha256=file_hash("reports/m5b_pretest_decision.json"),
            recipe_sha256=file_hash("datasets/m5b_pairs.py"),
        ),
    )
    atomic_json(
        "reports/m5b_test_opening.json",
        dict(
            kind="frozen_test_opening",
            decision_sha256=file_hash("reports/m5b_pretest_decision.json"),
            index_sha256=file_hash(root / "index.json"),
            pair_sha256=file_hash(root / "m5b_pairs.json"),
            source_sha256=file_hash(__file__),
            rules_selected_on_train_pilot_only=True,
        ),
    )


if __name__ == "__main__":
    main()
