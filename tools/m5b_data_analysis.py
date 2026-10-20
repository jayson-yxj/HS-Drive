"""Training/validation support audit. No test model inspection or rule tuning."""

import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    root = Path("/root/gpufree-data/hs-drive-data/processed/m5b_index")
    index = json.loads((root / "index.json").read_text())
    pairs = json.loads((root / "m5b_pairs.json").read_text())
    records = []
    x = -20 + (np.arange(160) + 0.5) * 0.5
    y = -30 + (np.arange(120) + 0.5) * 0.5
    dist = np.hypot(x[:, None], y[None, :])
    for r in pairs["pairs"]:
        with np.load(r["path"]) as z:
            change = z["appeared"] | z["vanished"]
            mixed = z["witness_conflict"]
        records.append(
            dict(
                group=r["group"],
                split=r["split"],
                a=r["a_configuration"],
                b=r["b_configuration"],
                n=int(change.sum()),
                by_layer=change.sum((1, 2)).tolist(),
                by_distance={
                    f"{lo}_{hi}": int(
                        (change & ((dist >= lo) & (dist < hi))[None]).sum()
                    )
                    for lo, hi in [(0, 7.5), (7.5, 12.5), (12.5, 20), (20, 35)]
                },
                witness_conflicts=int(mixed.sum()),
            )
        )
    atomic_json(
        "reports/m5b_pair_quality.json",
        dict(
            kind="offline_train_validation_only",
            records=records,
            independent_groups={
                s: len(set(r["group"] for r in records if r["split"] == s))
                for s in ["train", "val"]
            },
            no_volume_free_negative=True,
            recipe_sha256=file_hash("datasets/m5b_pairs.py"),
            index_sha256=file_hash(root / "index.json"),
            protocol_sha256=file_hash("configs/m5b_protocol.json"),
        ),
    )


if __name__ == "__main__":
    main()
