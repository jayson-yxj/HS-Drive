"""Reuse M5-A V2 indexing then add independent counterfactual witness sidecars."""

import argparse, json, sys
from pathlib import Path
import numpy as np
from tools import m5a_dataset
from datasets.m5b_pairs import pair_witness
from simulator.recorder import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    a = p.parse_args()
    # Reuse unchanged original label implementation, output only M5-C assets.
    sys.argv = [
        "m5a_dataset",
        "--raw",
        *a.raw,
        "--output",
        a.output,
        "--report",
        a.report,
    ]
    m5a_dataset.main()
    root = Path(a.output)
    index = json.loads((root / "index.json").read_text())
    pairs = []
    quality = []
    for group in sorted(set(r["group"] for r in index["samples"])):
        rows = sorted(
            [r for r in index["samples"] if r["group"] == group],
            key=lambda x: x["configuration"],
        )
        if rows[0]["split"] not in a.splits:
            continue
        raw = Path(rows[0]["raw_frame"]).parent.parent
        man = json.loads((raw / "manifest.json").read_text())
        frames = [json.loads(Path(r["raw_frame"]).read_text()) for r in rows]
        labels = []
        for r in rows:
            with np.load(r["label"]) as z:
                labels.append({k: z[k] for k in z.files})
        cache = {}
        for ai, bi in (
            [(0, i) for i in range(1, len(rows) - 1)]
            + [(i, len(rows) - 1) for i in range(1, len(rows) - 1)]
            + [(7, 8), (0, len(rows) - 1)]
        ):
            pair, detail = pair_witness(
                raw, man, frames[ai], frames[bi], labels[ai], labels[bi], cache
            )
            path = root / group / f"m5c_pair_{ai}_{bi}.npz"
            np.savez_compressed(path, **pair)
            record = dict(
                group=group,
                split=rows[0]["split"],
                a=rows[ai]["sample_id"],
                b=rows[bi]["sample_id"],
                a_configuration=ai,
                b_configuration=bi,
                path=str(path),
                sha256=file_hash(path),
                appeared=int(pair["appeared"].sum()),
                vanished=int(pair["vanished"].sum()),
                stable=int(pair["stable"].sum()),
                detail=detail,
            )
            pairs.append(record)
            print(group, ai, bi, record["appeared"], record["vanished"], flush=True)
        quality.append(
            dict(
                group=group,
                split=rows[0]["split"],
                pairs=len(pairs),
                empty_return_configuration=len(rows) - 1,
            )
        )
    atomic_json(
        root / "m5c_pairs.json",
        dict(
            pairs=pairs,
            contract="Visible same-ray obstacle point presence/absence; not voxel free. Count>=3 and opposite obstacle veto. No inference inputs.",
        ),
    )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["protocol_sha256"] = file_hash("configs/m5c_protocol.json")
    atomic_json(root / "manifest.json", manifest)
    manifest["pairs_sha256"] = file_hash(root / "m5c_pairs.json")
    atomic_json(root / "m5c_manifest.json", manifest)
    atomic_json(
        root / "m5c_quality.json",
        dict(
            kind="offline_actual_CARLA_pair_witnesses",
            groups=quality,
            counts={
                s: dict(
                    pairs=sum(p["split"] == s for p in pairs),
                    appeared=sum(p["appeared"] for p in pairs if p["split"] == s),
                    vanished=sum(p["vanished"] for p in pairs if p["split"] == s),
                )
                for s in a.splits
            },
            source_sha256={
                p: file_hash(p) for p in [__file__, "datasets/m5b_pairs.py"]
            },
            sealed_test="Pair diagnostics restricted to selected splits; no model/threshold test evaluation.",
        ),
    )


if __name__ == "__main__":
    main()
