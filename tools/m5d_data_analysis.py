"""Group-level support audit, preserving zero-support actors and strata."""

import argparse, json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash
from datasets.m5a_training import VisualEvidenceDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indices", nargs="+", required=True)
    p.add_argument("--report", default="reports/m5d_dataset_quality.json")
    a = p.parse_args()
    out = Path(a.report)
    if out.exists():
        raise FileExistsError(out)
    records = []
    pairs = []
    inputs = []
    source = []
    x = -20 + (np.arange(160) + 0.5) * 0.5
    y = -30 + (np.arange(120) + 0.5) * 0.5
    distance = np.hypot(x[:, None], y[None, :])
    bands = [(0, 10), (10, 15), (15, 20), (20, 35), (35, 80)]
    for index in a.indices:
        root = Path(index)
        surf = json.loads((root / "m5d_surface_index.json").read_text())["records"]
        lookup = {r["sample_id"]: r for r in surf}
        source.append(
            dict(
                index=index,
                sha256=file_hash(root / "index.json"),
                surface_sha256=file_hash(root / "m5d_surface_index.json"),
            )
        )
        for split in ["train", "val", "test"]:
            ds = VisualEvidenceDataset(index, split)
            for i, row in enumerate(ds.rows):
                r = lookup[row["sample_id"]]
                item = ds[i]
                lab = {k: v.numpy() for k, v in item.targets.items()}
                with np.load(r["path"]) as z:
                    d = {k: z[k] for k in z.files}
                actor = d["actor_masks"].any(0)
                sem = lab["surface_semantic"]
                safe = (
                    lab["obstacle_valid"]
                    & lab["surface_semantic_valid"]
                    & ~lab["ego_mask"]
                    & ~lab["evidence_conflict_mask"]
                    & ~lab["semantic_conflict_mask"]
                )
                pos = safe & sem[1] & ~sem[0] & ~sem[2]
                neg = safe & sem[0] & ~sem[1] & ~sem[2]
                records.append(
                    dict(
                        sample_id=row["sample_id"],
                        group=row["group"],
                        split=split,
                        configuration=row["configuration"],
                        weather=r["weather"],
                        geometry=r["geometry"],
                        valid=int(d["valid"].sum()),
                        positive=int((d["valid"] & d["target"]).sum()),
                        negative=int((d["valid"] & ~d["target"]).sum()),
                        mixed=int((d["valid"] & d["mixed"]).sum()),
                        direct_train_positive=int(pos.sum()),
                        direct_train_negative=int(neg.sum()),
                        unknown_fraction=float((~d["valid"]).mean()),
                        actors=[
                            dict(**x, heldout_surface_cells=int(n), eligible=n >= 3)
                            for x, n in zip(r["actors"], r["actor_visible_cells"])
                        ],
                        by_distance={
                            f"{lo}_{hi}": dict(
                                obstacle=int(
                                    (
                                        d["target"]
                                        & d["valid"]
                                        & ((distance >= lo) & (distance < hi))[None]
                                    ).sum()
                                ),
                                target_actor=int(
                                    (
                                        actor
                                        & d["valid"]
                                        & ((distance >= lo) & (distance < hi))[None]
                                    ).sum()
                                ),
                                direct_train_positive=int(
                                    (
                                        pos & ((distance >= lo) & (distance < hi))[None]
                                    ).sum()
                                ),
                            )
                            for lo, hi in bands
                        },
                        by_layer={
                            str(z): dict(
                                target_actor=int((actor & d["valid"])[z].sum()),
                                obstacle=int((d["target"] & d["valid"])[z].sum()),
                            )
                            for z in range(4)
                        },
                    )
                )
                assert not set(item.inputs) & {
                    "depth",
                    "instance",
                    "actors",
                    "target",
                    "valid",
                    "label",
                }
                inputs.append(list(item.inputs))
        for r in json.loads((root / "m5d_pairs.json").read_text())["pairs"]:
            with np.load(r["path"]) as z:
                change = z["appeared"] | z["vanished"]
                pairs.append(
                    dict(
                        group=r["group"],
                        split=r["split"],
                        a=r["a_configuration"],
                        b=r["b_configuration"],
                        n=int(change.sum()),
                        by_distance={
                            f"{lo}_{hi}": int(
                                (
                                    change & ((distance >= lo) & (distance < hi))[None]
                                ).sum()
                            )
                            for lo, hi in bands
                        },
                    )
                )
    summary = {}
    for split in ["train", "val", "test"]:
        rows = [r for r in records if r["split"] == split]
        pr = [r for r in pairs if r["split"] == split]
        actors = [a for r in rows for a in r["actors"]]
        summary[split] = dict(
            frames=len(rows),
            groups=sorted(set(r["group"] for r in rows)),
            direct_train_positive=sum(r["direct_train_positive"] for r in rows),
            direct_train_negative=sum(r["direct_train_negative"] for r in rows),
            placed_actors=len(actors),
            zero_support_actors=sum(a["heldout_surface_cells"] == 0 for a in actors),
            eligible_actors=sum(a["eligible"] for a in actors),
            pairs=len(pr),
            pair_events=sum(r["n"] for r in pr),
            by_distance={
                d: dict(
                    target_actor_cells=sum(
                        r["by_distance"][d]["target_actor"] for r in rows
                    ),
                    groups_with_target_support=sorted(
                        set(
                            r["group"]
                            for r in rows
                            if r["by_distance"][d]["target_actor"] > 0
                        )
                    ),
                    direct_train_positive=sum(
                        r["by_distance"][d]["direct_train_positive"] for r in rows
                    ),
                    pair_events=sum(r["by_distance"][d] for r in pr),
                )
                for d in [f"{lo}_{hi}" for lo, hi in bands]
            },
        )
    atomic_json(
        out,
        dict(
            kind="offline_audit_actual_CARLA_captures_no_model_selection",
            summary=summary,
            records=records,
            pairs=pairs,
            indices=source,
            source_sha256=file_hash(__file__),
            input_allowlist=sorted(set(k for keys in inputs for k in keys)),
            reference="Reliable same-renderer first hits with held-out pixels. AABB only surface association, not Mesh. Static physical states; synthetic raised-pedestrian stress. Cells/configurations not independent trials. Zero-support actors retained; no lowering masks.",
            train_gate=summary["train"]["direct_train_positive"] > 0
            and summary["train"]["direct_train_negative"] > 0,
            protocol_sha256=file_hash("configs/m5d_protocol.json"),
        ),
    )


if __name__ == "__main__":
    main()
