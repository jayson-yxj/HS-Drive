"""M5C gradient and label support audit. No test-dependent model decisions."""

import json
from pathlib import Path
import numpy as np, torch
from models.m5c_obstacle_head import VisibleObstacleHead
from training.m5b_change_loss import change_loss
from training.m5c_obstacle_loss import visible_obstacle_loss
from datasets.m5a_training import VisualEvidenceDataset
from simulator.recorder import atomic_json, file_hash
from training.runtime import prepare_cuda


def main():
    dest = Path("reports/m5d_supervision_audit.json")
    if dest.exists():
        raise FileExistsError(dest)
    prepare_cuda()
    torch.set_num_threads(2)
    torch.manual_seed(81)
    head = VisibleObstacleHead().cuda()
    out = head(torch.randn(2, 128, 4, 4, device="cuda"))
    blank = torch.zeros(4, 4, 4, dtype=torch.bool, device="cuda")
    pair = {k: blank.clone() for k in ["appeared", "vanished", "stable"]}
    pair["appeared"][1, 1, 1] = True
    pair["vanished"][2, 2, 2] = True
    pair["stable"][1, 2, 1] = True
    loss, _ = change_loss(out, pair)
    loss.backward()
    grads = {
        n: None if p.grad is None else float(p.grad.abs().max())
        for n, p in head.named_parameters()
    }
    records = []
    events = []
    root = Path("/root/gpufree-data/hs-drive-data/processed/m5c_index")
    lookup = {}
    labels = {}
    surface = {
        r["sample_id"]: r
        for r in json.loads((root / "m5c_surface_index.json").read_text())["records"]
    }
    xx = -20 + (np.arange(160) + 0.5) * 0.5
    yy = -30 + (np.arange(120) + 0.5) * 0.5
    distance = np.hypot(xx[:, None], yy[None, :])
    bands = [(0, 10), (10, 15), (15, 20), (20, 35), (35, 80)]
    for split in ["train", "val"]:
        ds = VisualEvidenceDataset(root, split)
        for i, row in enumerate(ds.rows):
            d = {k: v.numpy() for k, v in ds[i].targets.items()}
            labels[row["sample_id"]] = d
            lookup[row["sample_id"]] = row
            s = d["surface_semantic"]
            safe = (
                d["obstacle_valid"]
                & d["surface_semantic_valid"]
                & ~d["ego_mask"]
                & ~d["evidence_conflict_mask"]
                & ~d["semantic_conflict_mask"]
            )
            pos = safe & s[1] & ~s[0] & ~s[2]
            neg = safe & s[0] & ~s[1] & ~s[2]
            with np.load(surface[row["sample_id"]]["path"]) as z:
                actor = z["actor_masks"].any(0)
            # Actor association on heldout support is a lower-bound overlap, not full native actor segmentation.
            masks = dict(
                positive=pos,
                negative=neg,
                mixed_positive=pos & d["mixed_cell_mask"],
                mixed_negative=neg & d["mixed_cell_mask"],
                unknown=(pos | neg) & ~d["observability"],
                placed_actor_overlap=pos & actor,
                background_or_unassociated=pos & ~actor,
            )
            records.append(
                dict(
                    sample_id=row["sample_id"],
                    group=row["group"],
                    split=split,
                    counts={k: int(v.sum()) for k, v in masks.items()},
                    by_layer={
                        str(z): {k: int(v[z].sum()) for k, v in masks.items()}
                        for z in range(4)
                    },
                    by_distance={
                        f"{lo}_{hi}": {
                            k: int(
                                (v & ((distance >= lo) & (distance < hi))[None]).sum()
                            )
                            for k, v in masks.items()
                        }
                        for lo, hi in bands
                    },
                )
            )
    for row in json.loads((root / "m5c_pairs.json").read_text())["pairs"]:
        with np.load(row["path"]) as z:
            p = {k: z[k] for k in z.files}
        for direction, own, opp in [("appeared", "b", "a"), ("vanished", "a", "b")]:
            d = labels[row[opp]]
            sem = d["surface_semantic"]
            m = p[direction]
            safe = (
                d["obstacle_valid"]
                & d["surface_semantic_valid"]
                & ~d["evidence_conflict_mask"]
                & ~d["semantic_conflict_mask"]
                & ~d["ego_mask"]
            )
            neg = safe & sem[0] & ~sem[1] & ~sem[2]
            masks = dict(
                witness=m,
                direct_negative=m & neg,
                other_obstacle=m & sem[1],
                background_ground=m & sem[0],
                no_surface_endpoint=m & ~d["surface_evidence"],
                unknown=m & ~d["observability"],
                mixed=m & d["mixed_cell_mask"],
                opposite_conflict=m
                & (d["evidence_conflict_mask"] | d["semantic_conflict_mask"]),
            )
            events.append(
                dict(
                    group=row["group"],
                    split=row["split"],
                    direction=direction,
                    a=row["a_configuration"],
                    b=row["b_configuration"],
                    counts={k: int(v.sum()) for k, v in masks.items()},
                    by_layer={
                        str(z): {k: int(v[z].sum()) for k, v in masks.items()}
                        for z in range(4)
                    },
                    by_distance={
                        f"{lo}_{hi}": {
                            k: int(
                                (v & ((distance >= lo) & (distance < hi))[None]).sum()
                            )
                            for k, v in masks.items()
                        }
                        for lo, hi in bands
                    },
                )
            )
    atomic_json(
        dest,
        dict(
            kind="actual_RTX4090_gradient_probe_and_offline_train_val_label_audit",
            gradient_probe=grads,
            direct_pair_gradient_absent=all(
                grads[k] is None for k in grads if k.startswith("visible_obstacle.")
            ),
            records=records,
            events=events,
            summary=dict(
                surface={
                    k: sum(r["counts"][k] for r in records)
                    for k in records[0]["counts"]
                },
                absence={
                    k: sum(r["counts"][k] for r in events) for k in events[0]["counts"]
                },
            ),
            conclusion="M5B pair loss reads surface and semantic only: direct output layer has grad=None. Shared trunk can change indirectly. Direct BCE only reliable obstacle/ground; most absence witnesses have no direct negative. Missing endpoint is not absence.",
            witness_gap="Legacy opposite ray stable+finite may include depth>=80 and lacks opposite conflict veto; require reliable in-range first-hit and both-side semantic/evidence veto, plus dense reliable obstacle veto before direct negatives.",
            actor_scope="Placed Actor counts here overlap heldout rendered AABB-associated cells: lower-bound, remainder background OR unassociated, never claimed exact background category.",
            test_used=False,
            source_sha256={
                p: file_hash(p)
                for p in [
                    __file__,
                    "training/m5c_obstacle_loss.py",
                    "training/m5b_change_loss.py",
                    "models/m5c_obstacle_head.py",
                    "datasets/m5b_pairs.py",
                ]
            },
        ),
    )


if __name__ == "__main__":
    main()
