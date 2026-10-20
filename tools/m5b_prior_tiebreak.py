"""Prior change scores tie identically: apply predeclared event-IoU tie break."""

import json, time
from pathlib import Path
import numpy as np, torch
from models.m48_evidence_head import PositionPrior
from datasets.m5a_training import VisualEvidenceDataset
from tools.m5a_learning import stack, selection
from training.m48_evidence_loss import evidence_loss
from tools.m5b_learning import read_pairs
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    prepare_cuda()
    torch.set_num_threads(2)
    root = Path("runs/m5b_prior_tiebreak")
    root.mkdir(exist_ok=False)
    index = "/root/gpufree-data/hs-drive-data/processed/m5b_index"
    ds = {s: VisualEvidenceDataset(index, s) for s in ["train", "val"]}
    items = {s: [d[i] for i in range(len(d))] for s, d in ds.items()}
    labels = {s: stack(it, "targets") for s, it in items.items()}
    lookup = {r["sample_id"]: i for i, r in enumerate(ds["train"].rows)}
    pairs = read_pairs(index, "train")
    records = []
    for seed in [61, 62]:
        torch.manual_seed(seed)
        head = PositionPrior().cuda()
        opt = torch.optim.Adam(head.parameters(), lr=0.03)
        schedule = np.load(f"runs/m5b_learning/m5b_schedule_{seed}.npy")
        history = []
        best = -1
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        grad = 0
        for step in range(401):
            if step % 100 == 0:
                score, per = selection(
                    head,
                    torch.zeros(len(items["val"]), 1, device="cuda"),
                    labels["val"],
                    ds["val"].rows,
                )
                history.append(
                    dict(step=step, val_change_iou=0.0, secondary_event_iou=score)
                )
                if score > best:
                    best = score
                    selected = dict(
                        step=step,
                        val_change_iou=0.0,
                        threshold=0.025,
                        secondary_event_iou=score,
                    )
                    torch.save(
                        dict(
                            head=head.state_dict(),
                            context=None,
                            seed=seed,
                            model="A_prior",
                            selection=selected,
                        ),
                        root / f"m5b_A_prior_{seed}.pt",
                    )
            if step == 400:
                break
            r, _ = pairs[schedule[step]]
            ids = [lookup[r[k]] for k in ["a", "b"]]
            opt.zero_grad()
            pred = head(torch.zeros(2, 1, device="cuda"))
            loss, _ = evidence_loss(
                pred, {k: v[ids] for k, v in labels["train"].items()}
            )
            loss.backward()
            grad = max(grad, float(head.logits.grad.abs().max()))
            opt.step()
        ck = root / f"m5b_A_prior_{seed}.pt"
        records.append(
            dict(
                name="A_prior",
                seed=seed,
                steps=400,
                selected=selected,
                history=history,
                elapsed_s=time.perf_counter() - start,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                trainable_parameters=384000,
                finite_gradients=bool(np.isfinite(grad)),
                checkpoint=str(ck),
                sha256=file_hash(ck),
                correction="Same400step replay, same sample order, secondary val event IoU resolves inevitable all-zero change-score tie. Original run retained.",
            )
        )
    original = json.loads(Path("reports/m5b_context_ablation.json").read_text())
    original["initial_prior_records"] = [
        r for r in original["records"] if r["name"] == "A_prior"
    ]
    original["records"] = records + [
        r for r in original["records"] if r["name"] != "A_prior"
    ]
    original["prior_correction_source_sha256"] = file_hash(__file__)
    atomic_json("reports/m5b_context_ablation_corrected.json", original)


if __name__ == "__main__":
    main()
