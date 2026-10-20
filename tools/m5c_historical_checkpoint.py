"""Untouched M5B C_pair checkpoints on new VALIDATION only; fixed diagnostic threshold."""

import json
from pathlib import Path
import numpy as np, torch
from datasets.m5a_training import VisualEvidenceDataset
from tools.m5a_learning import encoder_load, stack
from tools.m5c_learning import read_pairs
from models.m48_evidence_head import EvidenceHead
from models.m5b_cached_encoder import cache_rgb, cached_forward
from training.m5b_change_loss import obstacle_score
from diagnostics.m5b_metrics import change_metrics, aggregate
from diagnostics.m5a_metrics import binary_metrics
from tools.m5c_evaluate import sum_metrics
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json, file_hash


def main():
    out = Path("reports/m5c_historical_checkpoint_validation.json")
    if out.exists():
        raise FileExistsError(out)
    prepare_cuda()
    torch.set_num_threads(2)
    index = Path("/root/gpufree-data/hs-drive-data/processed/m5c_index")
    ds = VisualEvidenceDataset(index, "val", supervision=False)
    lookup = {r["sample_id"]: i for i, r in enumerate(ds.rows)}
    pairs = read_pairs(index, "val")
    refs = {
        r["sample_id"]: r
        for r in json.loads((index / "m5c_surface_index.json").read_text())["records"]
    }
    net = encoder_load()
    caches = [cache_rgb(net, stack([ds[i]], "inputs")) for i in range(len(ds))]
    results = []
    for seed in [61, 62]:
        path = Path(f"runs/m5b_learning/m5b_C_pair_{seed}.pt")
        ck = torch.load(path, weights_only=True)
        net.context_head.load_state_dict(ck["context"])
        head = EvidenceHead().cuda().eval()
        head.load_state_dict(ck["head"])
        pred = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for c in caches:
                pred.append(
                    obstacle_score(head(cached_forward(net, c)))[0].cpu().numpy()
                )
        pred = np.array(pred)
        absolute = []
        actors = []
        for i, r in enumerate(ds.rows):
            with np.load(refs[r["sample_id"]]["path"]) as z:
                absolute.append(binary_metrics(pred[i], z["target"], z["valid"], 0.5))
                actors.append(
                    binary_metrics(
                        pred[i], z["target"], z["valid"] & z["actor_masks"].any(0), 0.5
                    )
                )
        changes = [
            dict(
                group=r["group"],
                metrics=change_metrics(
                    pred[lookup[r["a"]]],
                    pred[lookup[r["b"]]],
                    d,
                    ck["selection"]["threshold"],
                ),
            )
            for r, d in pairs
        ]
        results.append(
            dict(
                seed=seed,
                checkpoint=str(path),
                sha256=file_hash(path),
                absolute_fixed_half=sum_metrics(absolute),
                actor_surface_fixed_half=sum_metrics(actors),
                paired=aggregate(changes),
                original_change_threshold=ck["selection"]["threshold"],
            )
        )
    atomic_json(
        out,
        dict(
            kind="actual_GPU_historical_checkpoint_new_validation_regression_no_retraining",
            records=results,
            source_sha256=file_hash(__file__),
            scope="Old C_pair uses its original M5B validation-selected change threshold; absolute .5 is fixed diagnostic only, not a fair selected-threshold competitor. Main matched study retrains all methods on new data. New test not used.",
        ),
    )


if __name__ == "__main__":
    main()
