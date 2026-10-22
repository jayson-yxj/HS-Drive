"""Frozen M5-N localization, exact same candidates; oracle regions explicitly diagnostic."""

import argparse, json
from pathlib import Path
import numpy as np, torch
from models.m5n_direct import DirectCenterHead, reconstruct
from datasets.m5o_benchmark import METHODS, evaluate
from tools.m5o_common import write, asset, provenance
from simulator.recorder import file_hash
from training.runtime import prepare_cuda


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--oracle-candidates", action="store_true")
    a = p.parse_args()
    suffix = "_oracle" if a.oracle_candidates else ""
    prepare_cuda()
    torch.set_num_threads(2)
    root = Path(f"runs/m5o_cache_{a.split}{suffix}")
    X = torch.load(root / "features.pt", weights_only=True)["rgb_features"]
    cc = json.loads((root / "candidates.json").read_text())
    sup = json.loads((root / "supervision.json").read_text())
    training = json.loads(Path("reports/m5n_direct_range_training.json").read_text())
    results = {}
    records = []
    checkpoints = []
    for seed in [101, 102]:
        cs = [c for c in cc if c["seed"] == seed]
        ids = [c["feature_index"] for c in cs]
        features = X[ids].cuda()
        regions = torch.tensor([c["region"] for c in cs], device="cuda")
        K = torch.tensor([c["K"] for c in cs], device="cuda")
        T = torch.tensor([c["T"] for c in cs], device="cuda")
        preds = {
            c["candidate_id"]: dict(
                candidate_id=c["candidate_id"],
                group=c["group"],
                camera=c["camera"],
                timestamp=c["timestamp"],
                positions={m: c[m] for m in METHODS[:3]},
                confidence=1,
            )
            for c in cs
        }
        for arm in ["direct", "geometry_prior"]:
            rec = next(
                r for r in training["records"] if r["seed"] == seed and r["arm"] == arm
            )["selected"]
            assert file_hash(rec["path"]) == rec["sha256"]
            checkpoints.append(rec)
            ck = torch.load(rec["path"], weights_only=True)
            model = DirectCenterHead(ck["input_dim"]).cuda().eval()
            model.load_state_dict(ck["model"])
            pred = model(features if arm == "direct" else features[:, -24:])
            xyz = reconstruct(pred, regions, K, T).cpu().numpy()
            for c, pos in zip(cs, xyz):
                preds[c["candidate_id"]]["positions"][arm] = pos.tolist()
        actors = [x for x in sup["actors"] if x["seed"] == seed]
        for method in METHODS:
            records.append(
                dict(seed=seed, method=method, **evaluate(actors, preds, method))
            )
        results.update(preds)
    out = Path(f"runs/m5o_frozen_{a.split}{suffix}")
    out.mkdir(exist_ok=False)
    write(out / "predictions.json", results)
    write(
        f"reports/m5o_frozen_{a.split}{suffix}.json",
        dict(
            kind="actual_RTX4090_frozen_M5N_offline_diagnostic",
            oracle_candidates=a.oracle_candidates,
            records=records,
            checkpoints=checkpoints,
            predictions=asset(out / "predictions.json"),
            provenance=provenance(),
            limits="Oracle candidate shape changes input distribution; frozen regressors not trained on oracle regions. Candidate gain is diagnostic, not causal upper bound or retrained performance. No model/threshold selection. Plane point remains conditional contact hypothesis scored against common BoxCenter; Dense visible-surface depth not naturally center.",
        ),
    )


if __name__ == "__main__":
    main()
