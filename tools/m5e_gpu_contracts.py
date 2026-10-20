"""Actual CUDA singleframe-vs-cache verification with frozen checkpoints."""

import json
from pathlib import Path
import numpy as np
import torch
from datasets.m5a_training import VisualEvidenceDataset
from tools.m5a_learning import encoder_load, stack
from tools.m5e_common import load_checkpoint
from models.m5e_paths import depth_logits
from models.m5c_obstacle_head import VisibleObstacleHead
from models.m5d_inference import VisibleExistenceModel
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb, cached_forward
from simulator.recorder import atomic_json, file_hash
from training.runtime import prepare_cuda


def main():
    dest = Path("reports/m5e_gpu_contracts.json")
    if dest.exists():
        raise FileExistsError(dest)
    prepare_cuda()
    torch.set_num_threads(2)
    ds = VisualEvidenceDataset("/root/gpufree-data/hs-drive-data/processed/m5e_index", "val", supervision=False)
    results = []
    study = json.loads(Path("reports/m5e_model_ablation.json").read_text())
    for r in study["records"]:
        if r["name"] == "P_prior":
            continue
        net = encoder_load()
        ck = load_checkpoint(r["checkpoint"])
        net.context_head.load_state_dict(ck["context"])
        net.depth_head.load_state_dict(ck["depth"])
        head = VisibleObstacleHead().cuda().eval()
        head.load_state_dict(ck["head"])
        model = VisibleExistenceModel(net, head, r["stride"]).eval()
        inp = stack([ds[3]], "inputs")
        fn = cache_p8 if r["stride"] == 8 else cache_rgb
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            direct = model(**inp)["visible_obstacle"].float().sigmoid()
            c = fn(net, inp)
            cached = head(cached_forward(net, c, probabilities=depth_logits(net,c).float().softmax(2)))["visible_obstacle"].float().sigmoid()
        rejected = False
        try:
            model(**inp, witness_mask=torch.zeros_like(direct))
        except (ValueError, TypeError, KeyError):
            rejected = True
        th = ck["selection"]["threshold"]
        diff = (direct-cached).abs()
        results.append(dict(name=r["name"], seed=r["seed"], stride=r["stride"], shape=list(direct.shape), finite=bool(torch.isfinite(direct).all()), max_score_difference=float(diff.max()), mean_score_difference=float(diff.mean()), threshold_flips=int(((direct>th)!=(cached>th)).sum()), privileged_input_rejected=rejected, input_keys=list(inp)))
    atomic_json(dest, dict(kind="actual_RTX4090_BF16_singleframe_contracts", records=results, source_sha256=file_hash(__file__), wrapper_sha256=file_hash("models/m5d_inference.py"), passed=all(r["finite"] and r["privileged_input_rejected"] and r["max_score_difference"]<.02 for r in results), limits="BF16 accumulation ordering may cause near-threshold differences; not bitwise equivalence or driving validation."))


if __name__ == "__main__":
    main()
