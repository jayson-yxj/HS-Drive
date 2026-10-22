"""Finite fresh-data frozen GPU inference and contact audit; no optimization."""

import argparse, json, time
from pathlib import Path
import torch, numpy as np
from tools.m5l_common import write, asset
from tools.m5f_common import inputs
from tools.m5i_data import load
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5l_p4 import sample_map
from tools.m5m_contact_audit import audit_rows
from tools.m5l_object_eval import errors
from diagnostics.m5l_objects import components
from training.runtime import prepare_cuda
from simulator.recorder import file_hash


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    a = p.parse_args()
    rs = json.loads(
        Path(
            f"/root/gpufree-data/hs-drive-data/processed/m5m_{a.phase}_pixels/index.json"
        ).read_text()
    )["records"]
    dest = Path(f"runs/m5m_{a.phase}_cache")
    dest.mkdir(exist_ok=False)
    prepare_cuda()
    torch.set_num_threads(2)
    cfg = json.loads(Path("configs/m5j_models.json").read_text())
    models = {r["seed"]: load(r)[0] for r in cfg["pixel"] if r["name"] == "mobile_gt"}
    net = encoder_load()
    ix = {}
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for r in rs:
        x = inputs(r)
        saved = {}
        for seed, model in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = model(**x).float().sigmoid()
            saved[f"pixel_{seed}"] = (
                (torch.clamp((q * 256).floor(), max=255) / 256)[0]
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        c = cache_p8(net, x)
        saved["depth_probability"] = c["probs"][0].cpu().numpy().astype(np.float16)
        path = dest / (r["sample_id"].replace("/", "_") + ".npz")
        np.savez_compressed(path, **saved)
        ix[r["sample_id"]] = str(path)
        print(r["sample_id"], flush=True)
    result = audit_rows(rs, ix)
    rowmap = {r["sample_id"]: r for r in rs}
    for r in result["actors"]:
        if not r["detected"]:
            continue
        row = rowmap[r["sample_id"]]
        ci = row["camera_names"].index(r["camera"])
        z = np.load(ix[r["sample_id"]])
        cc = next(
            cc
            for cc in components(z[f"pixel_{r['seed']}"][ci].astype(np.float32))
            if cc["bbox"] == r["output"]["image_region"]
        )
        x = inputs(row, device="cpu")
        K = x["intrinsics"][0, ci].numpy()
        T = x["cam_to_ego"][0, ci].numpy()
        uv = cc["uv"]
        pr = sample_map(
            torch.tensor(z["depth_probability"][ci : ci + 1].astype(np.float32)),
            torch.tensor(uv[None, None], dtype=torch.float32),
            stride=8,
            center=4,
        )[0, 0]
        pr /= pr.sum()
        centers = net.splat.depth_centers.cpu()
        ray = T[:3, :3] @ np.linalg.solve(K, np.r_[uv, 1.0])
        for name, depth in [
            ("dense_expected", float((pr * centers).sum())),
            ("dense_mode", float(centers[pr.argmax()])),
        ]:
            pt = T[:3, 3] + depth * ray
            r[name + "_center"] = errors(pt, np.array(r["center"]))
            r[name + "_bottom"] = errors(pt, np.array(r["native_bottom_xyz"]))
    write(f"runs/m5m_contact_{a.phase}.json", result)
    write(dest / "index.json", ix)
    write(
        f"reports/m5m_{a.phase}_execution.json",
        dict(
            kind="actual_RTX4090_frozen_inference_on_new_CARLA_data",
            frames=len(rs),
            positions=len({r["group"] for r in rs}),
            elapsed_s=time.perf_counter() - start,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            raw=asset(f"runs/m5m_contact_{a.phase}.json"),
            cache_index=asset(dest / "index.json"),
            source_sha256=file_hash(__file__),
        ),
    )


if __name__ == "__main__":
    main()
