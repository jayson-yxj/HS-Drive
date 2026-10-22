"""Dense baselines on every native-matched RGB bottom, not just old eroded matches."""

import json
from pathlib import Path
import numpy as np, torch
from tools.m5l_common import rows, cache_index, write, asset
from tools.m5f_common import inputs
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5l_p4 import sample_map
from diagnostics.m5l_objects import components
from tools.m5l_object_eval import errors
from training.runtime import prepare_cuda


@torch.no_grad()
def main():
    prepare_cuda()
    torch.set_num_threads(2)
    net = encoder_load()
    out = []
    for split in ["val", "regression", "test"]:
        ix = cache_index(split)
        ar = json.loads(Path(f"runs/m5m_contact_{split}.json").read_text())["actors"]
        for row in rows(split):
            rr = [r for r in ar if r["sample_id"] == row["sample_id"] and r["detected"]]
            if not rr:
                continue
            x = inputs(row)
            c = cache_p8(net, x)
            z = np.load(ix[row["sample_id"]])
            for r in rr:
                ci = row["camera_names"].index(r["camera"])
                cc = next(
                    cc
                    for cc in components(z[f"pixel_{r['seed']}"][ci].astype(np.float32))
                    if cc["bbox"] == r["output"]["image_region"]
                )
                uv = cc["uv"]
                K = x["intrinsics"][0, ci].cpu().numpy()
                T = x["cam_to_ego"][0, ci].cpu().numpy()
                pr = sample_map(
                    c["probs"][0, ci : ci + 1],
                    torch.tensor(uv[None, None], device="cuda", dtype=torch.float32),
                    stride=8,
                    center=4,
                )[0, 0]
                pr = pr / pr.sum()
                ray = T[:3, :3] @ np.linalg.solve(K, np.r_[uv, 1.0])
                rec = {k: r[k] for k in ["sample_id", "seed", "camera", "actor_id"]}
                rec["split"] = split
                for name, dep in [
                    ("dense_expected", float((pr * net.splat.depth_centers).sum())),
                    ("dense_mode", float(net.splat.depth_centers[pr.argmax()])),
                ]:
                    point = T[:3, 3] + dep * ray
                    for ref, xyz in [
                        ("center", r["center"]),
                        ("bottom", r["native_bottom_xyz"]),
                    ]:
                        rec[name + "_" + ref] = errors(point, np.array(xyz))
                out.append(rec)
            print(split, row["sample_id"], flush=True)
    write("runs/m5m_dense_objects.json", out)
    write(
        "reports/m5m_dense_objects.json",
        dict(
            kind="actual_RTX4090_frozen_same_RGB_rays",
            records=len(out),
            raw=asset("runs/m5m_dense_objects.json"),
        ),
    )


if __name__ == "__main__":
    main()
