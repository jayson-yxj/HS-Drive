"""Same-renderer held-out pixels: measurement consistency, not independent mesh GT."""

import json
from pathlib import Path
import numpy as np
from tools.m48_spatial import frame_rays
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json, file_hash


def main():
    output = Path("reports/m5a_spatial_contract_v2.json")
    if output.exists():
        raise FileExistsError(output)
    index = json.loads(
        Path(
            "/root/gpufree-data/hs-drive-data/processed/m5a_index/index.json"
        ).read_text()
    )
    records = []
    g = VoxelGrid()
    for r in index["samples"]:
        if r["configuration"] not in [0, 1, 7]:
            continue
        root = Path(r["raw_frame"]).parent.parent
        man = json.loads((root / "manifest.json").read_text())
        fid = int(Path(r["raw_frame"]).stem)
        local = {**man, "counterfactual_frame": man["references"][str(fid)]}
        rays, box = frame_rays(root, local, fid)
        uv = rays["uv"]
        held = (uv[:, 0] + 2 * uv[:, 1]) % 4 == 1
        p = rays["endpoints"][held]
        sem = rays["semantics"][held]
        ix = np.floor((p[:, 0] - g.x[0]) / g.resolution).astype(int)
        iy = np.floor((p[:, 1] - g.y[0]) / g.resolution).astype(int)
        iz = np.searchsorted(g.z, p[:, 2], side="right") - 1
        inside = (ix >= 0) & (ix < 160) & (iy >= 0) & (iy < 120) & (iz >= 0) & (iz < 4)
        ix, iy, iz, sem, p = [x[inside] for x in [ix, iy, iz, sem, p]]
        with np.load(r["label"]) as z:
            d = {k: z[k] for k in z.files}
        rec = dict(
            sample_id=r["sample_id"],
            split=r["split"],
            configuration=r["configuration"],
            heldout_valid_pixels=len(p),
            tasks={},
        )
        for name, code, ch in [("ground", 1, 0), ("obstacle", 2, 1)]:
            selected = (sem == 1) if name == "ground" else np.isin(sem, [2, 3])
            cells = np.unique(
                np.ravel_multi_index(
                    (iz[selected], ix[selected], iy[selected]), (4, 160, 120)
                )
            )
            target = np.zeros(76800, bool)
            target[cells] = 1
            target = target.reshape(4, 160, 120)
            hit = d["surface_semantic"][ch] & d["surface_semantic_valid"]
            rec["tasks"][name] = dict(
                reference_cells=len(cells),
                recalled=int((hit & target).sum()),
                recall=float((hit & target).sum() / max(1, len(cells))),
                by_layer=[],
            )
            for l in range(4):
                rec["tasks"][name]["by_layer"].append(
                    dict(
                        layer=l,
                        n=int(target[l].sum()),
                        recalled=int((target[l] & hit[l]).sum()),
                    )
                )
        rec["coverage"] = float(d["observability"].mean())
        rec["mixed_fraction_observed"] = float(
            d["mixed_cell_mask"].sum() / max(1, d["observability"].sum())
        )
        rec["unknown"] = float((~d["observability"]).mean())
        rec["other_positives"] = int(
            (d["surface_semantic"][2] & d["surface_semantic_valid"]).sum()
        )
        records.append(rec)
        print(r["sample_id"], rec["tasks"], flush=True)
    atomic_json(
        output,
        dict(
            kind="offline_same_renderer_held_pixel_measurement_consistency",
            records=records,
            reference="Same CARLA renderer, withheld interleaved pixels. Not independent 3D truth. No mesh certification.",
            event_contract="Observability=S OR F denotes recorded events, not completely observed physical voxel. Negative event means absent from fixed recipe only.",
            physical_obstacle_contract="Obstacle positives supported by visible semantic surfaces; obstacle negatives only reliable ground/known nonobstacle surface in obstacle_valid. No volume-free negative and no safety traversability output.",
            other="Report sample counts; zero-positive groups cannot establish Other learnability.",
            nearfield_dynamic_limits="Scripted static actor layouts, no moving-object temporal supervision; ego removal is oracle label construction only.",
            unchanged_recipe_sha256=file_hash("datasets/m48_evidence.py"),
        ),
    )


if __name__ == "__main__":
    main()
