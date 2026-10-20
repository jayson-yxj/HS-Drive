"""Oracle-only dynamic/occlusion support audit on actual M4.6 sequences."""

import argparse, json
from pathlib import Path
import numpy as np, torch
from geometry.temporal_visibility import align_visible_history
from geometry.temporal_warp import warp_bev
from simulator.recorder import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/m46_temporal_validation.json")
    p.add_argument("--source", default="reports/m46_occupancy_validation.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    report = json.loads(Path(a.source).read_text())
    records = []
    for scene in sorted(set(r["scene"] for r in report["records"])):
        rows = [r for r in report["records"] if r["scene"] == scene]
        left, right = rows[0], rows[-1]
        arrays = []
        for row in (left, right):
            with np.load(row["path"]) as d:
                arrays.append({k: d[k] for k in d.files})
        a0, b0 = arrays
        # Surface evidence serves as diagnostic features, not categorical fusion.
        features = torch.tensor(a0["occupancy_state"] == 2, dtype=torch.float32)[None]
        # Treat height layers as independent batch items so an observation at
        # one height can never validate occluded support at another height.
        features = features.reshape(4, 1, 160, 120)
        pv = torch.tensor(a0["occupancy_valid"])[:, None]
        cv = torch.tensor(b0["occupancy_valid"])[:, None]
        pd = torch.tensor(a0["surface_semantic"] == 3)[:, None]
        cd = torch.tensor(b0["surface_semantic"] == 3)[:, None]
        pp = torch.tensor(left["pose"], dtype=torch.float32)[None].expand(4, -1, -1)
        cp = torch.tensor(right["pose"], dtype=torch.float32)[None].expand(4, -1, -1)
        naive, nv, _ = warp_bev(features, pp, cp, previous_valid=pv)
        gated, gv, _ = align_visible_history(
            features, pp, cp, pv, cv, previous_dynamic=pd, current_dynamic=cd
        )
        records.append(
            dict(
                scene=scene,
                frames=[left["frame_id"], right["frame_id"]],
                dt_s=right["timestamp_s"] - left["timestamp_s"],
                naive_valid_cells=int(nv.sum()),
                safe_valid_cells=int(gv.sum()),
                previous_dynamic_cells=int(pd.sum()),
                current_dynamic_cells=int(cd.sum()),
                naive_history_in_current_unknown=int((nv & ~cv).sum()),
                safe_history_in_current_unknown=int((gv & ~cv).sum()),
                safe_history_in_current_dynamic=int((gv & cd).sum()),
                nonzero_outside_mask=int(
                    torch.count_nonzero(gated[~gv.expand_as(gated)])
                ),
            )
        )
    atomic_json(
        a.output,
        dict(
            schema_version="hsd-m46-temporal-visibility-v1",
            records=records,
            passed=all(
                r["safe_history_in_current_unknown"]
                == r["safe_history_in_current_dynamic"]
                == r["nonzero_outside_mask"]
                == 0
                for r in records
            ),
            limitations=[
                "Oracle semantic potentially movable classes used as conservative dynamic mask; not a deployed visual motion estimator",
                "New actual recordings have fixed ego and moved actor; actual ego-motion geometry remains covered by M45 and regression tests",
                "Current unknown support rejected separately per height, not promoted to Free. No categorical fusion performed.",
                "Actor positions are scripted discrete 2m translations between frames, not physics-based motion performance; used only to test mask/occlusion rejection.",
                "No ConvGRU or learned temporal training",
            ],
        ),
    )


if __name__ == "__main__":
    main()
