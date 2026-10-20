"""Create separate heldout-pixel evaluation sidecars without changing V2 labels."""

import argparse, json
from pathlib import Path
import numpy as np
from datasets.m5c_surface import heldout_surface
from simulator.recorder import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    a = p.parse_args()
    root = Path(a.index)
    dest = root / "m5c_surface_index.json"
    if dest.exists():
        raise FileExistsError(dest)
    rows = json.loads((root / "index.json").read_text())["samples"]
    out = []
    for row in rows:
        raw = Path(row["raw_frame"]).parent.parent
        man = json.loads((raw / "manifest.json").read_text())
        frame = json.loads(Path(row["raw_frame"]).read_text())
        with np.load(row["label"]) as z:
            lab = {k: z[k] for k in z.files}
        data, actors = heldout_surface(raw, man, frame, row["camera_names"], lab)
        path = root / row["group"] / f"m5c_surface_{frame['frame_id']}.npz"
        np.savez_compressed(path, **data)
        out.append(
            dict(
                sample_id=row["sample_id"],
                group=row["group"],
                split=row["split"],
                configuration=row["configuration"],
                path=str(path),
                sha256=file_hash(path),
                actors=actors,
                actor_visible_cells=data["actor_masks"].sum((1, 2, 3)).tolist(),
                positive=int((data["target"] & data["valid"]).sum()),
                negative=int((~data["target"] & data["valid"]).sum()),
                mixed=int((data["mixed"] & data["valid"]).sum()),
                weather=man["weather"],
                geometry=man["geometry"],
            )
        )
        print(
            row["sample_id"],
            out[-1]["positive"],
            out[-1]["actor_visible_cells"],
            flush=True,
        )
    atomic_json(
        dest,
        dict(
            records=out,
            source_sha256={
                f: file_hash(f) for f in [__file__, "datasets/m5c_surface.py"]
            },
            reference="same renderer heldout pixels; not independent mesh truth; AABB gates first-hit surface points, no box volume positives",
        ),
    )


if __name__ == "__main__":
    main()
