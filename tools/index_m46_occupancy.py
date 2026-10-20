"""Normalize small M4.6 records into RGB-only inputs and separate supervision."""

import argparse, json
from pathlib import Path
from datasets.schema import CAMERA_IDS
from simulator.recorder import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="/root/gpufree-data/hs-drive-data/processed/m46_training_index",
    )
    p.add_argument("--labels", help="Optional separately audited dense label report")
    p.add_argument("--report", default="reports/m46_dataset_index.json")
    p.add_argument("--source", default="reports/m46_occupancy_validation.json")
    p.add_argument(
        "--raw", default="/root/gpufree-data/hs-drive-data/raw/m46_occupancy"
    )
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() or Path(a.report).exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / "inputs").mkdir()
    raw = Path(a.raw)
    report = json.loads(Path(a.source).read_text())
    rows = []
    refined = (
        {}
        if not a.labels
        else {
            (r["scene"], r["frame_id"]): r
            for r in json.loads(Path(a.labels).read_text())["records"]
        }
    )
    for row in report["records"]:
        if row["category"] == "dynamic":
            continue
        root = raw / row["scene"]
        man = json.loads((root / "manifest.json").read_text())
        frame_path = root / "frames" / f"{row['frame_id']:08d}.json"
        assert file_hash(frame_path) == row["source_frame_sha256"]
        f = json.loads(frame_path.read_text())
        cal = json.loads((root / "calibration.json").read_text())["cameras"]
        sample_id = f"{row['scene']}/{row['frame_id']:08d}"
        images = []
        for name in CAMERA_IDS:
            c = f["cameras"][name]
            images.append(dict(path=str(root / c["rgb"]), sha256=c["rgb_sha256"]))
        inputs = dict(
            sample_id=sample_id,
            images=images,
            intrinsics=[cal[n]["K"] for n in CAMERA_IDS],
            cam_to_ego=[cal[n]["T_hs_ego_from_opt"] for n in CAMERA_IDS],
            ego_pose=f["T_world_from_ego"],
            timestamp_s=f["timestamp_s"],
        )
        filename = f"inputs/{row['scene']}_{row['frame_id']:08d}.json"
        atomic_json(out / filename, inputs)
        # Same pose/route with different obstacles stays in same split. All
        # spawn45 variants held out together; suffix alone is not split policy.
        group = "Town03_spawn45" if man["spawn_index"] == 45 else "Town03_development"
        split = "val" if group == "Town03_spawn45" else "train"
        label = refined.get((row["scene"], row["frame_id"]))
        rows.append(
            dict(
                sample_id=sample_id,
                group=group,
                scene=row["scene"],
                category=row["category"],
                split=split,
                input=filename,
                input_sha256=file_hash(out / filename),
                label=row["path"] if label is None else label["path"],
                label_sha256=row["label_sha256"] if label is None else label["sha256"],
            )
        )
    index = dict(
        schema_version="hsd-m46-training-index-v1",
        samples=rows,
        camera_order=list(CAMERA_IDS),
        protocol="36 static frames for small plumbing/overfit experiment only. Group held out by route/pose; one map, validation not blind. All variants of spawn45 kept in val. No random neighboring frame split.",
        inference_input_contract="LearnedInputs, no depth/semantic/instance/counterfactual fields",
    )
    atomic_json(out / "index.json", index)
    atomic_json(
        out / "manifest.json",
        dict(
            schema_version=index["schema_version"],
            index_sha256=file_hash(out / "index.json"),
        ),
    )
    atomic_json(
        a.report,
        dict(
            index=str(out),
            index_sha256=file_hash(out / "index.json"),
            splits={
                s: [r["sample_id"] for r in rows if r["split"] == s]
                for s in ["train", "val"]
            },
            groups={
                g: sorted(set(r["split"] for r in rows if r["group"] == g))
                for g in set(r["group"] for r in rows)
            },
            small_sample_only=True,
            label_report=a.labels,
        ),
    )


if __name__ == "__main__":
    main()
