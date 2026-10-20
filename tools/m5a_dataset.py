"""Build new scene-group index and unchanged V2 sidecars from paired captures."""

import argparse, json
from pathlib import Path
import numpy as np
from tools.m48_spatial import frame_rays
from datasets.m48_evidence import build_evidence
from simulator.recorder import atomic_json, file_hash
from tools.m47_common import freeze_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", nargs="+", required=True)
    p.add_argument(
        "--output", default="/root/gpufree-data/hs-drive-data/processed/m5a_index"
    )
    p.add_argument("--report", default="reports/m5a_data_quality.json")
    a = p.parse_args()
    root = Path(a.output)
    if root.exists() or Path(a.report).exists():
        raise FileExistsError("new paths required")
    root.mkdir(parents=True)
    samples = []
    records = []
    for raw in a.raw:
        for directory in sorted(Path(raw).iterdir()):
            if not (directory / "manifest.json").exists():
                continue
            man = json.loads((directory / "manifest.json").read_text())
            cal = json.loads((directory / "calibration.json").read_text())["cameras"]
            names = list(cal)
            poses = []
            for fid in man["frames"]:
                local = {**man, "counterfactual_frame": man["references"][str(fid)]}
                rays, box = frame_rays(directory, local, fid)
                uv = rays["uv"]
                sparse = (uv % 16 == 0).all(-1)
                held = (uv[:, 0] + 2 * uv[:, 1]) % 4 == 1
                label, stats = build_evidence(
                    rays["origins"][sparse],
                    rays["endpoints"][sparse],
                    rays["endpoints"][~held],
                    rays["semantics"][~held],
                    ego_box=box,
                )
                scene = man["scene"]
                sample = f"{scene}/{fid:08d}"
                folder = root / scene
                folder.mkdir(exist_ok=True)
                labelpath = folder / f"{fid:08d}.npz"
                np.savez_compressed(labelpath, **label)
                frame = json.loads(
                    (directory / "frames" / f"{fid:08d}.json").read_text()
                )
                poses.append(frame["T_world_from_ego"])
                inp = dict(
                    sample_id=sample,
                    images=[
                        dict(
                            path=str(directory / frame["cameras"][n]["rgb"]),
                            sha256=frame["cameras"][n]["rgb_sha256"],
                        )
                        for n in names
                    ],
                    intrinsics=[cal[n]["K"] for n in names],
                    cam_to_ego=[cal[n]["T_hs_ego_from_opt"] for n in names],
                    ego_pose=frame["T_world_from_ego"],
                    timestamp_s=frame["timestamp_s"],
                )
                ip = folder / f"{fid:08d}_input.json"
                atomic_json(ip, inp)
                samples.append(
                    dict(
                        sample_id=sample,
                        group=scene,
                        split=man["split"],
                        configuration=frame["configuration"],
                        map=man["map"],
                        input=str(ip.relative_to(root)),
                        input_sha256=file_hash(ip),
                        label=str(labelpath),
                        label_sha256=file_hash(labelpath),
                        raw_frame=str(directory / "frames" / f"{fid:08d}.json"),
                        camera_names=names,
                    )
                )
                records.append(
                    dict(
                        sample_id=sample,
                        configuration=frame["configuration"],
                        stats=stats,
                        obstacle_positive=int(
                            (
                                label["surface_semantic"][1] & label["obstacle_valid"]
                            ).sum()
                        ),
                        ground_positive=int(
                            (
                                label["surface_semantic"][0]
                                & label["surface_semantic_valid"]
                            ).sum()
                        ),
                        other_positive=int(
                            (
                                label["surface_semantic"][2]
                                & label["surface_semantic_valid"]
                            ).sum()
                        ),
                        reliable_rays=len(rays["endpoints"]),
                        label_sha256=file_hash(labelpath),
                    )
                )
                print(
                    sample,
                    stats["valid_coverage"],
                    records[-1]["obstacle_positive"],
                    flush=True,
                )
            assert np.max(np.abs(np.array(poses) - poses[0])) < 1e-8
    index = dict(samples=samples)
    atomic_json(root / "index.json", index)
    atomic_json(
        root / "manifest.json",
        dict(
            schema_version="hsd-m46-training-index-v1",
            index_sha256=file_hash(root / "index.json"),
            supervision_schema="hsd-m48-spatial-evidence-v2",
            protocol_sha256=file_hash("configs/m5a_protocol.json"),
        ),
    )
    atomic_json(
        a.report,
        dict(
            kind="offline_V2_labels_actual_paired_CARLA",
            records=records,
            index=str(root),
            index_sha256=file_hash(root / "index.json"),
            sources={
                p: file_hash(p)
                for p in [
                    __file__,
                    "datasets/m48_evidence.py",
                    "tools/m48_spatial.py",
                    "datasets/occupancy_quality.py",
                ]
            },
            same_pose_within_group=True,
            split_counts={
                s: sum(x["split"] == s for x in samples)
                for s in ["train", "val", "test"]
            },
            notes="No label quality selection based on val/test. Existing V2 recipe, native nonheld endpoints/free16. Fixed per-configuration ego-removed reference.",
        ),
    )


if __name__ == "__main__":
    main()
