"""Geographic, physical placement and Empty/Removed audit without model selection."""

import argparse, json
from pathlib import Path
from collections import Counter
import cv2, numpy as np
from tools.m5o_common import DATA, write, asset, provenance
from geometry.reference import transform_points


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    a = p.parse_args()
    root = DATA / "raw" / f"m5o_{a.split}_indexroot"
    records = []
    failures = []
    for path in sorted(root.glob("*/manifest.json")):
        m = json.loads(path.read_text())
        frames = [
            json.loads((path.parent / "frames" / f"{fid:08d}.json").read_text())
            for fid in m["frames"]
        ]
        counts = []
        actors = []
        first, last = frames[0], frames[-1]
        empty = []
        for cam in first["cameras"]:
            im0 = cv2.imread(str(path.parent / first["cameras"][cam]["rgb"]))
            im1 = cv2.imread(str(path.parent / last["cameras"][cam]["rgb"]))
            empty.append(
                dict(
                    camera=cam,
                    rgb_abs_mean=float(np.abs(im0.astype(float) - im1).mean()),
                    rgb_exact_equal=bool(np.array_equal(im0, im1)),
                )
            )
        for f in frames:
            for obj in f["actors"]:
                center = np.mean(obj["corners_hs_ego_m"], axis=0)
                if obj["is_ego"] or np.linalg.norm(center[:2]) > 80:
                    continue
                contact = obj["contact_reference"] or {}
                probes = contact.get("probes", [])
                road = [
                    h["xyz"][2] for hs in probes for h in hs if h["label"] in [1, 24]
                ]
                world = transform_points(
                    np.array(obj["corners_hs_ego_m"]), np.array(f["T_world_from_ego"])
                )
                bottom = float(world[:, 2].min())
                gap = (
                    None
                    if not road
                    else bottom - min(road, key=lambda z: abs(z - bottom))
                )
                actors.append(
                    dict(
                        frame_id=f["frame_id"],
                        configuration=f["configuration"],
                        actor_id=obj["actor_id"],
                        type_id=obj["type_id"],
                        center_ego=center.tolist(),
                        road_probe_found=bool(contact.get("road_probe_found")),
                        obb_bottom_minus_nearest_road_m=gap,
                        limits="OBB bottom is not wheel/mesh contact; pitched vehicles may extend outside sampled road patch",
                    )
                )
                if not np.isfinite(center).all():
                    failures.append([m["group"], f["frame_id"], "nonfinite_center"])
        records.append(
            dict(
                group=m["group"],
                split=a.split,
                map=m["map"],
                frames=len(frames),
                weather=m["weather"],
                geometry=m["geometry"],
                placement=actors,
                empty_removed=empty,
                source=asset(path),
            )
        )
    write(
        f"reports/m5o_capture_checks_{a.split}.json",
        dict(
            kind="offline_actual_CARLA_geometry_and_rgb_pair_audit",
            records=records,
            failures=failures,
            provenance=provenance(),
            limits="Light/traffic signal time changes may alter Empty/Removed RGB; no pixel identity assumption. Collision-road hit and OBB gap are diagnostic proxies, not exact contact Gold. No RGB score-based sample selection.",
        ),
    )


if __name__ == "__main__":
    main()
