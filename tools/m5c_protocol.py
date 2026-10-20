"""Geometry-only scene allocation and frozen M5-C research decisions."""

import json
from pathlib import Path
import numpy as np
import carla
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("configs/m5c_protocol.json")
    if dest.exists():
        raise FileExistsError(dest)
    client = carla.Client("127.0.0.1", 2100)
    client.set_timeout(120)
    old = []
    for f in ["configs/m5a_protocol.json", "configs/m5b_protocol.json"]:
        old += json.loads(Path(f).read_text())["groups"]
    groups = []
    for town in ["Town01", "Town03", "Town04", "Town05"]:
        world = client.load_world(town)
        m = world.get_map()
        avoid = [g["pose"][:2] for g in old if g["map"] == town]
        poses = m.get_spawn_points()
        for j in range(4):
            candidates = []
            for i, p in enumerate(poses):
                xy = np.array([p.location.x, p.location.y])
                distance = min([np.linalg.norm(xy - a) for a in avoid] or [1000.0])
                if distance < 45:
                    continue
                wp = m.get_waypoint(p.location, lane_type=carla.LaneType.Driving)
                ahead = wp.next(15)
                if not ahead:
                    continue
                target = p.transform(carla.Location(x=15))
                if (
                    m.get_waypoint(
                        target, project_to_road=False, lane_type=carla.LaneType.Driving
                    )
                    is None
                ):
                    continue
                junction = wp.is_junction or any(q.is_junction for q in ahead)
                curve = (
                    abs(
                        (ahead[0].transform.rotation.yaw - p.rotation.yaw + 180) % 360
                        - 180
                    )
                    > 8
                )
                category = "junction" if junction else "curve" if curve else "straight"
                desired = ["straight", "junction", "curve", "straight"][
                    (j + ["Town01", "Town03", "Town04", "Town05"].index(town)) % 4
                ]
                candidates.append((category == desired, distance, i, p, category))
            if not candidates:
                raise RuntimeError(f"No independent pose {town} {j}")
            _, dist, i, p, category = max(candidates, key=lambda a: (a[0], a[1]))
            avoid.append([p.location.x, p.location.y])
            groups.append(
                dict(
                    group=f"{town}_c{j}",
                    map=town,
                    split=["train", "train", "val", "test"][j],
                    spawn_index=i,
                    pose=[
                        p.location.x,
                        p.location.y,
                        p.location.z,
                        p.rotation.pitch,
                        p.rotation.yaw,
                        p.rotation.roll,
                    ],
                    geometry=category,
                    min_previous_pose_distance_m=float(dist),
                    weather=(
                        ["ClearNoon", "CloudyNoon", "ClearNoon", "CloudyNoon"][j]
                        if town in ["Town01", "Town04"]
                        else ["ClearSunset", "ClearNoon", "ClearSunset", "ClearNoon"][j]
                    ),
                )
            )
    layouts = [[]] + [
        [[d, 0, "vehicle.tesla.model3", 0.05]] for d in [5, 10, 15, 20, 25, 30]
    ]
    layouts += [
        [[15, -3, "vehicle.tesla.model3", 0.05]],
        [[20, 3, "vehicle.tesla.model3", 0.05]],
        [[20, 0, "vehicle.carlamotors.carlacola", 0.05]],
        [
            [8, 0, "vehicle.mercedes.sprinter", 0.05],
            [20, 0, "vehicle.tesla.model3", 0.05],
        ],
        [[15, 1, "walker.pedestrian.0001", 0.85]],
        [[25, -1, "walker.pedestrian.0001", 0.85]],
        [[10, 1, "static.prop.trafficcone01", 0.05]],
        [[20, 1, "walker.pedestrian.0001", 1.5]],
        [],
    ]
    atomic_json(
        dest,
        dict(
            schema="hsd-m5c-protocol-v1",
            groups=groups,
            configurations=layouts,
            seed=71,
            label_recipe_sha256=file_hash("datasets/m48_evidence.py"),
            visibility_sha256=file_hash("datasets/occupancy_quality.py"),
            pair_sha256=file_hash("datasets/m5b_pairs.py"),
            training=dict(
                seeds=[71, 72],
                steps=400,
                validate_every=100,
                models=["P_prior", "A_frozen", "B_context", "C_pair", "D_direct"],
                head_lr=0.001,
                prior_lr=0.03,
                context_lr=0.0003,
                pair_weight=0.2,
                direct_weight=1 / 3,
                selection="maximum mean per-scene absolute visible-surface exact IoU on validation; threshold lower wins tie; earlier checkpoint wins tie",
                thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                fairness="same initial M4 encoder; seeded identical shared heads; paired schedule same seed/order allmodels; D differs from C only additional independent direct head+masked auxiliary BCE; old outputs retained",
                direct_target="V2 reliable obstacle surfaces positive; reliable ground surfaces negative; Unknown/free-ray-only/mixedsemantic excluded",
            ),
            evaluation=dict(
                independent_unit="position group; all variants stay together",
                absolute="Rendered held-out pixel surface witness >=3/cell, ground vs obstacle only, conflicting semantics or ego excluded; same-layer many-to-one nearest-neighbor coverage, NOT wholevolume or independentMesh truth",
                heldout="(u+2*v)%4==1; disjoint from V2 surface endpoints, same renderer",
                actor="AABB only associates reliable rendered surface endpoints; >=3visiblecells eligible; >=3 matched and >=20% within1m sameheight detected",
                test="sealed raw group excluded from train/val index; open once after all checkpoints thresholds research decisions frozen",
                change_thresholds=[0.025, 0.05, 0.1, 0.2],
                tolerances_m=[0.5, 1.0],
                distance_bands_m=[[0, 10], [10, 15], [15, 20], [20, 35], [35, 80]],
                quality_gate=dict(
                    precision=0.8, recall=0.8, iou=0.6, far_recall=0.7, each_seed=True
                ),
            ),
            runtime=dict(
                planner_hz=5,
                controller_hz=20,
                safety_horizon_s=6,
                interval_p99_max_ms=55,
                interval_max_ms=100,
                deadline_miss_fraction_max=0.001,
                expired_normal_control=0,
                planned_runs="two isolated campaigns and >=300s GPU-inference pressure soak plus fault cases",
                no_default_changes=True,
            ),
            notes="No layout visibility selection from test. Traffic cone semantic may be Other and is then not a valid obstacle target; report coverage failure instead of relabeling. Elevated pedestrian synthetic height stress. Groups >=45m from old M5A/B and other new groups; independent group is not independent map. Architecture/loss fixed before new capture; no depth tuning.",
        ),
    )
    print(json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
