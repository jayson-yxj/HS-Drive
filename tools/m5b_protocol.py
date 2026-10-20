"""Predeclare map-geometry-only groups and experiments, before label/model results."""

import json, math
from pathlib import Path
import carla, numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    out = Path("configs/m5b_protocol.json")
    if out.exists():
        raise FileExistsError(out)
    client = carla.Client("127.0.0.1", 2100)
    client.set_timeout(120)
    old = json.loads(Path("configs/m5a_protocol.json").read_text())
    groups = []
    for town in ["Town03", "Town02", "Town01"]:
        w = (
            client.load_world(town)
            if client.get_world().get_map().name.split("/")[-1] != town
            else client.get_world()
        )
        m = w.get_map()
        poses = m.get_spawn_points()
        avoid = [g["pose"][:2] for g in old["groups"] if g["map"] == town]
        chosen = []
        for j in range(4):
            candidates = []
            for i, p in enumerate(poses):
                if i in chosen or p.location.z > 1.2:
                    continue
                xy = np.array([p.location.x, p.location.y])
                distance = min(np.linalg.norm(xy - v) for v in avoid)
                if distance < 65:
                    continue
                # Require some roadway at 25m forward; geometry only, no rendered sample selection.
                target = p.transform(carla.Location(x=25))
                wp = m.get_waypoint(
                    target, project_to_road=False, lane_type=carla.LaneType.Driving
                )
                if wp is None:
                    continue
                candidates.append((distance, i, p))
            if not candidates:
                raise RuntimeError("Insufficient distinct geometry groups")
            _, i, p = max(candidates, key=lambda v: v[0])
            chosen.append(i)
            avoid.append([p.location.x, p.location.y])
            groups.append(
                dict(
                    group=f"{town}_b{j}",
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
                )
            )
    layouts = [
        [],
        [[5, 0, "vehicle.tesla.model3", 0.05]],
        [[10, 0, "vehicle.tesla.model3", 0.05]],
        [[15, 0, "vehicle.tesla.model3", 0.05]],
        [[25, 0, "vehicle.tesla.model3", 0.05]],
        [[10, -3, "vehicle.tesla.model3", 0.05]],
        [[10, 3, "vehicle.tesla.model3", 0.05]],
        [[10, 0, "vehicle.mercedes.sprinter", 0.05]],
        [[5, 0, "vehicle.audi.a2", 0.05], [15, 0, "vehicle.tesla.model3", 0.05]],
        [[8, 1, "walker.pedestrian.0001", 0.85]],
        [[12, -1, "walker.pedestrian.0001", 1.5]],
        [],
    ]
    atomic_json(
        out,
        dict(
            schema="hsd-m5b-protocol-v1",
            groups=groups,
            configurations=layouts,
            seed=61,
            weather="ClearNoon",
            label_recipe_sha256=file_hash("datasets/m48_evidence.py"),
            training=dict(
                seeds=[61, 62],
                steps=400,
                batch_pairs=1,
                validate_every=100,
                models=["A_prior", "B_frozen", "C_context", "C_pair"],
                head_lr=0.001,
                prior_lr=0.03,
                context_lr=0.0003,
                pair_weight=0.2,
                ranking_margin=0.1,
                stable_weight=0.1,
                selection="val mean scene signed-change IoU; thresholds [.025,.05,.1,.2] only; secondary event IoU tie break",
                fairness="B/C start same M4 encoder and identically seeded new head; same 400 updates/pair order. C_pair differs only pair loss.",
                D="Not approved unless pretest validation geometry study shows oracle at least +.10 changed-obstacle IoU and C improves <.05 over B both seeds. Record decision before test.",
            ),
            evaluation=dict(
                score="sigmoid(surface)*sigmoid(obstacle semantic); conditional semantic separately",
                thresholds=[0.025, 0.05, 0.1, 0.2],
                test_unseal="Only after A/B/C/C_pair checkpoints and D decision saved",
                independent_unit="pose group, not pixel/configuration",
                paired_negatives="point-witness absence supported by same-ray empty first-hit farther than target; not whole-voxel free",
                scope="new groups on historical maps; M5-A all historical regression; static configurations, no temporal learning",
            ),
            notes="Elevated pedestrian is synthetic height stress, not natural pedestrian distribution. End config11 is independently captured return to empty. No loss or recipe tuning on test.",
        ),
    )
    print(groups, flush=True)


if __name__ == "__main__":
    main()
