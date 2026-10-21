"""Geometry-only scene allocation and frozen M5-E research decisions."""

import json
from pathlib import Path
import numpy as np
import carla
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("configs/m5g_capture_protocol.json")
    if dest.exists():
        raise FileExistsError(dest)
    client = carla.Client("127.0.0.1", 2100)
    client.set_timeout(120)
    old = []
    for f in [
        "configs/m5a_protocol.json",
        "configs/m5b_protocol.json",
        "configs/m5c_protocol.json",
        "configs/m5d_protocol.json",
        "configs/m5e_protocol.json",
        "configs/m5f_capture_protocol.json",
    ]:
        old += json.loads(Path(f).read_text())["groups"]
    groups = []
    for town in ["Town02", "Town03", "Town05"]:
        world = client.load_world(town)
        m = world.get_map()
        avoid = [g["pose"][:2] for g in old if g["map"] == town]
        poses = m.get_spawn_points()
        for j in range(1):
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
                    (j + ["Town02", "Town03", "Town05"].index(town)) % 4
                ]
                candidates.append((category == desired, distance, i, p, category))
            if not candidates:
                raise RuntimeError(f"No independent pose {town} {j}")
            _, dist, i, p, category = max(candidates, key=lambda a: (a[0], a[1]))
            avoid.append([p.location.x, p.location.y])
            groups.append(
                dict(
                    group=f"{town}_g0",
                    map=town,
                    split="test",
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
                    weather={"Town02":"ClearSunset","Town03":"WetCloudySunset","Town05":"CloudyNoon"}[town],
                )
            )
    layouts=[[],[[10,0,"vehicle.tesla.model3",.05]],[[25,0,"vehicle.tesla.model3",.05]],[[32,0,"vehicle.tesla.model3",.05]],[[10,-10,"vehicle.tesla.model3",.05]],[[-12,-8,"vehicle.tesla.model3",.05]],[[-15,0,"vehicle.tesla.model3",.05]],[[25,1,"walker.pedestrian.0001",.85]],[[30,-1,"static.prop.trafficcone01",.05]],[[12,0,"vehicle.mercedes.sprinter",.05],[28,0,"vehicle.tesla.model3",.05]],[[-10,8,"vehicle.tesla.model3",.05]],[]]
    atomic_json(dest,dict(schema="hsd-m5g-newtest-capture-v1",groups=groups,configurations=layouts,role="sealed test afterpretestfreeze",source_sha256=file_hash(__file__)))
    print(json.dumps(groups),flush=True)

if __name__=='__main__':main()
