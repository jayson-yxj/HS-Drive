"""Geometry-only contact layout allocation; no rendered predictions used."""

import argparse, json, math
from pathlib import Path
import numpy as np, carla
from tools.m5l_common import write
from simulator.recorder import file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["dev", "sealed"], required=True)
    a = p.parse_args()
    if a.phase == "sealed":
        assert Path("reports/m5m_pretest_freeze.json").exists()
    c = carla.Client("127.0.0.1", 2100)
    c.set_timeout(120)
    old = []
    for p in Path("configs").glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(d, dict):
            old.extend(g for g in d.get("groups", []) if "pose" in g and "map" in g)
    man = json.loads(Path("runs/m4_expanded_joint/manifest.json").read_text())
    idx = json.loads(Path(man["config"]["index"], "index.json").read_text())
    groups = []
    for i, town in enumerate(["Town01", "Town03", "Town04", "Town05"]):
        w = c.load_world(town)
        mp = w.get_map()
        avoid = [g["pose"][:2] for g in old if g["map"].split("/")[-1] == town]
        for scene in idx["scenes"]:
            if scene["map"].split("/")[-1] == town:
                avoid.extend((np.array(scene["positions_xy"]) * [1, -1]).tolist())
        candidates = []
        for wp in mp.generate_waypoints(8):
            t = wp.transform
            dist = min(
                [
                    np.linalg.norm(np.array([t.location.x, t.location.y]) - xy)
                    for xy in avoid
                ]
                or [1000]
            )
            pitch = abs(
                math.degrees(math.asin(math.sin(math.radians(t.rotation.pitch))))
            )
            if dist >= 45 and not wp.is_junction and wp.next(20):
                candidates.append((pitch, dist, t))
        assert candidates, town
        pitch, dist, t = max(
            candidates, key=lambda r: (r[0] if i == 1 else -r[0], r[1])
        )
        groups.append(
            dict(
                group=f"{town}_m5m_{a.phase}{i}",
                map=town,
                split=(
                    "test" if a.phase == "sealed" else "train" if i % 2 == 0 else "val"
                ),
                pose=[
                    t.location.x,
                    t.location.y,
                    t.location.z + 0.4,
                    t.rotation.pitch,
                    t.rotation.yaw,
                    t.rotation.roll,
                ],
                min_prior_m=float(dist),
                geometry="slope" if pitch > 1 else "flat_road",
                actual_pitch_deg=pitch,
                weather="ClearNoon",
            )
        )
        print(groups[-1], flush=True)
    car = "vehicle.tesla.model3"
    van = "vehicle.mercedes.sprinter"
    ped = "walker.pedestrian.0001"
    cone = "static.prop.trafficcone01"
    configs = [
        [],
        [[8, 0, car, 0.05]],
        [[15, 1, van, 0.05]],
        [[8, -2, ped, 0.85]],
        [[6, 2, cone, 0.05]],
        [[10, 0, car, 0.05], [18, 0, ped, 0.85]],
        [[10, -14, car, 0.05]],
        [[12, 8, van, 0.05]],
        [],
    ]
    write(
        f"configs/m5m_capture_{a.phase}.json",
        dict(
            groups=groups,
            configurations=configs,
            source_sha256=file_hash(__file__),
            purpose="Anatomy and road consistency feasibility, no forced Gold labels; dev never sealed",
        ),
    )


if __name__ == "__main__":
    main()
