"""Requery static collision road around historical actor poses. No original mesh claim."""

import json
from pathlib import Path
import numpy as np, carla
from tools.m5l_common import rows, write, asset
from diagnostics.m5m_contact import fit_plane
from simulator.recorder import file_hash


def main():
    client = carla.Client("127.0.0.1", 2100)
    client.set_timeout(120)
    records = []
    rs = [r for split in ["train", "val", "regression", "test"] for r in rows(split)]
    for town in sorted({r["map"].split("/")[-1] for r in rs}):
        w = client.load_world(town)
        settings = w.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        w.apply_settings(settings)
        w.tick()
        try:
            cache = {}
            for row in [r for r in rs if r["map"].split("/")[-1] == town]:
                f = json.loads(Path(row["raw_frame"]).read_text())
                ego = np.array(f["T_world_from_ego"])
                inv = np.linalg.inv(ego)
                for actor in f["actors"]:
                    if actor["is_ego"]:
                        continue
                    tf = np.array(actor["world_matrix"])
                    key = (
                        row["group"],
                        actor["type_id"],
                        tuple(np.round(tf[:3, 3], 3)),
                    )
                    if key in cache:
                        continue
                    cache[key] = True
                    center = tf[:3, 3]
                    probes = []
                    pts = []
                    if (
                        np.linalg.norm(np.array(actor["corners_hs_ego_m"]).mean(0)[:2])
                        > 80
                    ):
                        continue
                    for dx, dy in [(0, 0), (-1, -0.8), (-1, 0.8), (1, -0.8), (1, 0.8)]:
                        start = center + np.array([dx, dy, 4])
                        end = center + np.array([dx, dy, -8])
                        hits = w.cast_ray(carla.Location(*start), carla.Location(*end))
                        rr = []
                        for h in hits:
                            loc = np.array([h.location.x, h.location.y, h.location.z])
                            hs = (inv @ np.r_[loc * [1, -1, 1], 1])[:3]
                            rr.append(
                                dict(
                                    label=int(h.label),
                                    world=loc.tolist(),
                                    hs_ego=hs.tolist(),
                                )
                            )
                            if int(h.label) in [1, 24]:
                                pts.append(hs)
                        probes.append(
                            dict(start=start.tolist(), end=end.tolist(), hits=rr)
                        )
                    plane = fit_plane(pts, minimum=3)
                    box = np.array(actor["corners_hs_ego_m"])
                    cen = box.mean(0)
                    gap = (
                        None
                        if plane is None
                        else float(
                            box[:, 2].min()
                            - np.dot(plane["coefficients"], [*cen[:2], 1])
                        )
                    )
                    records.append(
                        dict(
                            group=row["group"],
                            map=town,
                            sample_id=row["sample_id"],
                            configuration=row["configuration"],
                            type_id=actor["type_id"],
                            actor_id=actor["actor_id"],
                            actor_world_matrix=tf.tolist(),
                            probes=probes,
                            collision_road_plane=plane,
                            aabb_bottom_gap_m=gap,
                            physical_contact_verified=False,
                        )
                    )
            print(town, len(records), flush=True)
        finally:
            settings.synchronous_mode = False
            w.apply_settings(settings)
    write(
        "runs/m5m_collision_queries.json",
        dict(
            kind="actual_CARLA_static_collision_requery_of_recorded_poses",
            records=records,
            limits="Maps reloaded without original dynamic actors. Multiple road collision returns are supplemental approximate geometry, not original rendered mesh, wheel contact, or complete occlusion truth.",
        ),
    )
    write(
        "reports/m5m_collision_queries.json",
        dict(
            raw=asset("runs/m5m_collision_queries.json"),
            records=len(records),
            source_sha256=file_hash(__file__),
        ),
    )


if __name__ == "__main__":
    main()
