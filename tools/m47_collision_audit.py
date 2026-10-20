"""Replay saved actor transforms, query collision geometry for all 104 cells."""

import argparse, json
from pathlib import Path
import numpy as np, yaml
from simulator.session import CarlaSession
from simulator.recorder import atomic_json, file_hash
from geometry.reference import transform_points
from tools.m47_geometry_queries import cast
from tools.m47_common import freeze_sources


def main():
    import carla

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="reports/m47_collision_audit.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    audit = json.loads(Path("reports/m47_false_free_audit.json").read_text())
    raw = Path("/root/gpufree-data/hs-drive-data/raw/m46_occupancy")
    cfg = yaml.safe_load(Path("configs/carla.yaml").read_text())
    cfg.update(port=2100, map="Town03")
    results = []
    with CarlaSession(cfg, load_map=True) as session:
        w = session.world
        for scene in sorted(set(r["scene"] for r in audit["records"])):
            root = raw / scene
            spawned = []
            byframe = sorted(
                set(r["frame_id"] for r in audit["records"] if r["scene"] == scene)
            )
            for fid in byframe:
                frame = json.loads((root / "frames" / f"{fid:08d}.json").read_text())
                for actor in spawned:
                    actor.destroy()
                    session.actors.remove(actor)
                spawned = []
                for actor in frame["actors"]:
                    # Keep ego absent to query geometry beyond its body. All
                    # audited Free rays already carry M46 ego visibility veto.
                    if actor["is_ego"]:
                        continue
                    m = np.array(actor["world_matrix"])
                    r = m[:3, :3]
                    tf = carla.Transform(
                        carla.Location(*m[:3, 3]),
                        carla.Rotation(
                            pitch=float(np.degrees(np.arcsin(r[2, 0]))),
                            yaw=float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))),
                            roll=float(np.degrees(np.arctan2(-r[2, 1], r[2, 2]))),
                        ),
                    )
                    obj = session.spawn(
                        w.get_blueprint_library().find(actor["type_id"]), tf
                    )
                    obj.set_simulate_physics(False)
                    spawned.append(obj)
                w.tick()
                pose = np.array(frame["T_world_from_ego"])
                to_carla = np.diag([1, -1, 1, 1]) @ pose
                for rec in [
                    r
                    for r in audit["records"]
                    if r["scene"] == scene and r["frame_id"] == fid
                ]:
                    rays = []
                    matches = []
                    for ray in rec["free_rays"]:
                        seg = transform_points(
                            np.array([ray["origin"], ray["endpoint"]]), to_carla
                        )
                        hits = cast(w, *seg)
                        length = np.linalg.norm(seg[1] - seg[0])
                        for h in hits:
                            h["distance_from_origin"] = float(
                                np.linalg.norm(np.array(h["point"]) - seg[0])
                            )
                            h["point_hs"] = transform_points(
                                np.array(h["point"]), np.linalg.inv(to_carla)
                            ).tolist()
                        early = [
                            h
                            for h in hits
                            if 0.2 < h["distance_from_origin"] < length - 0.15
                        ]
                        rays.append(
                            dict(
                                camera=ray["camera"],
                                uv=ray["uv"],
                                hits=hits,
                                collision_before_render_endpoint=bool(early),
                                early_hits=early,
                            )
                        )
                    # Collision hit along heldout pixel: compare first collision
                    # endpoint to rendered endpoint; no assumption of equality.
                    for obs in rec["heldout_surface"]:
                        o = np.array(obs["origin"])
                        e = np.array(obs["endpoint"])
                        direction = (e - o) / np.linalg.norm(e - o)
                        seg = transform_points(
                            np.stack([o, e + 0.3 * direction]), to_carla
                        )
                        hits = cast(w, *seg)
                        distance = min(
                            (
                                float(
                                    np.linalg.norm(
                                        np.array(h["point"])
                                        - transform_points(e, to_carla)
                                    )
                                )
                                for h in hits
                            ),
                            default=None,
                        )
                        matches.append(
                            dict(
                                camera=obs["camera"],
                                uv=obs["uv"],
                                closest_collision_surface_distance_m=distance,
                                hits=hits,
                            )
                        )
                    result = dict(
                        id=rec["id"],
                        scene=scene,
                        frame_id=fid,
                        free_rays=rays,
                        heldout_surface_checks=matches,
                        dangerous_ray_contradiction_candidate=any(
                            r["collision_before_render_endpoint"] for r in rays
                        ),
                        render_collision_agreement_count=sum(
                            r["closest_collision_surface_distance_m"] is not None
                            and r["closest_collision_surface_distance_m"] < 0.15
                            for r in matches
                        ),
                        attribution="Collision representation discrepancy or physical obstruction needs inspection if early hit; finite rays do not certify whole cell.",
                    )
                    results.append(result)
                print(scene, fid, flush=True)
            for actor in spawned:
                actor.destroy()
                session.actors.remove(actor)
    atomic_json(
        a.output,
        dict(
            kind="actual_carla_geometry_replay",
            records=results,
            source_audit_sha256=file_hash("reports/m47_false_free_audit.json"),
            dangerous_candidates=sum(
                r["dangerous_ray_contradiction_candidate"] for r in results
            ),
            limitations=[
                "Collision geometry differs from render mesh and is not exact independent GT.",
                "Static environment reused; dynamic actors restored to recorded transforms, ego removed.",
                "No-hit is not proof of an empty voxel; conservative Unknown quarantine remains.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
