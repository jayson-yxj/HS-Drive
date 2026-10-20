"""CARLA collision ray queries are supplemental geometry, never exact render GT."""

import json
from pathlib import Path
import numpy as np
from geometry.reference import transform_points, unproject_pixels_to_optical
from simulator.recorder import atomic_json


def cast(world, origin, endpoint):
    import carla

    return [
        dict(point=[p.location.x, p.location.y, p.location.z], semantic=int(p.label))
        for p in world.cast_ray(
            carla.Location(*map(float, origin)), carla.Location(*map(float, endpoint))
        )
    ]


def query_frame(world, ego, frame, cal, directory):
    """Seeded native image rays independent of label stride and vertical probes.

    Query from beyond near body, recording that range. Same simulation assets,
    different collision representation. No claim of independent real-world GT.
    """
    rng = np.random.default_rng(4701)
    matrix = np.asarray(ego.get_transform().get_matrix())
    carla_from_hs = matrix @ np.diag([1, -1, 1, 1])
    records = []
    for name, c in cal["cameras"].items():
        uv = np.stack([rng.integers(0, 640, 100), rng.integers(0, 384, 100)], -1)
        rays = transform_points(
            unproject_pixels_to_optical(
                uv[:, 0], uv[:, 1], np.full(100, 50.0), np.array(c["K"]), "optical_z"
            ),
            np.array(c["T_hs_ego_from_opt"]),
        )
        origins = np.tile(np.array(c["T_hs_ego_from_opt"])[:3, 3], (100, 1))
        # Original ray origin preserved; collision can hit ego (explicitly).
        for pixel, o, e in zip(uv, origins, rays):
            points = transform_points(np.stack([o, e]), carla_from_hs)
            hits = cast(world, *points)
            for hit in hits:
                hit["point_hs"] = transform_points(
                    np.array(hit["point"]), np.linalg.inv(carla_from_hs)
                ).tolist()
            records.append(
                dict(
                    camera=name,
                    uv=pixel.tolist(),
                    origin=o.tolist(),
                    endpoint=e.tolist(),
                    hits=hits,
                )
            )
    # Vertical ground queries are not tied to the camera pixel lattice.
    vertical = []
    for x in np.linspace(-10, 40, 11):
        for y in np.linspace(-15, 15, 7):
            segment = np.array([[x, y, 4.0], [x, y, -3.0]])
            hits = cast(world, *transform_points(segment, carla_from_hs))
            for h in hits:
                h["point_hs"] = transform_points(
                    np.array(h["point"]), np.linalg.inv(carla_from_hs)
                ).tolist()
            vertical.append(dict(x=float(x), y=float(y), hits=hits))
    atomic_json(
        Path(directory) / f"m48_geometry_{frame['frame_id']:08d}.json",
        dict(
            frame_id=frame["frame_id"],
            ray_queries=records,
            vertical_queries=vertical,
            kind="actual_carla_collision_queries",
            representation="CARLA cast_ray collision geometry; may differ from render mesh",
        ),
    )
