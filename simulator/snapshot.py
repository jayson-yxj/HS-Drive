"""Capture ego and GT actors from the SAME WorldSnapshot, not live transforms."""
import numpy as np
from geometry.coordinates import (carla_pose_to_hs, carla_vector_to_hs,
                                  carla_angular_velocity_to_hs_rad)
from geometry.reference import CARLA_EGO_TO_HS_EGO, transform_points


def vector(v):
    return [v.x, v.y, v.z]


def extract_snapshot(snapshot, ego, actor_catalog):
    ego_snap = snapshot.find(ego.id)
    if ego_snap is None:
        raise RuntimeError('Ego missing from snapshot')
    T = np.asarray(ego_snap.get_transform().get_matrix())
    velocity = carla_vector_to_hs(vector(ego_snap.get_velocity()))
    state = {'T_carla_world_from_ego': T.tolist(),
             'T_world_from_ego': carla_pose_to_hs(T).tolist(),
             'coordinate_frame': 'hs_world', 'speed_mps': float(np.linalg.norm(velocity)),
             'velocity_world_mps': velocity.tolist(),
             'angular_velocity_world_radps': carla_angular_velocity_to_hs_rad(
                 vector(ego_snap.get_angular_velocity())).tolist()}
    objects = []
    world_to_ego = CARLA_EGO_TO_HS_EGO @ np.linalg.inv(T)
    for actor_id, (type_id, bbox) in actor_catalog.items():
        actor = snapshot.find(actor_id)
        if actor is None:
            continue
        corners = np.array([vector(p) for p in bbox.get_world_vertices(actor.get_transform())])
        objects.append({'actor_id': actor_id, 'type_id': type_id, 'is_ego': actor_id == ego.id,
                        'corners_hs_ego_m': transform_points(corners, world_to_ego).tolist(),
                        'source': 'actor_bbox_snapshot'})
    return state, objects
