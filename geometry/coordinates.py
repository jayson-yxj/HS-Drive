"""Coordinate transforms; geometry/reference.py remains the single math source."""
import numpy as np
from .reference import (CARLA_EGO_TO_HS_EGO, OPTICAL_TO_CARLA_CAM,
                        camera_mount_to_hs_from_optical, transform_points,
                        temporal_prev_to_current)


def carla_pose_to_hs(T_carla_world_from_carla_ego):
    """Both world and ego become right-handed; translations are in meters."""
    S = CARLA_EGO_TO_HS_EGO
    return S @ np.asarray(T_carla_world_from_carla_ego) @ S


def carla_vector_to_hs(vector):
    return np.asarray(vector) * [1, -1, 1]


def carla_angular_velocity_to_hs_rad(vector_deg):
    # An angular velocity is an axial vector: det(S)*S, not S.
    return np.deg2rad(vector_deg) * [-1, 1, -1]
