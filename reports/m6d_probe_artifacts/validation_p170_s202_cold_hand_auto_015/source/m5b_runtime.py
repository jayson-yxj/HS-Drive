"""One supplied immutable CARLA snapshot for state/measurement; no world tick."""

import math
import numpy as np
from simulator.motion_oracle import OracleState
from tools.m47_common import vector


def snapshot_state_measurement(snapshot, actor_id, control):
    actor = snapshot.find(actor_id)
    if actor is None:
        raise ValueError("Actor absent from current snapshot")
    tf = actor.get_transform()
    vel = np.asarray(vector(actor.get_velocity()))
    forward = np.asarray(vector(tf.get_forward_vector()))
    acceleration = np.asarray(vector(actor.get_acceleration()))
    timestamp = snapshot.timestamp.elapsed_seconds
    speed = float(np.linalg.norm(vel))
    state = OracleState(
        tf.location.x, -tf.location.y, -math.radians(tf.rotation.yaw), speed, timestamp
    )
    measurement = dict(
        frame=snapshot.frame,
        t=timestamp,
        position=vector(tf.location),
        rotation=[tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw],
        velocity=vel.tolist(),
        speed=speed,
        long_speed=float(vel @ forward),
        acceleration_api=acceleration.tolist(),
        long_acceleration_api=float(acceleration @ forward),
        applied={
            k: getattr(control, k)
            for k in ["throttle", "brake", "steer", "hand_brake", "reverse", "gear"]
        },
    )
    return state, measurement
