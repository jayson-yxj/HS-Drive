"""Read real CARLA telemetry without inventing timestamps or brake torque.

No inference/control code imports this optional research module.
"""
import math
import time

VEHICLE_FIELDS = ('speed', 'steer', 'throttle', 'brake', 'engine_rpm', 'gear', 'drag')
WHEEL_FIELDS = ('tire_friction', 'lat_slip', 'long_slip', 'omega', 'tire_load',
                'normalized_tire_load', 'torque', 'long_force', 'lat_force',
                'normalized_long_force', 'normalized_lat_force')


def scalar(value):
    """Keep invalid engine values explicitly; JSON never silently maps NaN to 0."""
    value = float(value)
    return value if math.isfinite(value) else {'invalid_raw': repr(value)}


def serialize(telemetry):
    result = {k: scalar(getattr(telemetry, k)) for k in VEHICLE_FIELDS}
    result['wheels'] = [{k: scalar(getattr(w, k)) for k in WHEEL_FIELDS}
                        for w in telemetry.wheels]
    return result


def observe(world, vehicle):
    """Bracket synchronous RPC by snapshots; caller owns the only tick.

    Stable frame bracketing is NOT an internal solver/substep timestamp.
    """
    before = world.get_snapshot()
    start = time.perf_counter()
    data = serialize(vehicle.get_telemetry_data())
    end = time.perf_counter()
    after = world.get_snapshot()
    stable = before.frame == after.frame
    return dict(data=data, frame_before=before.frame, frame_after=after.frame,
                associated_frame=before.frame if stable else None,
                timestamp_s=before.timestamp.elapsed_seconds if stable else None,
                engine_telemetry_timestamp=None, engine_substep=None,
                alignment='exclusive_tick_bracket' if stable else 'Invalid_FrameChanged',
                rpc_ms=1000*(end-start), true_engine_wheel_api=True)


def validate_pair(before, after, expected_next):
    if before['associated_frame'] is None or after['associated_frame'] != expected_next:
        raise ValueError('TelemetrySnapshotMismatch')
    if expected_next != before['associated_frame'] + 1:
        raise ValueError('NonconsecutiveTelemetryFrames')
