"""M4.7 reproducibility and measurement helpers; no changes to old defaults."""
import json
import shutil
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def freeze_sources(root, files):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    result={}
    for name in files:
        p=Path(name);dest=root/p.name
        if dest.exists():raise FileExistsError(dest)
        shutil.copy2(p,dest);result[str(p)]=file_hash(p)
    atomic_json(root/"manifest.json",result)
    return result


def vector(v):return [float(v.x),float(v.y),float(v.z)]


def physics_dict(vehicle):
    p=vehicle.get_physics_control()
    scalar=["max_rpm","moi","damping_rate_full_throttle","damping_rate_zero_throttle_clutch_engaged",
            "damping_rate_zero_throttle_clutch_disengaged","use_gear_autobox","gear_switch_time",
            "clutch_strength","final_ratio","mass","drag_coefficient","use_sweep_wheel_collision"]
    d={k:getattr(p,k) for k in scalar}
    d.update(center_of_mass=vector(p.center_of_mass),
        torque_curve=[[v.x,v.y] for v in p.torque_curve],
        steering_curve=[[v.x,v.y] for v in p.steering_curve],
        forward_gears=[{k:getattr(g,k) for k in ["ratio","down_ratio","up_ratio"]} for g in p.forward_gears],
        wheels=[{k:getattr(w,k) for k in ["tire_friction","damping_rate","max_steer_angle","radius",
            "max_brake_torque","max_handbrake_torque","lat_stiff_max_load","lat_stiff_value","long_stiff_value"]} for w in p.wheels])
    # CARLA wheel.position is world-space and changes with spawn; keep outside
    # the invariant physics signature used for matched comparisons.
    return d


def sample_vehicle(world,vehicle):
    snap=world.get_snapshot();a=snap.find(vehicle.id);tf=a.get_transform()
    vel=np.array(vector(a.get_velocity()));forward=np.array(vector(tf.get_forward_vector()))
    control=vehicle.get_control()
    return dict(frame=snap.frame,t=snap.timestamp.elapsed_seconds,position=vector(tf.location),
        rotation=[tf.rotation.roll,tf.rotation.pitch,tf.rotation.yaw],
        velocity=vel.tolist(),speed=float(np.linalg.norm(vel)),long_speed=float(vel@forward),
        acceleration_api=vector(a.get_acceleration()),
        long_acceleration_api=float(np.array(vector(a.get_acceleration()))@forward),
        applied={k:getattr(control,k) for k in ["throttle","brake","steer","hand_brake","reverse","gear"]})


def dynamics_metrics(rows):
    t=np.array([r["t"] for r in rows]);v=np.array([r["long_speed"] for r in rows])
    acc=np.diff(v)/np.diff(t);jerk=np.diff(acc)/np.diff(t[:-1])
    moving=(np.abs(v[:-2])>.2)|(np.abs(v[1:-1])>.2)|(np.abs(v[2:])>.2)
    sm=np.convolve(v,np.ones(5)/5,mode="valid")
    sa=np.diff(sm)/np.median(np.diff(t));sj=np.diff(sa)/np.median(np.diff(t))
    return dict(max_decel_raw=float(max(0,-acc.min())),jerk_peak_raw=float(abs(jerk).max()),
        jerk_p95_moving=float(np.quantile(abs(jerk[moving]),.95)) if moving.any() else 0.,
        jerk_p95_all=float(np.quantile(abs(jerk),.95)),max_decel_250ms=float(max(0,-sa.min())),
        jerk_p95_250ms=float(np.quantile(abs(sj),.95)),min_signed_speed=float(v.min()),
        reverse_ticks=int((v<-.05).sum()),low_speed_moving_seconds=float(((v>.08)&(v<1.5)).sum()*.05),
        api_accel_vs_velocity_derivative_mae=float(np.mean(abs(acc-np.array([r["long_acceleration_api"] for r in rows[1:]])))))


def jsonl(path, rows):
    with Path(path).open("x") as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+"\n")
