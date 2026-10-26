"""Serialization helpers for actual immutable CARLA snapshots, not placement."""
import numpy as np


def xyz(v):
    return [float(v.x),float(v.y),float(v.z)]


def pose(t):
    return dict(xyz=xyz(t.location),rotation=[t.rotation.pitch,t.rotation.yaw,t.rotation.roll],matrix=t.get_matrix())


def box(bb, trans=None):
    import carla
    m=np.asarray(carla.Transform(bb.location,bb.rotation).get_matrix())
    if trans is not None:
        m=np.asarray(trans.get_matrix())@m
    e=np.array(xyz(bb.extent))
    offsets=np.array([[x,y,z] for x in [-e[0],e[0]] for y in [-e[1],e[1]] for z in [-e[2],e[2]]])
    return dict(T_world_from_box=m.tolist(),center_world_m=m[:3,3].tolist(),extent_m=e.tolist(),
                dimensions_lwh_m=(2*e).tolist(),corners_world_m=(offsets@m[:3,:3].T+m[:3,3]).tolist())


def waypoint(wp):
    if wp is None:
        return None
    return dict(road_id=wp.road_id,lane_id=wp.lane_id,section_id=wp.section_id,s=wp.s,
                lane_type=str(wp.lane_type),lane_width=wp.lane_width,is_junction=wp.is_junction,pose=pose(wp.transform))


def physics(actor):
    if not actor.type_id.startswith('vehicle.'):
        return dict(kind='DefaultWalker_NoVehiclePhysicsControl',changed=False)
    pc=actor.get_physics_control()
    names=['mass','drag_coefficient','use_gear_autobox','gear_switch_time','clutch_strength','final_ratio',
           'max_rpm','moi','damping_rate_full_throttle','damping_rate_zero_throttle_clutch_engaged',
           'damping_rate_zero_throttle_clutch_disengaged','use_sweep_wheel_collision']
    d={k:getattr(pc,k) for k in names}
    d['center_of_mass']=xyz(pc.center_of_mass)
    d['wheels']=[{k:xyz(getattr(w,k)) if k=='position' else getattr(w,k) for k in
        ['tire_friction','damping_rate','max_steer_angle','radius','max_brake_torque','max_handbrake_torque','position']} for w in pc.wheels]
    d.update(full_repr=str(pc),changed=False)
    return d
