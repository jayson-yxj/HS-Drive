"""Fixed diagnostic actuator schedules; no normal Controller modification."""
import math


def cycle_command(interface,phase,elapsed):
    holding=phase in ['initialize','hold_initial','hold_warm','hold_final']
    running=phase in ['cold_run','warm_run']
    target=min(1.,elapsed*.5) if running else max(0.,1.-elapsed*.5) if phase in ['normal_stop','final_stop'] else 0.
    if phase=='initialize':return 'raw',dict(throttle=0.,brake=1.,steer=0.,hand_brake=True,reverse=False,manual_gear_shift=interface=='B',gear=1 if interface=='B' else 0),target
    if interface=='C':return 'ackermann',dict(speed=target,acceleration=.5,jerk=1.,steer=0.,steer_speed=0.),target
    return 'raw',dict(throttle=.15 if running else 0.,brake=.12 if holding and interface=='A' else .01 if phase in ['normal_stop','final_stop'] else 0.,steer=0.,hand_brake=holding and interface=='B',reverse=False,manual_gear_shift=interface=='B',gear=1 if interface=='B' else 0),target


def apply_command(vehicle,kind,command):
    import carla
    if kind=='raw':vehicle.apply_control(carla.VehicleControl(**command))
    else:vehicle.apply_ackermann_control(carla.VehicleAckermannControl(**command))


def serialize_physics(vehicle):
    from tools.m47_common import physics_dict,vector
    p=vehicle.get_physics_control();data=physics_dict(vehicle)
    # Fields absent from the old invariant summary retained here explicitly.
    extra={}
    for key in ['steering_curve','use_sweep_wheel_collision','center_of_mass']:
        assert key in data
    fields=[k for k in dir(p) if not k.startswith('_') and not callable(getattr(p,k))]
    represented=set(data)|{'wheels','forward_gears','torque_curve'}
    for k in fields:
        if k not in represented:
            v=getattr(p,k)
            if isinstance(v,(float,int,str,bool)):extra[k]=v
            else:extra[k]=str(v)
    return dict(invariant=data,extra=extra,wheel_world_positions=[vector(w.position) for w in p.wheels],api_fields=fields,
        invariant_reference='Wheel position is world-space, included separately and excluded only from invariant comparison SHA.')


def apply_variant(vehicle,config):
    """Exactly one allowed family; never mutate a default blueprint or global config."""
    p=vehicle.get_physics_control()
    if config['family']!='wheel_max_brake_torque':raise ValueError('Unsupported experimental family')
    wheels=p.wheels
    for w in wheels:w.max_brake_torque*=float(config['scale'])
    p.wheels=wheels;vehicle.apply_physics_control(p)
