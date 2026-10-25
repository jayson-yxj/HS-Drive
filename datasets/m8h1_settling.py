"""Predeclared stationary-window decision; no smoothing or physics modification."""
import numpy as np

def stationary(trace, cfg):
    if len(trace)<cfg['min_ticks'] or len(trace)<cfg['window_ticks']:return dict(stable=False,reason='InsufficientWindow')
    rows=trace[-cfg['window_ticks']:];pos=np.array([r['pose']['xyz'] for r in rows]);angles=np.array([r['pose']['rotation'] for r in rows]);angles=np.degrees(np.unwrap(np.radians(angles),axis=0));speed=max(float(np.linalg.norm(r['velocity_mps'])) for r in rows)
    span=float(np.linalg.norm(np.ptp(pos,axis=0)));angle=float(np.ptp(angles,axis=0).max())
    good=speed<=cfg['max_speed_mps'] and span<=cfg['max_window_position_span_m'] and angle<=cfg['max_window_angle_span_deg']
    return dict(stable=bool(good),max_speed_mps=speed,window_position_span_m=span,window_angle_span_deg=angle,window_ticks=len(rows),raw_no_filter=True)

def drift(initial, final, cfg):
    xy=float(np.linalg.norm(np.array(initial['xyz'][:2])-final['xyz'][:2]));z=float(final['xyz'][2]-initial['xyz'][2]);angle=float(np.abs((np.array(final['rotation'])-initial['rotation']+180)%360-180).max())
    return dict(xy_m=xy,z_m=z,angle_deg=angle,passed=bool(xy<=cfg['max_native_xy_drift_m'] and angle<=cfg['max_final_rotation_delta_deg']),vertical_settling_not_static_transport=True)
