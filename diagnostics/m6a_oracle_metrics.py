"""Offline metrics preserve failure denominators and the CARLA coordinate flip."""
from collections import Counter
import numpy as np
from simulator.motion_oracle import route_projection


def quantiles(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError('nonfinite measurement')
    return dict(n=len(a), p50=float(np.quantile(a, .5)), p95=float(np.quantile(a, .95)),
                p99=float(np.quantile(a, .99)), max=float(a.max()))


def analyze(rows, plans, route, result):
    if len(rows) < 2 or not all(r.get('source') == 'Privileged' for r in rows):
        raise ValueError('need actual privileged trace')
    # Measurement position is native CARLA world (x right-handed conversion y=-y).
    xy = np.array([[r['measurement']['position'][0], -r['measurement']['position'][1]] for r in rows])
    cte = [route_projection(route, p)[3] for p in xy]
    intervals = np.diff([r['control_apply_wall'] for r in rows])*1000
    sim_dt = np.diff([r['sim_time'] for r in rows])
    reasons = Counter(r['reason'] for r in rows if r['reason'])
    errors = sum(bool(p.get('error')) or p.get('reason') is not None for p in plans)
    stale = sum(r['reason'] in ('plan_expired', 'trajectory_timeout') for r in rows)
    rejected_stale = sum(r.get('reason') == 'plan_expired' for r in result['rejected'])
    speed = [r['measurement']['speed'] for r in rows]
    checks = dict(completion=result['route_completion'] >= .95,
                  no_collision=result['collisions'] == 0,
                  no_lane_invasion=result['lane_invasions'] == 0,
                  no_road_boundary=result['road_boundary_frames'] == 0,
                  cross_track_p95=np.quantile(cte,.95) <= .5,
                  no_expired_normal_control=result['normal_control_expired'] == 0,
                  stopped=speed[-1] <= .08)
    faults = ('emergency','stalled_worker','worker_crash','invalid_input')
    gaps = []
    for r, point in zip(rows, xy):
        for o in r.get('obstacles', []):
            # A route arc proxy, NOT exact 3D bumper or swept mesh clearance.
            s, _, _, _ = route_projection(route, point)
            so, _, _, _ = route_projection(route, o['xy'])
            gaps.append(so-s-o['half_length']-2.4)
    return dict(
        source='Privileged', scene=result['scene'], ticks=len(rows),
        route_completion=result['route_completion'], measured_travel_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
        cross_track_m=quantiles(cte), terminal_speed_mps=float(speed[-1]),
        collisions=result['collision_events'], lane_invasions=result['lane_events'],
        curb_contact_status='Unknown_no_dedicated_curb_contact_reference',
        road_boundary_frames=result['road_boundary_frames'],
        route_success_checks={k:bool(v) for k,v in checks.items()},
        route_success=bool(all(checks.values())),
        task_status=('Passed_limited_fault_stop' if result['scene'] in faults and
                     speed[-1] <= .08 and not result['collisions'] and not result['normal_control_expired']
                     and result['fault_to_brake_s'] is not None and result['fault_to_brake_s'] <= .4
                     else ('Passed_limited_route' if all(checks.values()) else 'Partial')),
        planner_outputs=len(plans), planner_failure_or_safety_rejection_count=errors,
        planner_failure_or_safety_rejection_rate=errors/len(plans) if plans else None,
        planner_reasons=dict(Counter(p.get('reason') for p in plans if p.get('reason'))),
        control_interval_ms=quantiles(intervals), control_intervals_over_50ms=int((intervals>50).sum()),
        control_intervals_over_55ms=int((intervals>55).sum()),
        completion_deadline_misses=sum(r['deadline_missed'] for r in rows),
        sim_step_ms=quantiles(sim_dt*1000), mean_wall_hz=result['wall_hz'],
        controller_timeout_count=sum(r['command'].get('reason') == 'controller_timeout' for r in rows),
        stale_plan_control_ticks=stale, stale_received_plan_count=rejected_stale,
        expired_normal_control=result['normal_control_expired'],
        degradation_ticks=dict(reasons), degradation_episodes=sum(bool(r['reason']) and (i==0 or rows[i-1]['reason']!=r['reason']) for i,r in enumerate(rows)),
        fault_to_brake_s=result['fault_to_brake_s'],
        planner_ms=result['planner_ms'], plan_sim_age_s=result['plan_sim_age_s'],
        plan_wall_age_s=result['plan_wall_age_s'], stages_ms=result['stage_profile_ms'],
        min_route_arc_gap_proxy_m=min(gaps) if gaps else None,
        strict_20hz=bool((intervals<=50).all()), raw=result['log'], raw_sha256=result['log_sha256'])
