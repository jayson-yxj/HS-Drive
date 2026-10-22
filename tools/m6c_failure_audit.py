"""Read-only reconstruction of four frozen M6-B failures using recorded snapshots."""
import json,math
from pathlib import Path
import numpy as np,carla
from tools.m6c_common import write,provenance,asset
from simulator.m6b_motion import StopGoal,goal_route,LowSpeedController
from simulator.motion_oracle import OracleState,MotionPlanner
from simulator.m6c_geometry import body_box,stopping_geometry,road_grade


def main():
    cmap=carla.Map('m6c_historical',Path('reports/m6b_execution_logs/Town03.xodr').read_text())
    body=json.loads(Path('reports/m6b_body_reference.json').read_text());events=[]
    names=['final_stop_p170_l9_s101_C','final_stop_p170_l9_s102_C','final_straight_s101','final_mild_curve_s101','final_junction_turn_s101','final_blocked_p126_s102','final_blocked_p126_s101_fixture_retry']
    for name in names:
        root=Path('reports/m6b_trials')/name;r=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()];route=np.array(json.loads((root/'route.json').read_text())['selected']);g=json.loads((root/'goal.json').read_text());goal=StopGoal(**{k:g[k] for k in StopGoal.__dataclass_fields__})
        speed=np.array([q['measurement']['long_speed'] for q in r]);t=np.array([q['sim_time'] for q in r]);acc=np.r_[0,np.diff(speed)/np.diff(t)];jerk=np.r_[0,np.diff(acc)/np.diff(t)]
        records=[];planner=MotionPlanner();planning=goal_route(route,goal)
        for i,q in enumerate(r):
            m=q['measurement'];st=OracleState(m['position'][0],-m['position'][1],-math.radians(m['rotation'][2]),m['speed'],q['sim_time'])
            geom=stopping_geometry(route,st,goal,body_box(m,body),q['obstacles']);_,meta=planner.plan(planning,st,q['obstacles'],half_length=body['extent'][0],half_width=body['extent'][1])
            records.append(dict(frame=q['frame'],timestamp=q['sim_time'],signed_speed=speed[i],speed=m['speed'],raw_acceleration=acc[i],raw_jerk=jerk[i],
                geometry=geom,road=road_grade(cmap,m['position']),vehicle_pitch=m['rotation'][1],command=q['command'],applied=m['applied'],
                predicted_coast_distance=LowSpeedController.coast_distance(m['speed']),predicted_brake_distance=None,
                predicted_brake_reason='historical_C_has_no_validated_brake_distance_model',planner_available_recomputed=meta['remaining_m'],
                planner_original_timestamp=None if q['plan_sim_age'] is None else q['sim_time']-q['plan_sim_age'],plan_wall_age=q['plan_wall_age'],safety_reason=q['reason'],source='Privileged'))
        path=Path('reports/m6c_failure_events')/(name+'.jsonl');path.parent.mkdir(exist_ok=True)
        with path.open('x') as f:
            for row in records:f.write(json.dumps(row)+'\n')
        events.append(dict(name=name,trace=asset(root/'trace.jsonl'),audit=asset(path),terminal=records[-1],
            first_coast=next((v for v in records if v['command'].get('mode')=='SETTLE'),None),
            first_moving_safety=next((v for v in records if v['safety_reason'] and v['speed']>.2),None)))
    runtime=json.loads(Path('reports/m6b_runtime_full_campaign.json').read_text());worst=max(runtime['trials'],key=lambda v:v['runtime']['control_interval_ms']['max']);w=worst['worst_interval'];p=w['previous'];c=w['current']
    parts=dict(previous_apply_to_tick=(p['tick_start']-p['apply_control'])*1000,previous_tick=(p['tick_end']-p['tick_start'])*1000,
        record_tail=(c['previous_sleep_start']-p['tick_end'])*1000,sleep_wakeup=(c['wake_up']-c['previous_sleep_start'])*1000,current_preapply=(c['apply_control']-c['wake_up'])*1000)
    write('reports/m6c_goal_geometry_regression.json',dict(**provenance('offline_reconstruction_of_actual_frozen_M6B'),source='Privileged',events=events,
        historical_worst=dict(name=worst['name'],interval_ms=w['ms'],parts_ms=parts,reconstruction_error_ms=abs(sum(parts.values())-w['ms']),original=w),
        findings=['Goal and auxiliary1m semantics agree; signed endpoint overshoot remains negative. No target shift proposed.',
          '3m is subtracted once from physical OBB gap; frozen planner uses route-arc gap based on actor origin, differs by local OBB offset and curvature.',
          'Road126 residual safety gap cannot be explained by ~3cm ego OBB offset; coast model/execution dominates metre-scale loss.',
          'Raw motion/command/grade association is not causal proof of Gear-only mechanism.']))

if __name__=='__main__':main()
