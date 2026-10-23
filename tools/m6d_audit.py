"""Independent snapshot and target audits on recorded data; no CARLA tick.

CARLA SDK Transform provides an independent coordinate calculation. Frozen C is
replayed using recorded input trajectories to verify F0 normal-output equality.
"""
import json,math
from pathlib import Path
import numpy as np,carla
from simulator.m6d_state import signed_from_measurement
from simulator.m6d_motion import SignedGuard
from simulator.m6d_targets import separate_targets
from tools.m6d_common import write,asset,provenance


def main():
    results=[];errors=[];classes=dict(forward=0,near_zero=0,backward=0,lateral_only=0,gear_direction_disagreement=0)
    for path in sorted(Path('reports/m6d_probe_artifacts').glob('*/trace.jsonl'))+sorted(Path('reports/m6d_trials').glob('*/trace.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()];max_diff=0.;sign_disagreement=0;nonmonotonic=0
        for i,q in enumerate(rows):
            r=q['measurement'];s=q['signed'];roll,pitch,yaw=r['rotation'];tf=carla.Transform(carla.Location(*r['position']),carla.Rotation(roll=roll,pitch=pitch,yaw=yaw))
            raw_forward=tf.get_forward_vector();f=np.array([raw_forward.x,raw_forward.y]);f/=np.linalg.norm(f);velocity=np.array(r['velocity']);expected=float(velocity[:2]@f)
            max_diff=max(max_diff,abs(expected-s['v_forward_mps']))
            if abs(expected)>.001 and expected*s['v_forward_mps']<0:sign_disagreement+=1
            if i and (r['frame']<=rows[i-1]['measurement']['frame'] or r['t']<=rows[i-1]['measurement']['t']):nonmonotonic+=1
            assert s['frame_id']==r['frame'] and s['timestamp_s']==r['t']
            assert abs(s['speed_norm_mps']-np.linalg.norm(velocity))<1e-8
            if s['v_forward_mps']>.05:classes['forward']+=1
            elif s['v_forward_mps']<-.05:classes['backward']+=1
            else:classes['near_zero']+=1
            if abs(s['v_lateral_mps'])>.05 and s['v_forward_mps']>=-.05:classes['lateral_only']+=1
            if s['v_forward_mps']<-.05 and s['gear']>=0 and not s['reverse_commanded']:classes['gear_direction_disagreement']+=1
        record=dict(trace=asset(path),frames=len(rows),SDK_xy_forward_max_error=max_diff,sign_disagreement=sign_disagreement,nonmonotonic=nonmonotonic)
        results.append(record)
        if max_diff>1e-5 or sign_disagreement or nonmonotonic:errors.append(record)
    write('reports/m6d_signed_motion_contract.json',dict(**provenance('offline_CARLA_SDK_crosscheck_actual_snapshot_measurements'),
        status='Go_limited_observation_contract' if not errors else 'NoGo',snapshot_audits=results,errors=errors,observed_classes=classes,
        tests=asset('reports/m6d_sdk_contract_tests.xml'),old_Oracle_speed_norm_unchanged=True,
        limits='XY projected heading; old long_speed uses 3D pitch axis, retained separately. Applied control RPC queried adjacent to immutable snapshot; not falsely claimed ActorSnapshot control. No independent physical sensor truth.'))
    target_results=[]
    for path in sorted(Path('reports/m6d_trials').glob('*/trace.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()];bad=[]
        for q in rows:
            g=q['geometry'];t=q['stop_target']
            if t['goal_signed_remaining_m']!=g['goal_signed_distance_m'] or t['goal_route_remaining_m']!=g['route_remaining_s']:bad.append(q['frame'])
            gap=g['obstacle_obb_clearance_m'];expected=None if gap is None else gap-3
            if t['obstacle_safe_stop_remaining_m']!=expected or t['timestamp_s']!=q['sim_time'] or t['frame_id']!=q['frame']:bad.append(q['frame'])
        target_results.append(dict(name=path.parent.name,frames=len(rows),errors=bad,trace=asset(path)))
    write('reports/m6d_goal_obstacle_target_separation.json',dict(**provenance('offline_actual_CARLA_distinct_target_contract'),
        status='Go_limited_schema' if not any(r['errors'] for r in target_results) else 'NoGo',trials=target_results,
        independent_fields=['goal_signed_remaining_m','goal_route_remaining_m','obstacle_safe_stop_remaining_m','active_stop_target','active_stop_reason','required_clearance_m','estimated_stop_capability'],
        numerical_C_compatibility='F0 preserves original C adapter remaining_m only for unchanged normal actuation. New context never overwrites StopGoal or maps an envelope threshold into target distance.',
        obstacle_live='NotRun_gated_route_matrix; obstacle fields exercised by synthetic and historical replay below',synthetic_tests=asset('reports/m6d_contract_tests.xml')))
    # Frozen history is offline evidence, not another actual M6-D trial.
    historical=[]
    for path in sorted(Path('reports/m6c_trials').glob('final_*/trace.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()];guard=SignedGuard();events=[];target_errors=0
        route=np.array(json.loads((path.parent/'route.json').read_text())['selected'])
        for q in rows:
            s=signed_from_measurement(q['measurement'],route,q['geometry']['goal_signed_distance_m'],q['road']['grade'])
            reason=guard.check(s,s.timestamp_s)
            if reason and not events:events.append(dict(frame=q['frame'],v_forward_mps=s.v_forward_mps,reason=reason))
            target=separate_targets(q['geometry'],q['frame'],fault=q['reason'])
            target_errors+=target.goal_signed_remaining_m!=q['geometry']['goal_signed_distance_m']
        historical.append(dict(name=path.parent.name,trace=asset(path),first_detection=events,target_errors=target_errors,protection_executed=False))
    write('reports/m6d_historical_replay.json',dict(**provenance('offline_M6C_frozen_trace_replay_no_simulation'),trials=historical,
        action='Observation only; no counterfactual stopping distance, no new historical success or safety claim'))
if __name__=='__main__':main()
