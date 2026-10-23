"""Independent stage results, strict gates and explicit NotRun scope."""
import json,math
from pathlib import Path
import numpy as np
from tools.m6d_common import write,asset,provenance
from tools.m47_common import dynamics_metrics
from diagnostics.m6a_oracle_metrics import quantiles


def load(p):return json.loads(Path(p).read_text())

def inspect(p):
    d=load(p);r=[json.loads(s) for s in (p.parent/'trace.jsonl').read_text().splitlines()]
    v=np.array([q['signed']['v_forward_mps'] for q in r]);neg=v<-.05;events=int(np.sum(neg & ~np.r_[False,neg[:-1]]))
    t=np.array([q['sim_time'] for q in r]);interval=np.diff([q['timeline']['apply_control'] for q in r])*1000
    reasons=[q for q in r if q['reason'] and q['measurement']['speed']>.2]
    obs=[q for q in r if q['geometry']['obstacle_obb_clearance_m'] is not None]
    moving_normal=[q['measurement'] for q in r if not q['reason']]
    # Do NOT concatenate disjoint normal segments and differentiate across gaps.
    segments=[];current=[]
    for q in r:
        if not q['reason']:current.append(q['measurement'])
        else:
            if len(current)>=8:segments.append(dynamics_metrics(current))
            current=[]
    if len(current)>=8:segments.append(dynamics_metrics(current))
    normal_comfort=bool(segments and all(x['max_decel_raw']<=1.8 and x['jerk_peak_raw']<=5 and x['jerk_p95_moving']<=2.5 for x in segments))
    last=r[-21:];actual_final_gap=obs[-1]['geometry']['obstacle_obb_clearance_m'] if obs else None
    summary=dict(name=p.parent.name,config=d['config'],trace=asset(p.parent/'trace.jsonl'),result=asset(p),physics=d['physics'],
        goal_completed=d['goal_passed'],goal_error_m=d['final_goal_error_m'],signed_goal_remaining_m=r[-1]['stop_target']['goal_signed_remaining_m'],
        normal_stop_passed=bool(d['goal_passed'] and normal_comfort and not reasons),emergency_intervention=bool(reasons),
        actual_final_gap=actual_final_gap,minimum_gap_during_motion=min((q['geometry']['obstacle_obb_clearance_m'] for q in obs if q['measurement']['speed']>.02),default=None),
        rollback_events=events,rollback_ticks=int(neg.sum()),rollback_duration_s=float(neg.sum()*.05),minimum_signed_speed=float(v.min()),
        recovery_attempts=sum(q['command'].get('transition_reason') in ['target_advanced','valid_plan_recovered'] for q in r),
        hold_evidence_stable_ticks=sum(q['command'].get('hold_evidence_mode')=='HOLD_STABLE' for q in r),
        terminal_stable_1s=len(last)==21 and all(q['measurement']['speed']<=.08 for q in last),
        collisions=d['collisions'],illegal_lane_events=d['illegal_lane_events'],boundary_ticks=sum(q['boundary'] for q in r),
        raw_all=d['dynamics'],normal_contiguous_segments=segments,normal_comfort_passed=normal_comfort,
        original_raw_comfort_passed=d['comfort_pass'],degradation=d['degradation'],
        runtime={k:d[k] for k in ['control_interval_ms','over50','over55','over100','deadline_miss','skipped_wall_slots','mean_hz','expired_normal_control','world_tick_ms','controller_ms']},
        stages_ms={name:quantiles([(q['timeline'][b]-q['timeline'][a])*1000 for q in r]) for name,a,b in [('snapshot','wake_up','snapshot_end'),('IPC','snapshot_end','submit_poll_end'),('safety','submit_poll_end','safety_end'),('controller_and_geometry','safety_end','controller_end')]},
        plan_age_wall=quantiles([q['plan_wall_age'] for q in r if q['plan_wall_age'] is not None]),
        plan_age_sim=quantiles([q['plan_sim_age'] for q in r if q['plan_sim_age'] is not None]),
        actual_travel_m=float(np.linalg.norm(np.diff([q['measurement']['position'] for q in r],axis=0),axis=1).sum()))
    fault=d['config']['fault']
    if fault:
        index=d.get('fault_tick_actual',d['config']['fault_tick'])
        q=r[index] if index is not None and index<len(r) else None
        summary['fault']=dict(type=fault,actual_tick=index,injection_speed=None if q is None else q['measurement']['speed'],valid_moving_injection=bool(q and q['measurement']['speed']>=1.),response_s=d['fault_to_brake_s'],final_stopped=summary['terminal_stable_1s'])
    return summary,r


def main():
    pairs=[inspect(p) for p in sorted(Path('reports/m6d_trials').glob('*/result.json'))];results=[d for d,r in pairs]
    short=[d for d in results if d['config']['scene']=='stop'];faults=[d for d in results if 'fault' in d]
    arms={a:dict(trials=sum(q['config']['arm']==a for q in short),goal_completed=sum(q['config']['arm']==a and q['goal_completed'] for q in short),normal_stop_passed=sum(q['config']['arm']==a and q['normal_stop_passed'] for q in short),rollback_events=sum(q['rollback_events'] for q in short if q['config']['arm']==a),comfort_passed=sum(q['config']['arm']==a and q['original_raw_comfort_passed'] for q in short)) for a in ['C','D','F0']}
    write('reports/m6d_controller_ablation.json',dict(**provenance('actual_CARLA_development_fixed_C_D2_F0_comparison'),status='Partial',arms=arms,trials=short,F1='NotRun_physical_prerequisites_failed',
        limitation='Different actual simulator/OS executions; F0 normal command equality proven with identical-input unit sequence. Paired routes do not imply bitwise identical vehicle trajectories. D names mean frozen M6-C D2.'))
    guard=[]
    for p in sorted(Path('reports/m6d_probe_artifacts').glob('guard_diagnostic*/result.json')):
        d=load(p);r=[json.loads(s) for s in (p.parent/'trace.jsonl').read_text().splitlines()];i=next((i for i,q in enumerate(r) if q['protection']),None)
        release_rows=[q for q in r if q['phase']=='release']
        positions=np.array([q['measurement']['position'][:2] for q in release_rows])*[1,-1]
        yaw=release_rows[0]['signed']['yaw_rad'];projected=(positions-positions[0])@np.array([math.cos(yaw),math.sin(yaw)])
        guard.append(dict(release_min_pose_progress_m=float(projected.min()),pose_backward_segments=int((np.diff(projected)<-.0001).sum()),name=p.parent.name,trace=asset(p.parent/'trace.jsonl'),physics=d['physics'],
            protection_triggered=i is not None,trigger=None if i is None else dict(frame=r[i]['frame'],signed=r[i]['signed'],command=r[i]['command'],phase=r[i]['phase']),
            immediate_brake=bool(i is not None and r[i]['command']['brake']==1 and r[i]['command']['throttle']==0),
            final_stopped_1s=all(q['signed']['speed_norm_mps']<=.02 for q in r[-21:]),
            min_signed_speed=min(q['signed']['v_forward_mps'] for q in r),
            release_max_rollback_m=sum(max(0,-q['signed']['v_forward_mps'])*.05 for q in r if q['phase'] in ['release','protected_stop']),
            raw=d['raw_all_including_initialize'],source='Privileged',evaluation='actual_CARLA_existing_guard_emergency_probe'))
    write('reports/m6d_rollback_regression.json',dict(**provenance('actual_CARLA_and_separately_offline_rollback_evidence'),status='NoGo_intermittent_negative_motion_not_protected',
        actual_guard_probes=guard,actual_short_routes=short,historical_replay=asset('reports/m6d_historical_replay.json'),
        natural_cold_release='4 actual probes, no rollback observed; not counted as successful rollback recovery',
        limitation='Warm diagnostic protection is measured; automatic rollback release is not implemented. Latched F0 stop needs new task/session, not demonstrated autonomous recovery.'))
    notrun=dict(status='NotRun',reason='F1 failed hold/restart/comfort prerequisites; no sealed or large route campaign',
        cases=['25m','new_independent_roads','blocked_p126','blocked_p170','legal_bypass','obstacle_removed','lead_stop_restart','sudden_infeasible'])
    write('reports/m6d_stop_safety_comparison.json',dict(**provenance('actual_CARLA_limited_development_safety_not_F1_qualification'),status='Partial',trials=results,faults=faults,conditional_regression=notrun,
        historical_M6C=asset('reports/m6c_acceptance.json'),historical_quality_unchanged=True,
        strict_3m_rule_unchanged=True,normal_requires_comfort_no_emergency=True,
        collisions=sum(len(d['collisions']) for d in results),illegal_solid_events=sum(len(d['illegal_lane_events']) for d in results),
        curb_mesh='Unknown_no_exact_mesh_audit',body_road_limit='Inherited online corner guard and final dense body check; no new continuous contact proof.'))
    worst=max(results,key=lambda d:d['runtime']['control_interval_ms']['max'])
    write('reports/m6d_runtime_status.json',dict(**provenance('actual_CARLA_paced_development_runtime_unaltered_safety_contract'),status='Partial_strict20Hz_NotPassed',
        trials=results,all_max_ms=worst['runtime']['control_interval_ms']['max'],worst_name=worst['name'],
        deadline_miss=sum(d['runtime']['deadline_miss'] for d in results),skipped_slots=sum(d['runtime']['skipped_wall_slots'] for d in results),
        expired_normal=sum(d['runtime']['expired_normal_control'] for d in results),over100=sum(d['runtime']['over100'] for d in results),
        fixed_contract=dict(planner_hz=5,controller_hz=20,max_age_s=.35,horizon_s=6),
        actuator_probes='Unpaced fixed simulation dt; not included as 20Hz driving samples',
        history=asset('reports/m6c_runtime_tail.json'),limits='No affinity search, no reset epoch, no burst catchup. Localized timing only, no internal UE/OS causality claim. No visual GPU load.'))
    print(json.dumps(dict(arms=arms,guard=[{k:v for k,v in x.items() if k not in ['trigger','trace','physics','raw']} for x in guard],runtime_max_ms=worst['runtime']['control_interval_ms']['max'],faults=[x['fault'] for x in faults]),indent=2))
if __name__=='__main__':main()
