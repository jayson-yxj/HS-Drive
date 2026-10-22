"""Compute independent acceptance decisions from frozen actual trial reports."""
import json,shutil,hashlib,subprocess
from pathlib import Path
from tools.m6b_common import write,provenance,asset


def load(p):return json.loads(Path(p).read_text())


def main():
    low=load('reports/m6b_low_speed_regression.json');maneuver=load('reports/m6b_oracle_maneuver_results.json');runtime=load('reports/m6b_runtime_tail.json');body=load('reports/m6b_recorded_body_audit.json');safety=load('reports/m6b_safety_regression.json')
    trials=maneuver['trials'];body_by={v['name']:v for v in body['trials']}
    bypass=[v for v in trials if v['config']['scene']=='bypass' and not v['config'].get('fault')]
    refused=[v for v in trials if v['config']['scene']=='blocked']
    removed=[v for v in trials if v['config']['scene']=='removed']
    normal=[v for v in trials if v['config']['scene'] in ['straight','mild_curve','junction_turn']]
    lead=[v for v in trials if v['config']['scene']=='lead']
    faults=[v for v in trials if 'fault_gate' in v]
    moving=[v for v in faults if v['fault_gate']['requires_motion_for_stop_evidence']]
    execution=[]
    for kind in ['stop_campaign','maneuver_campaign','runtime_campaign','supplement']:
        d=load(f'reports/m6b_{kind}_execution.json')
        execution.extend(d['runs'])
    original_failure=[r for r in execution if r['exit_code']]
    data=dict(**provenance('actual_CARLA_and_offline_independent_acceptance'),stage='M6-B',source='Privileged',status='Partial',
        trial_counts=dict(completed=safety['total_trials'],formal_attempts=len(execution),initialization_failures=len(original_failure),initialization_failure_logs=[r['log'] for r in original_failure],calibration_probes=6),
        low_speed=dict(status='Partial_candidate_NoGo_as_default',by_arm=low['by_arm'],
            final_C_failures=[v['name'] for v in low['trials'] if v['config']['arm']=='C' and not v['passed']],
            comfort='Failed_unchanged_raw_gates',thresholds=load('configs/m6b/protocol.json')['stop_goal']),
        normal_navigation=dict(status='Partial',strict_passed=sum(v['normal_route_strict_passed'] for v in normal),runs=len(normal),
            goal_passed=sum(v['passed'] for v in normal),no_lane_events=all(v['lane_events']==0 for v in normal),
            failures=[dict(name=v['name'],error_m=v['error_m'],completion=v['completion']) for v in normal if not v['normal_route_strict_passed']]),
        static_bypass=dict(status='Partial_limited_geometry_completion_only',runs=len(bypass),
            goal_and_body_passed=sum(v['passed'] and body_by[v['name']]['footprint_outside_frames']==0 and not body_by[v['name']]['OBB_intersections'] for v in bypass),
            comfort_passed=sum(v['comfort_pass'] for v in bypass),
            refusal_runs=len(refused),refusal_geometry_no_contact_passed=sum(v['refusal_geometry_and_collision_passed'] for v in refused),
            refusal_standstill_3m_passed=sum(v['standstill_gap_3m_passed'] for v in refused),
            refusal_actual_OBB_final_gaps={v['name']:body_by[v['name']]['terminal_XY_OBB_gap_m'] for v in refused},
            removed_recovery_passed=sum(v['removed_recovery']['moved'] and v['passed'] for v in removed),removed_runs=len(removed),
            curb_mesh='Unknown',dynamic_avoidance='NotValidated'),
        lead=dict(status='Partial',runs=len(lead),stopped=sum(v['lead_capabilities']['ego_stopped_during_lead_stop'] for v in lead),
            restarted=sum(v['lead_capabilities']['ego_moved_after_restart'] for v in lead),goal_passed=sum(v['passed'] for v in lead),
            standstill_gap='Not_certified_by_stopping_and_restart_alone'),
        fault_degradation=dict(status='Passed_limited_injected_fault_checks' if len(moving)==4 and all(v['fault_gate']['terminal_stopped'] and v['fault_gate']['no_collision'] and v['fault_gate']['no_body_boundary'] and v['fault_gate']['no_expired_normal_control'] and v['fault_gate']['brake_reaction_s'] is not None and v['fault_gate']['brake_reaction_s']<=.35 for v in moving) else 'Partial',
            stationary_injections=len(faults)-len(moving),moving_injections=len(moving),
            moving_results=[dict(name=v['name'],**v['fault_gate']) for v in moving]),
        safety=dict(collisions=safety['collisions'],illegal_lane_events=safety['illegal_lane_events'],expired_normal_control=safety['expired_normal_control'],
            recorded_body_outside_frames=sum(v['footprint_outside_frames'] for v in body['trials']),
            OBB_intersections=sum(len(v['OBB_intersections']) for v in body['trials']),
            original_6s_contract_retained=True,curb_status='Unknown_discrete_map_and_OBB_only'),
        runtime=dict(status='Partial_strict_20Hz_NotPassed',runs=len(runtime['trials']),
            max_ms=max(v['runtime']['control_interval_ms']['max'] for v in runtime['trials']),
            over55=sum(v['runtime']['over55'] for v in runtime['trials']),over100=sum(v['runtime']['over100'] for v in runtime['trials']),
            deadline_misses=sum(v['runtime']['deadline_miss'] for v in runtime['trials']),
            skipped_slots=sum(v['runtime']['skipped_wall_slots'] for v in runtime['trials']),
            causal_improvement='NotEstablished_short_affinity_repeats_no_UE_internal_profile',shared_visual_gpu='NotRun'),
        decisions=dict(A_low_speed='NoGo_default_replacement_or_reverse_parking',B_bypass='Go_limited_offline_and_Oracle_research_Partial_full_acceptance',
            C_control_cycle='NoGo_strict_realtime',D_oracle_engineering='Go_targeted_fault_and_low_speed_identification_only_NoGo_complex_traffic_or_parking',
            E_visual_safety='Unchanged_No_visual_control'),
        not_run=['reverse_bay_parking','PETR_asset_retry_inference_or_training','shared_visual_GPU_runtime','complex_dynamic_avoidance','traffic_right_of_way'],
        engineering_interfaces='Passed_contract_and_synthetic_checks_only',
        perception_gates='Unchanged_ResearchCompleted_CandidateNoGo_SafetyNotValidated',
        original_defaults_unchanged=True,physics_changes=False,
        tests=load('reports/m6b_tests.json'),preservation=load('reports/m6b_preservation.json'))
    write('reports/m6b_acceptance.json',data)
    # Archive exact commands/server logs and historical freeze in tracked paths.
    out=Path('reports/m6b_execution_logs');out.mkdir(exist_ok=False)
    for p in sorted(Path('runs/m6b_logs').iterdir()):
        if not p.is_file():continue
        dest=out/(p.name+'.txt' if p.suffix=='.log' else p.name)
        shutil.copy2(p,dest)
    # Every current artifact gets bytes and SHA; the manifest excludes itself.
    assets=[]
    for p in sorted(Path('reports').glob('m6b_*')):
        if p.is_file():assets.append(asset(p))
        else:assets.extend(asset(q) for q in sorted(p.rglob('*')) if q.is_file())
    write('reports/m6b_asset_manifest.json',dict(**provenance('SHA256_actual_artifacts'),assets=assets,archive_logs=str(out)))
    print(json.dumps({k:data[k] for k in ['trial_counts','low_speed','static_bypass','normal_navigation','runtime','fault_degradation']},indent=2))

if __name__=='__main__':main()
