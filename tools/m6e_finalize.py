"""Separate observed engineering validity, physical feasibility and safety decisions."""
import json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from tools.m6e_common import write,asset,provenance
from tools.m6e_analyze import inspect
from simulator.recorder import file_hash


def load(path):return json.loads(Path(path).read_text())

def main():
    allr=[inspect(p) for p in sorted(Path('reports/m6e_trials').glob('*/result.json'))]
    p0=[r for r in allr if r['config']['physics']=='P0' and r['config']['mode']=='cycle'];p1=[r for r in allr if r['config']['physics']=='P1'];guards=[r for r in allr if r['config']['mode']=='guard']
    replay=load('reports/m6e_guard_historical_replay.json')
    groups={}
    for split in ['development','validation']:
        groups[split]={}
        for a in ['A','B','C']:
            rs=[r for r in p0 if r['config']['split']==split and r['config']['interface']==a]
            groups[split][a]=dict(trials=len(rs),raw_comfort_passed=sum(r['normal_comfort_passed'] for r in rs),
                cold_delays_s=[r['phases']['cold_run']['start_delay_s'] for r in rs],warm_delays_s=[r['phases']['warm_run']['start_delay_s'] for r in rs],
                hold_passed=sum(all(r['phases'][p]['hold_passed'] for p in ['hold_initial','hold_warm','hold_final']) for r in rs),
                normal_brake_max_decel=[r['phases']['normal_stop']['raw']['max_decel_raw'] for r in rs],
                normal_brake_jerk_peak=[r['phases']['normal_stop']['raw']['jerk_peak_raw'] for r in rs],
                normal_brake_initial_speeds=[r['phases']['normal_stop']['initial_speed'] for r in rs],
                startup_jerk_peak=[r['phases']['cold_run']['raw']['jerk_peak_raw'] for r in rs])
    write('reports/m6e_actuation_interface_comparison.json',dict(**provenance('actual_CARLA_P0_actuation_policy_comparison'),groups=groups,trials=p0,
        comparison_limit='A/B vs C different policies and actual speed distributions. Raw decel reductions not strict matched-initial-speed braking superiority; not physical single-variable ablation.',
        source_reference='reports/m6e_execution_logs/AckermannController_0.9.15.cpp.txt: TargetJerk assigned but not consumed in official control loop; server gains read unchanged. Actual binary not instrumented, no enforced jerk claim.'))
    bases=[r for r in allr if r['config']['physics']=='P0']
    versions={version:sorted({r['physics']['sha256'] for r in allr if r['config']['physics']==version}) for version in ['P0','P1']}
    assertions=[]
    for r in allr:
        root=Path(r['result']['path']).parent;before=load(root/'physics_original.json');after=load(root/'physics.json');final=load(root/'physics_final.json')
        expected=json.loads(json.dumps(before))
        if r['config']['physics']=='P1':
            for w in expected['wheels']:w['max_brake_torque']*=.5
        assertions.append(dict(name=r['name'],exact_expected_readback=expected==after,unchanged_during_trial=after==final,
            original_sha=asset(root/'physics_original.json')['sha256'],full=asset(root/'physics_full.json'),invariant=r['physics']))
    write('reports/m6e_physics_baseline.json',dict(**provenance('actual_CARLA_spawn_readback_and_trial_end_verification'),label='Physics-P0 / FrozenBaseline',versions=versions,
        blueprint='vehicle.tesla.model3',carla='0.9.15',map='Town03',readbacks=assertions,
        default_unchanged=all(a['exact_expected_readback'] and a['unchanged_during_trial'] for a in assertions),
        full_serialization_limit='Public Python physics fields, including wheel world positions separately. Internal tire contacts/solver states unavailable. Invariant signature excludes only world-space wheel positions.'))
    paired=[]
    for r in p1:
        c=r['config'];ref=next((x for x in allr if x['config']['physics']=='P0' and all(x['config'][k]==c[k] for k in ['spawn','seed','interface','mode','split'])),None)
        assert ref is not None,r['name']
        phase='normal_stop' if c['mode']=='cycle' else 'diagnostic_action';a=ref['phases'][phase];b=r['phases'][phase]
        delta=None if a['stopped_travel_m'] is None or b['stopped_travel_m'] is None else b['stopped_travel_m']-a['stopped_travel_m']
        dt=None if a['stop_duration_s'] is None or b['stop_duration_s'] is None else b['stop_duration_s']-a['stop_duration_s']
        paired.append(dict(P0=ref['name'],P1=r['name'],mode=c['mode'],split=c['split'],spawn=c['spawn'],seed=c['seed'],
            P0_initial_speed=a['initial_speed'],P1_initial_speed=b['initial_speed'],initial_speed_delta=b['initial_speed']-a['initial_speed'],
            metrics_P0=a,metrics_P1=b,stopping_distance_delta_m=delta,stopping_time_delta_s=dt,
            emergency_noninferiority=None if c['mode'] not in ['emergency','obstacle'] else bool(delta is not None and dt is not None and delta<=.25 and dt<=.25),
            gap_P0=ref['obstacle'],gap_P1=r['obstacle'],no_collision=not r['collisions'],body_road=r['final_body_road']))
    write('reports/m6e_physics_variant.json',dict(**provenance('actual_CARLA_single_wheel_brake_family_intervention'),label='Physics-P1-Experimental',status='Partial_research_variant_not_promoted',
        declaration=asset('configs/m6e/physics_P1.json'),versions=versions,readback_verified=all(a['exact_expected_readback'] for a in assertions),paired=paired,trials=p1,
        limitation='Simulation research variant only, no external real-Tesla validation. Applying PhysicsControl reconstructs vehicle state; no sham reapply group, so effect cannot uniquely exclude recreation effects. Original P0 actual signatures checked on all fresh spawns; original restored before destruction.'))
    write('reports/m6e_startup_braking_results.json',dict(**provenance('offline_unfiltered_actual_CARLA_derivatives'),groups=groups,trials=allr,
        gates=dict(max_decel_raw=1.8,jerk_peak_raw=5,moving_jerk_p95=2.5),emergency_is_not_comfort=True,
        normal_metrics='Derivative across phase boundary retains preceding2 samples; no startup deletion. All-phase and emergency raw separately retained.',
        simple_brake_torque_check=dict(P0_static_nominal_mps2=4*700*.01/(1845*.37),P1_static_nominal_mps2=4*350*.01/(1845*.37),
            inference='Simplified no-slip wheel torque/mass/radius only, not a complete dynamics model; mismatch means max torque alone is not proven cause.')))
    holds=[]
    for r in allr:
        hold={k:p for k,p in r['phases'].items() if k.startswith('hold') or k=='guard_hold'}
        if hold:holds.append(dict(name=r['name'],physics=r['config']['physics'],holds=hold,restarts={k:p for k,p in r['phases'].items() if k in ['warm_run','guard_release']},trace=r['trace']))
    write('reports/m6e_hold_restart_results.json',dict(**provenance('actual_CARLA_hold_and_release_measurements'),trials=holds,
        limit='Hold near measured pose is not arrival at navigation StopGoal. No independent wheel contact forces; roll/pitch/pose/API velocity are proxies.'))
    live=[dict(name=r['name'],split=r['config']['split'],**r['rollback_protection'],trace=r['trace'],phases=r['phases']) for r in guards]
    # Live quiet hold false positives, and rollout risk counts outside explicit
    # protection trials: do not declare every event false without pose audit.
    quiet=[];events=[];displacements=[]
    for r in allr:
        rows=[json.loads(s) for s in Path(r['trace']['path']).read_text().splitlines()]
        for phase in dict.fromkeys(q['phase'] for q in rows):
            q=[x for x in rows if x['phase']==phase]
            if phase.startswith('hold') or phase=='guard_hold':
                if max(x['signed']['speed_norm_mps'] for x in q)<=.02:
                    first_index=rows.index(q[0]);prior=rows[first_index-1]['guard'] if first_index else dict(risk=False,emergency=False)
                    new_risk=new_emergency=0
                    for sample in q:
                        new_risk+=bool(sample['guard']['risk'] and not prior['risk'])
                        new_emergency+=bool(sample['guard']['emergency'] and not prior['emergency'])
                        prior=sample['guard']
                    quiet.append(dict(name=r['name'],phase=phase,n=len(q),risk_ticks=sum(x['guard']['risk'] for x in q),emergency_ticks=sum(x['guard']['emergency'] for x in q),new_risk_events=new_risk,new_emergency_events=new_emergency,
                        explanation='Already latched risk/emergency from preceding measured rollback is not a new quiet-hold false trigger.'))
        em=next((i for i,x in enumerate(rows) if x['protection_applied']),None)
        if em is not None:
            ix=max(0,em-8);q=rows[ix:em+6];pose=np.array([x['measurement']['position'][:2] for x in q])*[1,-1];yaw=q[0]['signed']['yaw_rad'];proj=(pose-pose[0])@np.array([math.cos(yaw),math.sin(yaw)])
            displacements.append(dict(name=r['name'],event=rows[em]['guard'],samples=q,pose_longitudinal_relative_m=proj.tolist(),
                actual_stop_observed=r['rollback_protection']['final_stopped']))
        event_path=Path(r['trace']['path']).parent/'guard_events.json';events.append(dict(name=r['name'],mode=r['config']['mode'],events=load(event_path)))
    write('reports/m6e_signed_displacement_audit.json',dict(**provenance('actual_CARLA_signed_pose_velocity_event_windows'),events=displacements,
        historical_regression=asset('reports/m6e_guard_historical_replay.json'),reference='Raw CARLA pose and velocity, XY forward projection; route-forward tracked independently. Pose quantization and finite .05s samples remain.'))
    write('reports/m6e_rollback_guard_validation.json',dict(**provenance('synthetic_offline_and_actual_CARLA_guard_separate_evidence'),
        config=asset('configs/m6e/protocol.json'),known_failure_replay=dict(detected=replay['known_failure_detection'],total=replay['known_failures']),
        historical_quiet_segments=replay['stationary_control_segments'],historical_quiet_false=replay['stationary_false_decisions'],
        live=live,quiet_hold_controls=quiet,all_event_logs=events,
        unit_tests=asset('reports/m6e_guard_fixed_tests.xml'),limits='No global false-positive estimate: quiet controls selected on measurements and few fixed locations. Guard emergency latched, recovery not automatic. Normal starts with true backward motion are positive risk, not automatically false alarms.'))
    safety_trials=[r for r in allr if r['config']['mode'] in ['guard','emergency','obstacle','coast']]
    write('reports/m6e_safety_regression.json',dict(**provenance('actual_CARLA_finite_actuator_safety_diagnostics'),trials=safety_trials,
        paired_physics=paired,collisions=sum(len(r['collisions']) for r in allr),boundary_ticks=sum(r['boundary_ticks'] for r in allr),
        conditional_full_closed_loop=dict(status='NotRun',reason='Original raw dynamics prerequisites failed; no new Oracle candidate',
            cases=['9m','25m','lead_stop_restart','legal_bypass','NoBypass3m_navigation','worker_fault_matrix']),
        defaults_unchanged=True,history=asset('reports/m6d_acceptance.json'),
        limits='Obstacle probe uses true stationary actor OBB and emergency, not normal planned safe stopping. No exact curb mesh or instrumented wheel contact. No new Planner/Worker run.'))
    runtime_rows=[dict(name=r['name'],physics=r['config']['physics'],mode=r['config']['mode'],runtime=r['runtime'],stages_ms=r['stages_ms']) for r in allr]
    write('reports/m6e_runtime_status.json',dict(**provenance('actual_fixed50ms_actuator_schedule_wallclock'),status='Partial_strict20Hz_open',trials=runtime_rows,
        max_ms=max(r['runtime']['control_interval_ms']['max'] for r in allr),deadline_miss=sum(r['runtime']['deadline_miss'] for r in allr),skipped_slots=sum(r['runtime']['skipped_slots'] for r in allr),
        worker_queue_and_age='NotApplicable_no_Planner',expired_plan_normal='NotApplicable_no_Planner',
        baseline_worst=asset('reports/m6d_runtime_worst_event.json'),limitations='Tick/Apply measured with fixed epoch and no catchup; Physics versions grouped, not UE cause attribution or shared-GPU runtime certification.'))
    print(json.dumps(dict(groups=groups,live=[{k:v for k,v in q.items() if k not in ['phases','trace']} for q in live],versions=versions,
        paired=[{k:q[k] for k in ['P0','P1','mode','split','stopping_distance_delta_m','stopping_time_delta_s','emergency_noninferiority']} for q in paired]),indent=2))
if __name__=='__main__':main()
