"""Evidence aggregation with independent states; does not run control or train."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import xml.etree.ElementTree as ET
from tools.m7a_common import read,write,asset,provenance,sha


def main():
    prov=provenance();reports={s:read('reports/m7a_planner_'+s+'.json') for s in ['development_v2','validation','final']}
    audit=read('reports/m7a_independent_geometry_audit.json')
    # The predeclared lead_restart script stays still for2s although snapshot
    # velocity=1m/s. Preserve it, but do not call it a valid motion-continuous
    # restart reference or pool it with braking failures.
    future_checks=[]
    for split,d in reports.items():
        for r in d['records']:
            for o in r['request']['scene']['boxes']:
                ref=r['future_reference_evaluator_only'].get(o['object_id'])
                if not ref or len(ref)<2:continue
                observed=(np.array(ref[1][1])-ref[0][1])/(ref[1][0]-ref[0][0])
                discrepancy=float(np.linalg.norm(observed-o['velocity']))
                future_checks.append(dict(scene_id=r['scene_id'],object=o['object_id'],
                   first_interval_velocity_error_mps=discrepancy,
                   valid_initial_motion=discrepancy<=.1,
                   reason='ReferenceInitialVelocityMismatch' if discrepancy>.1 else 'FiniteDifferenceWithinAccelerationTolerance'))
    badfuture={r['scene_id'] for r in future_checks if not r['valid_initial_motion']}
    validfuturefails=[r for r in audit['accepted_candidate_checks'] if r['scripted_future_collisions'] and r['scene_id'] not in badfuture]
    write('reports/m7a_fixture_validity.json',dict(kind='PostEvaluationQualityAudit_NoTuning',provenance=prov,
       static_placement_issues=audit['fixture_issues'],future_initial_motion=future_checks,
       raw_accepted_future_collision_count=audit['accepted_with_scripted_future_collision'],
       valid_initial_motion_future_collision_count=len(validfuturefails),
       valid_future_failures=validfuturefails,
       invalid_restart_references_retained=sorted(badfuture),
       decision='Do not repair and rerun final set; fix development generator separately and reserve fresh final roads'))
    allrecords=[r for d in reports.values() for r in d['records']]
    evaluation=[e for r in allrecords for e in r['evaluations']]
    catalog=[c for d in reports.values() for c in d['catalog']]
    write('reports/m7a_scenario_catalog.json',dict(kind='OfflineCARLAOpenDRIVE_WithSyntheticActors',provenance=prov,
      records=catalog,scenarios=len(catalog),location_split=read('configs/m7a/protocol.json')['location_split'],
      location_distances=audit['location_distances'],fixture_issues=audit['fixture_issues'],fixture_issues_repaired_after_test=False,
      validity='Partial: tangent-placed actors miss intended curved road in6 final fixtures; do not certify those scenario semantics',
      field_definitions='PlanningRequest scene snapshot contains only current boxes/velocity; scripted future stored evaluator-side',
      reference='CARLA0.9.15 saved OpenDRIVE, simulator OBB geometry; not render mesh or physical contact gold',
      pattern_correlation='Same14 scenario templates per road; six road locations are independent units, not84 independent real scenes',
      right_bypass='Implemented but no physical right-neighbor eligible in selected routes; support not proven',
      true_vehicle_dynamics='NotRun_in_new_scenarios'))
    history=read('reports/m7a_historical_regression.json')
    categories=[]
    for r in history['M6B']['trials']:
        result=read(r['result']['path']);config=result['config']
        if r['name'].startswith('runtime_'):
            label='RuntimeProbe_NotManeuverSuccess'
        elif not r['comfort_pass']:
            label='ActuationComfortFailed_MixedCommandPhases_NoSingleRootCauseClaim'
        elif r['vehicle_executed_successfully']:
            label='LimitedGoalAndComfortMetricsPassed_NotSafetyCertified'
        elif config.get('fault'):
            label='FaultDegradationProbe_NotNavigationSuccess'
        else:label='GoalOrTaskNotPassed'
        categories.append(dict(name=r['name'],classification=label,
          goal_pass=r['goal_passed'],comfort_pass=r['comfort_pass'],new_execution=False))
    write('reports/m7a_execution_accounting.json',dict(kind='HistoricalExecutionScopeCorrection',provenance=prov,
      authoritative_classification=categories,
      reason='Do not label runtime/fault runs LimitedHistoricalPass merely because comfort boolean true. Historical replay raw classification retained for audit.',
      original_report=asset('reports/m7a_historical_regression.json')))
    per_split={}
    for split,d in reports.items():
        rr=d['records'];generated=[r for r in rr if r['kind'] not in ['missing','worker_error']]
        expectedreject=[r for r in rr if r['expected']=='reject'];validtime=[r for r in rr if r['kind'] not in ['missing','worker_error','expired','future']]
        per_split[split]=dict(**d['summary'],generated_nonfault_requests=len(generated),
          generation_success_rate=sum(bool(r['candidates']) for r in generated)/len(generated),
          expected_reject_count=len(expectedreject),correct_reject_count=sum(r['decision']['selected'] is None for r in expectedreject),
          decisions=dict(Counter(r['decision']['decision'] for r in rr)),
          rejected_when_predeclared_safe=[r['scene_id'] for r in rr if r['expected']=='safe_candidate' and r['decision']['selected'] is None],
          future_metrics='Independent audit must be read; expectation count alone is not safety pass')
    write('reports/m7a_oracle_planning_benchmark.json',dict(kind='OfflinePlannerBenchmark_AndSeparateHistoricalCARLAExecution',provenance=prov,
      status='ReusableResearchHarness_Partial_NotDynamicSafetyPassed',per_split=per_split,
      reports={s:asset('reports/m7a_planner_'+s+'.json') for s in reports},
      generated_candidate_count=len(evaluation),accepted_candidate_count=sum(e['planning_feasible'] for e in evaluation),
      expected_no_solution_and_fault_reject_total=sum(x['correct_reject_count'] for x in per_split.values()),
      invalid_candidates_accepted=dict(under_CV_dense_reference=audit['accepted_with_CV_collision'],
        exact_map_sampled=audit['accepted_with_road_violation'],
        under_scripted_future_raw=audit['accepted_with_scripted_future_collision'],
        under_valid_initial_motion_scripted_future=len(validfuturefails)),
      execution=asset('reports/m7a_historical_regression.json'),execution_scope=asset('reports/m7a_execution_accounting.json'),new_vehicle_experiments=0,
      historical_execution_counts=dict(M6A=history['M6A']['count'],M6B=history['M6B']['count'],M6B_initialization_failures=2),
      historical_regression='Default source byte unchanged;16 cold planner replays use stronger offline evaluator, not exact worker-state replay or new vehicle A/B',
      PlanningFeasible='Only within input snapshot, CV/yaw-rate forecast and geometric reference; not VehicleExecutedSuccessfully',
      VehicleExecutedSuccessfully='New scenarios NotRun; historical low-speed/comfort failures retained as ActuationFailure',
      actuator_status='ControlNoGo_no_comfort_upgrade',
      dynamic_forecast='Partial_constant_velocity_does_not_cover_lead_braking',
      independent_audit=asset('reports/m7a_independent_geometry_audit.json')))
    write('reports/m7a_candidate_trajectory_quality.json',dict(kind='OfflineCandidateAndIndependentReferenceEvaluation',provenance=prov,
      candidates=len(evaluation),accepted=sum(e['planning_feasible'] for e in evaluation),
      rejection_reasons=dict(Counter(reason for e in evaluation for reason in e['reasons'])),
      independent_audit=audit,full_candidate_values='reports/m7a_planner_{development_v2,validation,final}.json',
      limits=['Conservative continuous translation SAT under linear position/constant yaw-rate interpolation',
        'Road polygon approximation plus sampled CARLA checks not continuous curb mesh certificate',
        'Guide path outside6s is geometric intent only; no dynamic guarantee beyond6s',
        'Stopping trajectory internal jerk separate from historical vehicle jerk',
        'No covariance or worst-case acceleration forecast; future collisions cannot be hidden by CV success',
        'Six accepted restart conflicts have inconsistent initial future velocity; see separate fixture_validity before causal use']))
    write('reports/m7a_safety_semantics_audit.json',dict(kind='OfflineCodeContract_AndSyntheticRegression',provenance=prov,
      status='ContractVerified_SafetyNotValidated',states=dict(OracleSceneState='Privileged',RGBSpatialEvidence='RGB_estimate_OfflineShadow',Unknown='not_Free'),
      full_horizon_s=6.,original_age_s=.2,source_schema='m7a.v1',
      continuous_claim='Constant velocity/yaw SAT intervals with conservative rotation padding; not exact accelerating rotating mesh collision',
      body='M6B Tesla measured half extents and center offset; full2D polygon envelope, not just actor origin or corners',
      roads='same-direction route and permitted neighbor polygons; Broken and go/return permission; no junction right-of-way certification',
      time_faults=['expired','future','missing','worker_error'],fault_method='Offline missing-result/error representation plus existing runtime unit tests; no new live Worker crash',
      production_adoption=False,original_Planner_Safety_Controller='unchanged SHA',
      no_safe_plan='Explicit RejectNoFeasibleCandidate_DegradationRequired; real protected stop not implied',
      stop_accepted='Geometric D_stop only; current vehicle execution remains ControlNoGo',
      G1='Guard_Partial_NoIndependentPositive',Physics_P1='NoGo',visual_driving='NotValidated'))
    iso=read('reports/m7a_oracle_vision_isolation.json');gpu=read('reports/m7a_shadow_rgb_forward.json')
    write('reports/m7a_runtime_status.json',dict(kind='HistoricalCARLA_OfflinePlanner_GPUReplay_SeparatePopulations',provenance=prov,
      status='RuntimePartial',new_control_trials=0,single_tick_owner='Preserved_no_new_live_owner',
      synchronous0916telemetry_in0915=False,
      historical_actual_CARLA_M6B=history['runtime'],historical_M6A_report=asset('reports/m6a_runtime_status.json'),
      historical_stage_timing_report=asset('reports/m6b_runtime_full_campaign.json'),
      current_planner_offline={s:d['summary']['timing_ms'] for s,d in reports.items()},
      current_GPU_replay=dict(frames=len(gpu['records']),peak_allocated_mib=gpu['peak_allocated_mib'],
        inference_ms=[r['inference_ms'] for r in gpu['records']],cold_start_included=True),
      queue_age=iso['queue_age_simulation'],queue_live='NotRun',GPU_shared_with_CARLA='NotRun',
      deadline_definition='Historical work finish after wall slot; distinct from apply interval>50ms',
      original_timestamps_preserved=True,expired_plan_normal_control_historical=history['runtime']['expired_normal_control'],
      strict20Hz='NotPassed',production5Hz_evaluator='NotApproved_offline_complete_audit_tail_exceeds200ms'))
    tests={}
    for name in ['full_final_tests','sdk_final_tests']:
        path='reports/m7a_execution_logs/'+name+'.xml';tests[name]=dict(asset=asset(path),counts=ET.parse(path).getroot().find('testsuite').attrib)
    bench=read('reports/m7a_bench2drive_feasibility.json')
    write('reports/m7a_acceptance.json',dict(kind='IndependentEngineeringResearchDecisions',provenance=prov,status='Partial',
      decisions=dict(A_M6_closeout='ResearchCompleted_CausalityPartial_InstrumentationNotRun_ControlNoGo_RuntimePartial',
       B_Oracle_benchmark='Go_reuse_offline_harness_Partial_fixture_and_dynamic_safety',
       C_six_camera_shadow='Go_limited_offline_interface_CandidateNoGo_perception',
       D_Bench2Drive='PreparationOnly_NoGo_full_interface_runtime_and220_scoring',
       E_camera3D_P0='NotApproved_no_new_official_checkpoint_complete_environment_or4090_forward_evidence'),
      actual_work=dict(offline_scenarios=84,road_locations=6,accepted_reference_checks=audit['accepted_candidates'],
        synthetic_future_conflicting_accepted_candidates_raw=audit['accepted_with_scripted_future_collision'],
        valid_initial_motion_future_conflicting_candidates=len(validfuturefails),
        new_CARLA_runs=0,history_M6A_runs=13,history_M6B_runs=69,
        six_camera_packets=40,historical_visual_groups=38,GPU_frames=2,GPU_repeat_frames=2,training_steps=0),
      tests=tests,preservation='reports/m7a_execution_logs/preservation.json',
      unchanged_gates=dict(comfort='ControlNoGo',runtime='RuntimePartial',Physics_P1='NoGo',Guard='Guard_Partial_NoIndependentPositive',
        perception='CandidateNoGo',occupancy='NotPassed',visual_safety='NotValidated'),
      not_run=['new_CARLA_closed_loop','physics_compile','controller_search','model_training','full_Bench2Drive_runtime','220_routes','mature3D_checkpoint_forward'],
      failures_retained=['initial_synthetic_test_fixture_error','development_v1','6_final_scenario_placement_mismatches','CV_future_prediction_failures',
        'historical_vehicle_and_initialization_failures','official_route_transfer_attempts'],
      next_phase='Repair scenario generator in development, add forecast uncertainty and independent fresh final roads; isolated B2D environment preparation only; no control or perception promotion'))
    print('M7-A reports finalized as independent Partial/NoGo states')

if __name__=='__main__':main()
