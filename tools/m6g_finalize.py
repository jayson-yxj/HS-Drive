"""Seal offline decisions and gate conditional work; never invokes simulator."""
import json
import hashlib
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from tools.m6g_analyze import read,write,asset


def main():
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    sources={str(p):asset(p)['sha256'] for root in ['diagnostics','tools','tests','configs/m6g'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m6g' in p.name or root=='configs/m6g')}
    common=dict(stage='M6-G',base='51b410942428e9994a180f5717aec9fd22369d2e',source='Privileged',new_CARLA_trials=0,
        analysis_commit=head,source_sha256=sources,protocol=asset('configs/m6g/protocol.json'),reproduction='docs/M6G_IMPLEMENTATION.md',
        original_binary_manifest=asset('reports/m6f_execution_logs/server_binary_manifest.json'),original_trial_SHA=asset('reports/m6f_execution_logs/trial_manifest.json'),
        event_acquisition_source_and_Physics='reports/m6g_execution_logs/trial_manifest.json; per-event acquisition_git/physics_sha256/trace_sha256',
        evidence_types=['OfflineAnalysisOfActualM6F_CARLA','OfficialSourceReview','ActualServerBuildEnvironmentAudit','SyntheticAndRecordedUnitTests'],
        new_engine_internal_fields='None',original_binary_exact_source_identity='NotVerified')
    env=read('reports/m6g_execution_logs/build_environment.json');access=read('reports/m6g_execution_logs/UE_access_existing_git_configuration.json')
    feasibility=dict(**common,kind='ActualServerBuildEnvironmentAudit',status='Instrumentation_NotRun_BuildPrerequisiteFailed',
        prerequisites=dict(official_CARLA_source='Retrieved_pinned294096e',authorized_UE_source='NotEstablished_404_and_repository_not_found',local_UE_source='Absent_in_searched_roots',matching_UE_toolchain='NotAvailable',
          resources='Disk194867585024bytes_free; containerRAM50GiB CPUquota14; host503GiB/128CPUs not container allocation',
          storage='Above_official130GB_reference_only_not_validated_build_peak',isolated_build_directory='Possible_not_created_no_build',
          original_Physics_and_binary='Frozen_available',rebuilt_binary_equivalence='NotRun',observation_only_patch='NotRun',true_substep_field_provenance='NotAvailable_runtime'),
        evidence=dict(environment=asset('reports/m6g_execution_logs/build_environment.json'),access=asset('reports/m6g_execution_logs/official_UE_access.json'),configured_git_access=asset('reports/m6g_execution_logs/UE_access_existing_git_configuration.json')),
        exact_reason='No accessible matching Unreal source or engine toolchain/build identity; stop before compilation, no unofficial UE mirror, no environment repair loop.',
        binary_SHA_new=None,compile_started=False,server_started=False,physics_modified=False,
        desired_fields_not_implemented=[dict(name='accepted_analog_inputs',unit='normalized',location='mControlAnalogVals before vehicle update',availability='NotAvailable'),dict(name='actual_brake_torques',unit='engine scale torque; SI conversion requires engine length scale',location='computeBrakeAndHandBrakeTorques local output at each vehicle substep',availability='NotAvailable'),dict(name='current_target_gear_and_timers',unit='engine enum / seconds',location='PxVehicleDriveDynData',availability='NotAvailable'),dict(name='sticky_flags_and_constraint_impulse',unit='bool / impulse with explicit units',location='vehicle constraint creation and scene solver',availability='NotAvailable'),dict(name='true_solver_time',unit='s',location='UE scene step and vehicle internal k/subTimestep separately',availability='NotAvailable')],
        stop_decision='Seal DynamicsObservabilityBaseline; no active instrumentation follow-up without separately established prerequisites and bounded benefit',
        official_build_reference='https://carla.readthedocs.io/en/0.9.16/build_linux/')
    write('reports/m6g_instrumentation_feasibility.json',feasibility)
    write('reports/m6g_instrumented_event_comparison.json',dict(**common,kind='ConditionalWorkPackageStatus',status='NotRun',reason='Instrumentation_NotRun_BuildPrerequisiteFailed',
        baseline='M6F_official0916_only',instrumented_binary_SHA=None,equivalence_result=None,measurements=None,
        scenarios={x:'NotRun' for x in ['auto_cold_start','fixed_forward_cold_start','P0_brake001','matched_speed_emergency','uphill_release_rollback']},
        emergency_vs_comfort='Independent_gates_unchanged',extra_causal_information='No_new_internal_measurement',no_invented_substep_timestamps=True))
    guard=read('reports/m6f_guard_validation.json')
    write('reports/m6g_guard_status.json',dict(**common,kind='OfflineReviewOfExistingCARLA0915',carla_version='0.9.15',status='Guard_Partial_NoIndependentPositive',
        guard_source=asset('simulator/m6e_guard.py'),historical_report=asset('reports/m6f_guard_validation.json'),new_live_tests=0,
        inherited_trials=[dict(path=x['path'],max_release_backward_m=x['max_release_backward_m'],guard_events=x['guard_events'],physics_sha256=x['result']['physics_sha256'],source_commit=x['result']['source_commit']) for x in guard['trials']],
        natural_independent_positive_trials=0,new_protected_response='NotRun',threshold_changes=False,synthetic_tests_certify_real_safety=False))
    for name in ['timeline_alignment','braking_event_analysis','startup_transmission_analysis','physics_rebuild_analysis','runtime_overhead']:
        p=Path('reports/m6g_'+name+'.json');d=read(p);write('reports/m6g_execution_logs/analysis_v1_'+name+'.json',d);d['finalization_provenance']=common
        d['event_details']=asset('reports/m6g_execution_logs/event_details.json')
        if name=='braking_event_analysis':
            d['detail_summary']=read('reports/m6g_execution_logs/event_details.json')['summary']
            missing=[];near_zero=[]
            for e in d['jerk_clusters']:
                if not e['same_frame_debug_force_insufficient']:continue
                sample=json.loads(Path('reports/m6g_event_timelines/'+e['trial_id']+'.jsonl').read_text().splitlines()[e['peak_index']])
                (missing if sample['telemetry'] is None else near_zero).append(e['event_id'])
            d['same_frame_force_insufficiency_breakdown']=dict(observed_near_zero_events=near_zero,missing_telemetry_events=missing,missing_is_not_zero=True)
        p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    timeline=read('reports/m6g_timeline_alignment.json');brake=read('reports/m6g_braking_event_analysis.json');detail=read('reports/m6g_execution_logs/event_details.json')
    tests=[]
    for name in ['full_tests','sdk_tests']:
        path=Path('reports/m6g_execution_logs/'+name+'.xml');tests.append(dict(name=name,xml=asset(path),result=ET.parse(path).getroot().find('testsuite').attrib))
    preserve=read('reports/m6g_execution_logs/preservation.json')
    archive_files=[]
    for root in ['reports/m6f_trials_0916','reports/m6f_guard_0915','reports/m6g_event_timelines']:
        archive_files.extend(asset(p) for p in sorted(Path(root).rglob('*')) if p.is_file())
    write('reports/m6g_dynamics_observability_baseline.json',dict(**common,name='DynamicsObservabilityBaseline',status='Sealed_ResearchCompleted_CausalityPartial',
        actual_trials=dict(CARLA0916=26,CARLA0915_G1=4),assets=archive_files,original_historical_gate_state='unchanged',
        resume_condition='Separate explicit bounded research proposal with lawful matching UE source, toolchain/build equivalence and a specific internal-measurement benefit; no automatic continuation'))
    write('reports/m6g_acceptance.json',dict(**common,status='Partial / ResearchCompleted / InstrumentationNotRun',
        independent_decisions=dict(A_offline_evidence='Partial: startup wait externally localized; initial rebuild discontinuity demonstrated; braking mechanism not causally resolved',B_instrumentation='NoGo_current_build_prerequisites_failed_stop_deep_dive',C_controller_or_Physics='NotApproved',
          default_backend='CARLA0915_unchanged',Physics_P1='NoGo',Guard='Guard_Partial_NoIndependentPositive',normal_comfort='Open_Failed',strict20Hz='Open_Failed',visual_safety='NotValidated'),
        engineering_checks=dict(rows=timeline['rows'],trials=timeline['trial_count'],commands=timeline['command_event_count'],gear_events=timeline['gear_event_count'],delayed25_preserved=timeline['all25_preserved'],
          source_manifest=True,no_new_collection=True,no_new_model=True,no_main_merge=True),
        findings=dict(light_brake_events=brake['light_brake_counts'],jerk_summary=brake['jerk_summary'],detail_summary=detail['summary'],
          rebuild_first_tick_vz_difference_mps=detail['rebuild_pairs'][0]['first_tick_velocity_difference_world_mps'][2],rebuild_first_tick_z_difference_m=detail['rebuild_pairs'][0]['first_tick_z_difference_m']),
        evidence_levels=dict(Observed='recorded outer-frame values and reset discontinuity',Correlated='gear/slip/lock associations',MechanismSupported='pinned official upstream source mechanisms; not Shipping execution certification',Unresolved='accepted analog controls/brake torque/sticky activation/impulse/substep phase'),
        tests=tests,preservation=preserve,instrumentation='Instrumentation_NotRun_BuildPrerequisiteFailed',
        not_run=['native_compile','instrumented_binary_equivalence','new_physics_events','async_diagnostic_runtime','new_G1_live_positive','new_controller','Physics_search','vision_control'],
        next_action='End unbounded PhysX root-cause work; reuse sealed baseline. Oracle route/planning and perception may independently advance under original contracts; no dynamics/safety gates waived.'))
    print('Finalized all required M6G machine-readable reports; zero new CARLA trials')

if __name__=='__main__':main()
