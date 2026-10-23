"""Finalize finite M6-F evidence and independent decisions; no simulations."""
import json
import hashlib
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from tools.m6f_analyze import read,write,quant
from tools.m6f_evidence import asset


def main():
    detail=read('reports/m6f_execution_logs/detailed_evidence.json');causal=read('reports/m6f_actuation_causal_probe.json');guard=read('reports/m6f_guard_validation.json')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    sources={str(p):asset(p)['sha256'] for folder in ['tools','simulator','tests','configs/m6f'] for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m6f' in p.name or folder=='configs/m6f')}
    provenance=dict(base='b0ee249236415627896000d89fe6d9fa31023e59',analysis_source_commit=head,source_sha256=sources,
        execution_sources='Each batch launch/completion includes actual import source/code SHA and Git commit; analysis commit is not claimed as simulation execution commit.',
        configuration=asset('configs/m6f/protocol.json'),acquisition_budget=asset('configs/m6f/acquisition_budget.json'),
        reproduction='docs/M6F_IMPLEMENTATION.md',versions={'main':'0.9.15','independent_research':'0.9.16'})
    manifest=[asset(p) for root in ['reports/m6f_trials_0916','reports/m6f_guard_0915'] for p in sorted(Path(root).rglob('*')) if p.is_file()]
    write('reports/m6f_execution_logs/trial_manifest.json',dict(**provenance,files=manifest))
    p0=read('reports/m6f_trials_0916/feasibility/p27_api/physics.json')
    # Exact numerical relation is evidence about the API debug field, not an estimate of actuator torque.
    residual=[];ratio=[];api_error=[];all_interval=[]
    for p in sorted(Path('reports/m6f_trials_0916').glob('*/*/trace.jsonl')):
        rs=[json.loads(s) for s in p.read_text().splitlines()];all_interval.extend(np.diff([r['timeline']['apply'] for r in rs])*1000)
        for i,r in enumerate(rs):
            if i:
                prev=rs[i-1]['body'];cur=r['body'];api_error.append(abs((cur['long_speed']-prev['long_speed'])/(cur['t']-prev['t'])-cur['long_acceleration_api']))
            if r['telemetry_after']:
                for w in r['telemetry_after']['data']['wheels']:
                    if abs(w['long_force'])>1:
                        ratio.append(w['torque']/w['long_force']);residual.append(abs(w['torque']+.37*w['long_force']))
    relation=dict(n=len(ratio),torque_div_long_force_m=[min(ratio),max(ratio)],max_abs_torque_plus_radius_force_Nm=max(residual),interpretation='Consistent with longitudinal tire reaction torque at0.37m radius. Not separately measured brake actuator torque; no solver equation reconstruction.')
    feasibility=dict(**provenance,kind='actual_server_install_start_and_CARLA_telemetry',source='Privileged',carla_version='0.9.16',true_engine_wheel_api=True,
        status='Go_isolated_observability_research_Partial_full_causal_chain',server=read('reports/m6f_execution_logs/server_binary_manifest.json'),
        official_archive=Path('reports/m6f_execution_logs/server_archive.sha256').read_text().strip(),
        SDK=read('reports/m6f_execution_logs/sdk_official_origin.json'),physics_P0_sha256=p0['sha256'],
        physics_signature_note='canonical sorted JSON hash; legacy indented-file hash3c493... uses different serialization; exact public parameter equality verified',
        physics_same_public_values_as0915=True,version_single_factor_comparison=False,main_migrated=False,
        acquisition=dict(archive_bytes=8346095504,range_download=read('reports/m6f_execution_logs/range_download.json'),startup_attempts=1,server_port=2200),
        validation=dict(actual_trials=26,rows=detail['rows'],wheel_observation_rows=14370,four_wheel_records=57480,frame_mismatches=detail['bad_frame_brackets'],no_tick_repeat=causal['repeat_no_tick'],speed_abs_error=causal['body_vs_telemetry_speed_abs'],fields=causal['fields']),
        not_available=read('reports/m6f_telemetry_contract.json')['not_available'],native0915_instrumentation='NotRun_0916_path_obtained_no_parallel_rebuild',
        sources=['https://github.com/carla-simulator/carla/releases/tag/0.9.16','https://pypi.org/project/carla/0.9.16/'],
        limits=['No engine substep timestamp or net impulse','Physics reapply changes live state at unchanged external frame','25 command transition rows apply one tick late','Actual brake torque unavailable','Finite repeatability across3locations not broad physical validation'])
    write('reports/m6f_0916_telemetry_feasibility.json',feasibility)
    for name in ['0915_observability_limit','telemetry_contract','actuation_causal_probe','physics_sham_comparison','guard_validation','runtime_overhead']:
        path=Path('reports/m6f_'+name+'.json');d=read(path);backup=Path('reports/m6f_execution_logs/pre_finalize_'+name+'.json');backup.write_bytes(path.read_bytes());d.update(provenance);d['trial_manifest']=asset('reports/m6f_execution_logs/trial_manifest.json')
        if name=='actuation_causal_probe':d['debug_torque_relation']=relation;d['api_accel_vs_raw_fd_abs']=quant(api_error)
        if name=='telemetry_contract':
            d['time']['state_mutation_barrier']='PhysicsControl/teleport mutation invalidates same-state snapshot association even without tick. Init records kept separate; no derivatives across rebuild.'
        if name=='runtime_overhead':
            d.update(global_control_intervals_ms=quant(all_interval),global_mean_Hz=1000/float(np.mean(all_interval)),deadline_misses=sum(x['deadline_misses'] for x in detail['runtime']),skipped_slots=sum(x['skipped_slots'] for x in detail['runtime']))
        path.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    testresults=[]
    for n in ['full_tests','sdk_tests_corrected','preflight_tests']:
        xml=ET.parse('reports/m6f_execution_logs/'+n+'.xml').getroot();s=xml.find('testsuite');testresults.append(dict(name=n,counts=s.attrib,artifact=asset('reports/m6f_execution_logs/'+n+'.xml')))
    acceptance=dict(**provenance,stage='M6-F',status='Partial',kind='separated_actual_CARLA_offline_synthetic_acceptance',source='Privileged',
        decisions=dict(telemetry_interface='Passed_limited_real_engine_API',execution_chain_causality='Partial',isolated0916_research='Go_limited_observability',main_backend_migration='NotApproved',normal_stop_controller='NoGo_prerequisites_not_met',Physics_P1='NoGo_historical_unchanged',new_physics_search='NotRun_not_justified',G1='Partial_no_independent_dangerous_positive',strict20Hz='Open_failed_long_tail',visual_driving_safety='NotValidated',Oracle_complete_Pilot='NotRun'),
        counts=dict(CARLA0916=26,CARLA0915_guard=4,total_vehicle_trials=30,frames0916=detail['rows'],frame_alignment_failures=0,delayed_applied_command_rows=25),
        measurements=dict(rpc=read('reports/m6f_runtime_overhead.json')['telemetry_rpc_ms'],control_intervals=quant(all_interval),deadline_misses=sum(x['deadline_misses'] for x in detail['runtime']),sham_live_speed_jump_mps=-.002978399395942688,guard_natural_positive_trials=guard['natural_positive_trials']),
        tests=testresults,failed_test_command=asset('reports/m6f_execution_logs/sdk_tests.txt'),preservation=read('reports/m6f_execution_logs/preservation.json'),
        unchanged=['default_control','0915_PhysicsP0','Planner/Safety6s','DepthBins48','models','historical_reports','G1_thresholds','main_branch'],
        not_run=['0915_native_rebuild','new_controller_or_PID_search','new_Physics_family','full_Oracle_closed_loop_Pilot','vision_control','new_Occupancy_training','independent_dangerous_G1_protected_response_absent_natural_positive'],
        figures=[asset(p) for p in sorted(Path('reports/m6f_figures').glob('*.png'))],trial_manifest=asset('reports/m6f_execution_logs/trial_manifest.json'))
    write('reports/m6f_acceptance.json',acceptance)

if __name__=='__main__':main()
