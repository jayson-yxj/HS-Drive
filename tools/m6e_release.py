"""Independent acceptance, original evidence preservation and SHA sealing."""
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from tools.m6e_common import write,provenance,asset
from simulator.recorder import file_hash


def load(path):return json.loads(Path(path).read_text())

def main():
    required=['summary.md','rollback_guard_validation.json','signed_displacement_audit.json','actuation_interface_comparison.json','physics_baseline.json','physics_variant.json','startup_braking_results.json','hold_restart_results.json','safety_regression.json','runtime_status.json']
    for name in required:assert Path('reports','m6e_'+name).is_file(),name
    tests=[]
    for name in ['full','sdk']:
        p=Path('reports')/f'm6e_{name}_tests.xml';suite=ET.parse(p).getroot().find('testsuite');a=suite.attrib
        xfail=sum(t.find('skipped') is not None and t.find('skipped').get('type')=='pytest.xfail' for t in suite.findall('testcase'))
        tests.append(dict(name=name,tests=int(a['tests']),passed=int(a['tests'])-int(a['failures'])-int(a['errors'])-int(a['skipped']),skipped=int(a['skipped'])-xfail,xfailed=xfail,failures=int(a['failures']),errors=int(a['errors']),seconds=float(a['time']),junit=asset(p),
            skip_reasons=[dict(test=t.attrib,reason=t.find('skipped').attrib) for t in suite.findall('testcase') if t.find('skipped') is not None]))
    assert all(x['failures']==x['errors']==0 for x in tests)
    write('reports/m6e_tests.json',dict(**provenance('actual_server_pytest'),suites=tests,historical_xfail='M6-D old guard remains expected failure; M6-E regression passes independently.'))
    guard=load('reports/m6e_rollback_guard_validation.json');physics=load('reports/m6e_physics_variant.json');interface=load('reports/m6e_actuation_interface_comparison.json');safety=load('reports/m6e_safety_regression.json');runtime=load('reports/m6e_runtime_status.json')
    live=guard['live'];positives=[q for q in live if q['split']=='development'];negatives=[q for q in live if q['split']=='validation']
    p1cycles=[q for q in physics['trials'] if q['config']['mode']=='cycle'];assert not any(q['normal_comfort_passed'] for q in p1cycles)
    record=dict(**provenance('actual_CARLA_offline_synthetic_independent_acceptance'),stage='M6-E',status='Partial / CandidateNoGo',
        counts=dict(P0_interface_cycles=24,live_guard=4,P1_cycles=4,paired_safety_and_coast=24,total_actual_trials=56),
        decisions=dict(guard='Go_limited_research_regression_and_actual_development; independent_positive_support_Partial',
            P0_actuation='NoGo_original_raw_comfort',P1='NoGo_no_consistent_comfort_gain_and_emergency_distance_regression',normal_stop_Pilot='NotRun_prerequisites_failed',
            default_replacement=False,strict_runtime='Partial_open',visual_safety='NotValidated',physics_road_realism='NotValidated'),
        guard=dict(known_failure_replay=guard['known_failure_replay'],actual_positive_trials=len(positives),positive_detected_stopped=sum(q['emergency'] and q['final_stopped'] for q in positives),
            actual_validation_small_motion=len(negatives),validation_emergencies=sum(q['emergency'] for q in negatives),
            historical_quiet_false=guard['historical_quiet_false'],live_quiet_latched_risk_ticks=sum(q['risk_ticks'] for q in guard['quiet_hold_controls']),live_quiet_latched_emergency_ticks=sum(q['emergency_ticks'] for q in guard['quiet_hold_controls']),live_quiet_new_events=sum(q['new_risk_events']+q['new_emergency_events'] for q in guard['quiet_hold_controls']),
            limitation='Validation live trials did not reproduce large rollback; not independent positive sensitivity proof. No general false-positive rate from small selected controls.'),
        interface_groups=interface['groups'],physics_versions=physics['versions'],
        P1_safety_pairs=[{k:q[k] for k in ['P0','P1','mode','split','initial_speed_delta','stopping_distance_delta_m','stopping_time_delta_s','emergency_noninferiority','gap_P0','gap_P1']} for q in physics['paired'] if q['mode'] in ['emergency','obstacle']],
        safety=dict(collisions=safety['collisions'],boundary_ticks=safety['boundary_ticks'],closed_loop=safety['conditional_full_closed_loop']),
        runtime={k:runtime[k] for k in ['status','max_ms','deadline_miss','skipped_slots','worker_queue_and_age','expired_plan_normal']},
        tests=tests,preservation=load('reports/m6e_preservation.json'),source_loading_audit=asset('reports/m6e_loaded_source_audit.json'),
        unchanged=dict(default_controller=True,Planner=True,Safety=True,baseline_physics=True,vision_shadow=True,M5_closeout=True),
        next_decision='Stop current fixed-policy/half-brake variant expansion. Limited guard research may continue. Request separate instrumented execution architecture/solver investigation before new physics/model migration; no automatic third variant.')
    assert record['preservation']['passed']
    freeze=load('reports/m6e_execution_logs/base_freeze.json')
    assert all(Path(p).is_file() and file_hash(p)==sha for p,sha in freeze['files'].items())
    write('reports/m6e_acceptance.json',record)
    files=[]
    for p in Path('reports').glob('m6e_*'):
        if p.name=='m6e_release_manifest.json':continue
        files.extend([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()])
    for d in ['simulator','tools','tests','docs','configs/m6e']:
        files.extend(p for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in str(p) and ('m6e' in p.name.lower() or d=='configs/m6e'))
    changes=subprocess.check_output(['git','diff','--name-status','ba5ba6e'],text=True).splitlines();assert all(q.startswith('A\t') for q in changes)
    write('reports/m6e_release_manifest.json',dict(**provenance('actual_server_release_SHA256_seal'),passed=True,files=[asset(p) for p in sorted(set(files))]))
    print('sealed',len(set(files)),'files; tests',[(q['passed'],q['skipped'],q['xfailed']) for q in tests])
if __name__=='__main__':main()
