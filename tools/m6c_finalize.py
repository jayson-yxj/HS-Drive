"""Independent M6-C decisions without upgrading failed task/safety gates."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from tools.m6c_common import write,provenance,asset


def load(p):return json.loads(Path(p).read_text())


def main():
    stop=load('reports/m6c_stop_abcde_comparison.json');no=load('reports/m6c_no_bypass_safety.json');safety=load('reports/m6c_safety_regression.json');runtime=load('reports/m6c_runtime_tail.json');body=load('reports/m6c_recorded_body_audit.json');env=load('reports/m6c_coast_brake_envelope.json');residual=load('reports/m6c_envelope_residual_validation.json')
    blocked=[v for v in no['trials'] if v['config']['scene']=='blocked'];sudden=[v for v in no['trials'] if v['config']['scene']=='sudden'];bypass=[v for v in no['trials'] if v['config']['scene']=='bypass'];removed=[v for v in no['trials'] if v['config']['scene']=='removed'];bybody={v['name']:v for v in body['trials']}
    tests=[]
    for kind in ['full_tests','sdk_tests']:
        path=f'reports/m6c_{kind}.xml';suite=ET.parse(path).getroot().find('testsuite');a=suite.attrib
        tests.append(dict(kind=kind,tests=int(a['tests']),passed=int(a['tests'])-int(a['failures'])-int(a['errors'])-int(a['skipped']),failures=int(a['failures']),errors=int(a['errors']),skipped=int(a['skipped']),seconds=float(a['time']),junit=asset(path),
            skip_reasons=[dict(test=t.attrib,reason=t.find('skipped').get('message'),detail=t.find('skipped').text) for t in suite.findall('testcase') if t.find('skipped') is not None]))
    write('reports/m6c_tests.json',dict(**provenance('actual_server_pytest'),suites=tests,passed=all(t['failures']==t['errors']==0 for t in tests)))
    execution=[]
    for kind in ['stop','maneuver','corrected_matrix','preserved_regression']:execution+=load(f'reports/m6c_{kind}_execution.json')['runs']
    def obstacle_summary(values):
        out={}
        for arm in ['C','D']:
            sub=[v for v in values if v['config']['arm']==arm];out[arm]=dict(runs=len(sub),
                final_gap3_passed=sum(v.get('obstacle',{}).get('final_3m_passed',False) for v in sub),
                moving_safety_fallback=sum(v.get('obstacle',{}).get('final_stop_requires_safety',False) for v in sub),
                insufficient_margin_reported=sum(v.get('obstacle',{}).get('any_InsufficientStoppingMargin',False) for v in sub),
                normal_stop_with_gap3=sum(v.get('obstacle',{}).get('final_3m_passed',False) and v['obstacle']['normal_only'] for v in sub),
                initial_feasible=sum(v.get('obstacle',{}).get('initial_stopping_feasible',False) for v in sub),
                final_gaps={v['name']:v.get('obstacle',{}).get('final_relevant_OBB_gap_m') for v in sub})
        return out
    result=dict(**provenance('actual_CARLA_offline_synthetic_independent_acceptance'),stage='M6-C',status='Partial',source='Privileged',
        counts=dict(probes=112,formal_attempts=len(execution),failed_commands=sum(v['exit_code']!=0 for v in execution),completed=safety['completed_runs']),
        geometry=dict(status='Passed_limited_contract_and_regression',no_goal_shift=True,original_3m_gap=True,exact_mesh='Unknown'),
        envelope=dict(status='Partial_empirical_validation_not_universal_guarantee',summary_original_with_shared_downhill_guard=env['summary'],strict_ablation=load('reports/m6c_envelope_strict_ablation.json')['groups'],
            validation_remaining_samples=sum(v['stats']['C']['supported_samples'] for v in residual['trials']),
            validation_remaining_underbounds=sum(v['stats']['C']['underbound_samples'] for v in residual['trials']),
            unsupported_downhill_coast_preserved=True),
        stop=dict(geometry_bug_excluded_trials=len(stop['invalid_D_trials']),status='NoGo_candidate_upgrade',by_arm=stop['by_arm'],B='Historical_only',E='NotRun'),
        blocked=dict(status='Partial',by_arm=obstacle_summary(blocked)),sudden=dict(status='Partial_infeasible_cases_retained',by_arm=obstacle_summary(sudden)),
        bypass=dict(by_arm={arm:dict(runs=sum(v['config']['arm']==arm for v in bypass),goal_passed=sum(v['config']['arm']==arm and v['goal_passed'] for v in bypass)) for arm in ['C','D']},runs=len(bypass),goal_passed=sum(v['goal_passed'] for v in bypass),body_violation_frames=sum(bybody[v['name']]['footprint_outside_frames'] for v in bypass),
            collisions=sum(len(v['collisions']) for v in bypass),comfort_passed=sum(v['comfort_passed'] for v in bypass)),
        removed=dict(runs=len(removed),restarted=sum(v['removed']['restarted'] for v in removed),goal_passed=sum(v['goal_passed'] for v in removed)),
        navigation={arm:dict(runs=sum(v['config']['arm']==arm for v in safety['navigation']),goal_passed=sum(v['config']['arm']==arm and v['goal_passed'] for v in safety['navigation'])) for arm in ['C','D']},
        lead=[dict(name=v['name'],**v['lead']) for v in safety['lead']],
        fault_results=[dict(name=v['name'],**v['fault'],collisions=len(v['collisions']),expired_normal=v['runtime']['expired_normal_control']) for v in safety['faults']],
        safety=dict(collisions=safety['collisions'],illegal_solid_events=safety['illegal_solid_events'],expired_normal=safety['expired_normal_control'],
            body_outside_frames=sum(v['footprint_outside_frames'] for v in body['trials']),body_intersections=sum(len(v['OBB_intersections']) for v in body['trials']),
            old_six_second_contract_unchanged=True,curb_mesh='Unknown'),
        runtime=dict(status='Partial_strict_20Hz_NotPassed',all_max_ms=runtime['all_trial_max']['runtime']['control_interval_ms']['max'],worst_name=runtime['all_trial_max']['name'],
            all_deadline_miss=runtime['all_deadline_miss'],all_over100=runtime['all_over100'],internal_UE_root_cause='Unknown_not_proven_by_client_and_proc_samples'),
        comfort='Failed_raw_gates_startup_and_physical_braking',
        default_replacement=False,reverse_parking='NoGo',E_policy_search='NotRun',PETR='Paused_NotRun',visual_safety='Unchanged_NotValidated',
        tests=tests,preservation=load('reports/m6c_preservation.json'))
    write('reports/m6c_acceptance.json',result)
    out=Path('reports/m6c_execution_logs');out.mkdir()
    for p in Path('runs/m6c_logs').iterdir():
        if p.is_file():shutil.copy2(p,out/(p.name+'.txt' if p.suffix=='.log' else p.name))
    print(json.dumps({k:result[k] for k in ['counts','stop','blocked','sudden','bypass','removed','navigation','runtime','safety']},indent=2))

if __name__=='__main__':main()
