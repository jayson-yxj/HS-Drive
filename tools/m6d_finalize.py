"""Independent M6-D acceptance and append-only release metadata."""
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from tools.m6d_common import write,asset,provenance


def load(p):return json.loads(Path(p).read_text())

def main():
    tests=[]
    for name in ['final_full_tests','final_sdk_tests']:
        p=Path(f'reports/m6d_{name}.xml');s=ET.parse(p).getroot().find('testsuite');a=s.attrib
        tests.append(dict(name=name,tests=int(a['tests']),failures=int(a['failures']),errors=int(a['errors']),junit_skipped_including_xfail=int(a['skipped']),xfailed=sum(t.find('skipped') is not None and t.find('skipped').get('type')=='pytest.xfail' for t in s.findall('testcase')),skipped=sum(t.find('skipped') is not None and t.find('skipped').get('type')!='pytest.xfail' for t in s.findall('testcase')),passed=int(a['tests'])-int(a['failures'])-int(a['errors'])-int(a['skipped']),
            seconds=float(a['time']),junit=asset(p),skip_details=[dict(name=t.get('name'),type=t.find('skipped').get('type'),message=t.find('skipped').get('message')) for t in s.findall('testcase') if t.find('skipped') is not None]))
    write('reports/m6d_tests.json',dict(**provenance('actual_server_pytest'),suites=tests,known_xfail='Intermittent measured rollback bypasses duration guard; tracked NoGo, not counted passed.'))
    runtime=load('reports/m6d_runtime_status.json');safety=load('reports/m6d_stop_safety_comparison.json');hold=load('reports/m6d_hold_release_feasibility.json');motion=load('reports/m6d_signed_motion_contract.json')
    worst=Path('reports/m6d_trials')/runtime['worst_name']/'trace.jsonl';r=[json.loads(s) for s in worst.read_text().splitlines()];gaps=np.diff([q['timeline']['apply_control'] for q in r]);i=int(gaps.argmax())+1;a=r[i-1]['timeline'];b=r[i]['timeline']
    timeline=dict(name=runtime['worst_name'],frame=r[i]['frame'],interval_ms=float(gaps[i-1]*1000),previous=a,current=b,
        parts_ms=dict(previous_apply_to_tick=(a['tick_start']-a['apply_control'])*1000,previous_tick=(a['tick_end']-a['tick_start'])*1000,record_tail=(b['previous_sleep_start']-a['tick_end'])*1000,sleep_wakeup=(b['wake_up']-b['previous_sleep_start'])*1000,current_preapply=(b['apply_control']-b['wake_up'])*1000),
        inference='Timing localization only; no evidence identifying a unique UE/OS internal cause.',log=asset(worst))
    write('reports/m6d_runtime_worst_event.json',dict(**provenance('offline_actual_timeline_decomposition'),event=timeline))
    result=dict(**provenance('actual_CARLA_offline_and_synthetic_independent_acceptance'),stage='M6-D',status='Partial / CandidateNoGo',
        counts=dict(probes=38,hold_release_protocol_trials=32,cold_release_diagnostics=4,warm_guard_followups=2,closed_loop_development_trials=28,short_routes=12,stationary_fault_trials=8,moving_fault_trials=8,audited_snapshot_frames=sum(q['frames'] for q in motion['snapshot_audits'])),
        decisions=dict(signed_observation_schema='Go_limited_research',target_separation_schema='Go_limited_research',complete_signed_motion_protection='NoGo_intermittent_rollback_missed',
            fixed_position_hold='Go_limited_32_trials',hold_restart_with_comfort='NoGo',normal_stop_execution='NoGo_CurrentPhysics_CurrentActuationPolicy',
            F1='NotRun_prerequisites_failed',sealed_regression='NotRun_prerequisites_failed',independent_physics_study='Evidence_supports_scoped_proposal_only_NotExecuted',reverse_parking='NoGo'),
        hold_groups=hold['groups'],short_route_comparison=load('reports/m6d_controller_ablation.json')['arms'],
        guard_failures=load('reports/m6d_rollback_regression.json')['actual_guard_probes'],
        faults=[dict(name=q['name'],**q['fault']) for q in safety['faults']],
        runtime={k:runtime[k] for k in ['status','all_max_ms','worst_name','deadline_miss','skipped_slots','expired_normal','over100']},
        runtime_worst=asset('reports/m6d_runtime_worst_event.json'),
        safety=dict(collisions=safety['collisions'],illegal_solid=safety['illegal_solid_events'],boundary_ticks=sum(q['boundary_ticks'] for q in safety['trials']),exact_curb_contact='Unknown'),
        original_planner_safety_physics_unchanged=True,default_controller_unchanged=True,
        M5='ResearchCompleted / CandidateNoGo / SafetyNotValidated',PETR='Paused_NotRun',Occupancy='NotRun',
        tests=tests,preservation=load('reports/m6d_preservation.json'),
        evidence_limits=['Same Town03, position repeats correlated. No independent sealed test after failed prerequisites.',
            'F0 normal outputs unchanged in identical-input test; actual route results still depend on physics and scheduling.',
            'Real negative velocity supported by 11.7mm backward pose excursion; guard misses it. Schema pass cannot close complete motion gate.',
            'No universal impossibility proof; only predeclared current actuation policies rejected.'])
    write('reports/m6d_acceptance.json',result)
    print(json.dumps(dict(decisions=result['decisions'],runtime=result['runtime'],tests=tests,worst=timeline['parts_ms']),indent=2))
if __name__=='__main__':main()
