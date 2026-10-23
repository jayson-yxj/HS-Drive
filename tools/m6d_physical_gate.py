"""Retain initial analysis and clarify unmeasured phase metrics, never tune actuation."""
import json
from pathlib import Path
from tools.m6d_common import write,provenance


def main():
    data=[]
    for split in ['development','validation']:
        d=json.loads(Path(f'reports/m6d_probe_{split}_analysis.json').read_text())
        for row in d['trials']:
            # A five-sample idle brake phase has insufficient window length for
            # the inherited combined metric helper. It is Unknown, not a raw
            # comfort failure. No actual moving phase is excluded.
            normal=[(name,p) for name,p in row['phases'].items() if name!='initialize' and not p['emergency']]
            failures=[name for name,p in normal if p['raw'] is not None and not p['comfort_passed']]
            missing=[name for name,p in normal if p['raw'] is None]
            row['comfort_status_corrected']='Failed' if failures else 'Unknown_insufficient_phase_samples' if missing else 'Passed_measured_phases'
            row['comfort_failing_phases']=failures;row['comfort_unmeasured_phases']=missing;data.append(row)
    groups={}
    for name in sorted({r['task']['name'] for r in data if not r['task'].get('rollback')}):
        rows=[r for r in data if r['task']['name']==name]
        groups[name]=dict(trials=len(rows),hold_passed=sum(r['hold_passed'] for r in rows),restart_passed=sum(r['restart_passed'] for r in rows),
            comfort_status_counts={k:sum(r['comfort_status_corrected']==k for r in rows) for k in sorted({r['comfort_status_corrected'] for r in rows})})
    gate=dict(**provenance('offline_fixed_protocol_physical_feasibility_no_parameter_fit'),groups=groups,trials=data,
        F1='NotRun_prerequisites_failed',sealed_regression='NotRun_prerequisites_failed',
        decision='NoGo_CurrentPhysics_CurrentActuationPolicy',
        basic_hold='Go_limited_fixed_position_probes',normal_hold_restart='NoGo_raw_comfort_and_auto_restart',
        caveat='No universal impossibility claim. No policy selected from validation. Validation diagnoses predeclared mechanisms, not repeated tuning. Cold .10 idle trials have no successful startup and too-short idle brake segment; not falsely claimed measured high jerk.')
    write('reports/m6d_hold_release_feasibility.json',gate)
    write('reports/m6d_startup_restart_modes.json',dict(**provenance('offline_actual_CARLA_phase_transition_analysis'),groups=groups,trials=[dict(name=r['name'],task=r['task'],phases={k:v for k,v in r['phases'].items() if k in ['prepare','release','restart','normal_stop','brake_stop']}) for r in data],
        limits='Paired initialization/hold/gear histories differ. Auto vs manual comparison matched cold hand hold, but warm vs cold uses different initialization. No unique causal claim about gearbox.'))
    write('reports/m6d_comfort_feasibility.json',dict(**provenance('offline_unfiltered_raw_CARLA_motion'),decision=gate['decision'],groups=groups,
        gates=dict(raw_max_decel_mps2=1.8,raw_jerk_peak_mps3=5,moving_jerk_p95_mps3=2.5),
        raw_all=[dict(name=r['name'],raw=r['all_raw'],phases=r['phases'],status=r['comfort_status_corrected']) for r in data],
        emergency_separate=True,original_records_included=True,physics_experiment_recommendation='Justified scoped proposal; not executed or authorized by this report.'))
    print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
