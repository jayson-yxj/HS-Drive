"""Action-matched bound quality; sample coverage and censoring stay explicit."""
import json
from pathlib import Path
import numpy as np
from tools.m6c_common import write,provenance,asset


def main():
    report=json.loads(Path('reports/m6c_coast_brake_envelope.json').read_text());phase=json.loads(Path('reports/m6c_probe_phase_analysis.json').read_text());by={x['name']:x for x in phase['trials']};groups=[]
    for split in ['development','validation']:
        for action in ['coast','brake_001','brake_005','emergency']:
            rows=[r for r in report['trials'] if r['split']==split and r['action']==action];summary={}
            for method in ['A','B','C']:
                values=[r for r in rows if r['predictions'][method]['distance_m'] is not None];stopped=[r for r in values if r['stopped']]
                errors=[r['predictions'][method]['actual_stop_error_m'] for r in stopped]
                summary[method]=dict(action_matched=method!='A' or action=='coast',finite=len(values),finite_coverage=len(values)/len(rows),
                    bound_failures=sum(r['predictions'][method]['upper_bound_failed'] for r in values),
                    censored_finite_claims=sum(r['predictions'][method]['no_stop_finite_claim'] for r in values),
                    min_upper_minus_actual=None if not errors else min(errors),max_upper_minus_actual=None if not errors else max(errors),
                    mean_absolute_error=None if not errors else float(np.mean(abs(np.array(errors)))))
            groups.append(dict(split=split,action=action,trials=len(rows),unique_positions=sorted({r['spawn'] for r in rows}),stopped=sum(r['stopped'] for r in rows),
                min_observed_travel_m=min(r['observed_distance_m'] for r in rows),max_observed_travel_m=max(r['observed_distance_m'] for r in rows),
                max_original_probe_raw_decel=max(by[r['name']]['probe_metrics']['max_decel_raw'] for r in rows),
                max_original_probe_raw_jerk=max(by[r['name']]['probe_metrics']['jerk_peak_raw'] for r in rows),predictions=summary))
    write('reports/m6c_envelope_quality.json',dict(**provenance('offline_action_matched_actual_CARLA_probe_quality'),source='Privileged',groups=groups,
        limits='No CI from correlated within-location repeats; finite support percentage is not general safety coverage. Max observed travel includes non-stopping censored rows. A brake results excluded from action-matched claims.'))

if __name__=='__main__':main()
