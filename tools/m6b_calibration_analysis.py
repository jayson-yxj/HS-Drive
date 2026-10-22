"""Keep actuator commands, actual response, coast/hold and raw peaks separate."""
import json
from pathlib import Path
import numpy as np
from tools.m6b_common import write,provenance,asset


def main():
    records=[]
    for p in sorted(Path('reports/m6b_calibration_artifacts').glob('*.jsonl')):
        r=[json.loads(s) for s in p.read_text().splitlines()];t=np.array([q['elapsed'] for q in r]);v=np.array([q['long_speed'] for q in r]);a=np.diff(v)/np.diff(t);j=np.diff(a)/np.diff(t[:-1])
        start=next((k for k in range(len(r)) if r[k]['speed']>.2),None);hold=next((k for k in range(len(r)) if r[k]['hold']),None)
        peak=int(abs(j).argmax());coast=next((k for k in range(len(r)) if t[k]>=4-1e-6),None)
        records.append(dict(name=p.stem,source='Privileged',trace=asset(p),
            first_moving_time_s=None if start is None else float(t[start]),
            first_hold_time_s=None if hold is None else float(t[hold]),hold_entry_speed=None if hold is None else float(v[hold]),
            coast_start_speed=None if coast is None else float(v[coast]),
            coast_displacement_m=None if coast is None or hold is None else float(np.linalg.norm(np.array(r[hold]['position'])-r[coast]['position'])),
            peak_jerk=float(j[peak]),peak_window=r[max(0,peak-2):peak+5],
            gear_changes=[dict(time=float(t[k]),speed=float(v[k]),previous=r[k-1]['applied'],current=r[k]['applied']) for k in range(1,len(r)) if r[k]['applied']['gear']!=r[k-1]['applied']['gear']],
            request_applied_difference_ticks=sum(abs(q['command']['throttle']-q['applied']['throttle'])>1e-5 or abs(q['command']['brake']-q['applied']['brake'])>1e-5 for q in r)))
    write('reports/m6b_longitudinal_analysis.json',dict(**provenance('offline_analysis_of_actual_CARLA_actuator_probes'),source='Privileged',trials=records,
        conclusion='Constant .15 throttle can produce delayed launch and gear0->1 raw jerk while .10 and first4s ramp remain stationary. Command-rate smoothing alone does not ensure actual comfort. Coast can settle in this one position; cross-position tracking trials disprove universal stopping accuracy of candidate approximation.',
        limits='Finite two-repeat fixed-location probes. Gear change and acceleration are temporally associated, not an internal PhysX causal decomposition. No statement that every possible controller or stock Physics configuration is infeasible.'))

if __name__=='__main__':main()
