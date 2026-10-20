"""Raw local HOLD windows vs complete maneuver; never discard onset in gates."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash

def main():
    output=Path('reports/m48_hold_feasibility_analysis.json')
    if output.exists():raise FileExistsError(output)
    source=json.loads(Path('reports/m48_dynamics_feasibility.json').read_text());records=[]
    for result in source['results']:
        rows=[json.loads(x) for x in Path(result['log']).read_text().splitlines()]
        t=np.array([r['elapsed'] for r in rows]);v=np.array([r['long_speed'] for r in rows]);a=np.diff(v)/np.diff(t);j=np.diff(a)/np.diff(t[:-1]);peak=int(abs(j).argmax())
        entry=result['first_hold'];window=None
        if entry:
            mask=(t[1:-1]>=entry['t']-.25)&(t[1:-1]<=entry['t']+.5)
            window=dict(raw_jerk_peak=float(abs(j[mask]).max()),raw_max_decel=float(max(0,-a[:-1][mask].min())),hold_speed=entry['speed'])
        records.append(dict(case=result['case'],global_peak_time=float(t[peak+1]),global_raw_jerk_peak=float(abs(j).max()),hold_window=window,initial_actual_speed=result['initial_speed'],target_initial_speed=result['case']['v']))
    atomic_json(output,dict(kind='offline_raw_dynamics_local_window_analysis',records=records,
        source_sha256=file_hash(__file__),source_report_sha256=file_hash('reports/m48_dynamics_feasibility.json'),
        scope='Local HOLD statistics do not replace global raw gates. Launch cases targeting .1/.2m/s overshot to .695m/s, not valid probes at requested speeds. No impossibility proof.'))
if __name__=='__main__':main()
