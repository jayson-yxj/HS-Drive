"""Raw phase metrics and independently gated hold/restart, without fitting policy."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.m6d_common import write,asset,provenance
from tools.m47_common import dynamics_metrics


def inspect(path):
    d=json.loads(path.read_text());rows=[json.loads(s) for s in (path.parent/'trace.jsonl').read_text().splitlines()]
    phases={}
    for phase in dict.fromkeys(q['phase'] for q in rows):
        r=[q for q in rows if q['phase']==phase];v=np.array([q['signed']['v_forward_mps'] for q in r]);t=np.array([q['measurement']['t'] for q in r]);pos=np.array([q['measurement']['position'] for q in r]);drift=np.linalg.norm(pos[:,:2]-pos[0,:2],axis=1)
        # Prepend two samples to retain command-switch acceleration/jerk.
        first=rows.index(r[0]);metric_rows=rows[max(0,first-2):first+len(r)]
        metrics=dynamics_metrics([q['measurement'] for q in metric_rows]) if len(metric_rows)>=8 else None
        valid=[q for q in r if q['hold_mode']=='HOLD_STABLE'];forward=np.flatnonzero(v>.1)
        phases[phase]=dict(ticks=len(r),duration_s=float(t[-1]-t[0]),initial_signed_speed=float(v[0]),final_signed_speed=float(v[-1]),maximum_speed=float(abs(v).max()),
            max_displacement_m=float(drift.max()),backward_ticks=int((v<-.05).sum()),
            hold_stable_ticks=len(valid),hold_physical_passed=bool(len(r)>=60 and all(q['signed']['speed_norm_mps']<=.02 for q in r) and drift.max()<=.05 and len(valid)>0),
            restart_delay_s=float(t[forward[0]]-t[0]) if len(forward) else None,
            restart_passed=bool(len(forward) and t[forward[0]]-t[0]<=2 and not (v<-.05).any()),
            raw=metrics,comfort_passed=bool(metrics and metrics['max_decel_raw']<=1.8 and metrics['jerk_peak_raw']<=5 and metrics['jerk_p95_moving']<=2.5),
            gear_states=sorted({q['measurement']['applied']['gear'] for q in r}),
            emergency=any(q['protection'] for q in r),slope_range=[min(q['road']['grade'] for q in r),max(q['road']['grade'] for q in r)])
    relevant=[v for k,v in phases.items() if k!='initialize']
    holds=[v for k,v in phases.items() if k.startswith('hold')];releases=[v for k,v in phases.items() if k in ['release','restart']]
    return dict(name=d['name'],task=d['task'],physics=d['physics'],trace=d['trace'],phases=phases,
        hold_passed=bool(holds and all(v['hold_physical_passed'] for v in holds)),
        restart_passed=bool(releases and all(v['restart_passed'] for v in releases)),
        normal_comfort_passed=all(v['comfort_passed'] for v in relevant if not v['emergency']),
        all_raw=d['raw_all_including_initialize'],collisions=d['collisions'],complete=d['complete'],
        reference='Real CARLA snapshots; phase raw metrics include previous two samples at command boundary. Initialize kept separately; prepare/startup not removed from normal gates.')


def main():
    p=argparse.ArgumentParser();p.add_argument('split',choices=['development','validation']);a=p.parse_args()
    records=[inspect(p) for p in sorted(Path('reports/m6d_probe_artifacts').glob(a.split+'*/result.json'))]
    methods={}
    for name in sorted({r['task']['name'] for r in records if not r['task'].get('rollback')}):
        subset=[r for r in records if r['task']['name']==name]
        methods[name]=dict(trials=len(subset),hold_passed=sum(r['hold_passed'] for r in subset),restart_passed=sum(r['restart_passed'] for r in subset),comfort_passed=sum(r['normal_comfort_passed'] for r in subset),
            all_prerequisites_passed=bool(len(subset)==4 and all(r['hold_passed'] and r['restart_passed'] and r['normal_comfort_passed'] for r in subset)))
    report=dict(**provenance('offline_raw_actual_CARLA_probe_analysis'),split=a.split,trials=records,methods=methods,
        F1_prerequisite_passed=any(v['all_prerequisites_passed'] for v in methods.values()),
        independent_city_generalization=False,limits='Two repeated seeds per fixed position, not independent worlds. Same Town03. Diagnostic no Planner, not closed-loop navigation or strict runtime.')
    write(f'reports/m6d_probe_{a.split}_analysis.json',report)
    print(json.dumps(methods,indent=2))
if __name__=='__main__':main()
