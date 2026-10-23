"""Offline raw telemetry diagnostics. Never produce engine-internal estimates."""
import json
from pathlib import Path
import hashlib
import subprocess
import numpy as np


def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def read(path):return json.loads(Path(path).read_text())


def quant(x):
    x=np.asarray(x,float)
    return dict(n=len(x),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),max=float(x.max())) if len(x) else None


def raw_metrics(rows):
    t=np.array([r['body']['t'] for r in rows]);v=np.array([r['body']['long_speed'] for r in rows]);a=np.diff(v)/np.diff(t);j=np.diff(a)/np.diff(t[1:])
    return dict(max_decel_raw=float(max(0,-a.min())),max_accel_raw=float(a.max()),jerk_peak_raw=float(abs(j).max()),jerk=quant(abs(j)),min_signed_speed=float(v.min()),max_speed=float(v.max()),end_speed=float(v[-1]))


def main():
    trials=[];all_rows=[]
    for p in sorted(Path('reports/m6f_trials_0916').glob('*/*/result.json')):
        d=p.parent;rows=[json.loads(s) for s in (d/'trace.jsonl').read_text().splitlines()];all_rows.extend(rows)
        phases={}
        for phase in dict.fromkeys(r['phase'] for r in rows):
            indices=[i for i,r in enumerate(rows) if r['phase']==phase];group=rows[max(0,indices[0]-2):indices[-1]+1]
            entry=raw_metrics(group);start=rows[indices[0]]['body_before']['t']
            moving=[r['body']['t']-start for r in rows[indices[0]:indices[-1]+1] if r['body']['long_speed']>.08]
            entry['first_above_008_s']=min(moving) if moving else None
            entry['initial_speed']=rows[indices[0]]['body_before']['long_speed']
            phases[phase]=entry
        trials.append(dict(path=str(d),config=read(d/'config.json'),result=read(p),initialization=read(d/'initialization.json'),phases=phases,raw=raw_metrics(rows)))
    ts=[r for r in all_rows if r['telemetry_after'] is not None];wheels=[w for r in ts for w in r['telemetry_after']['data']['wheels']]
    fields={}
    for k in wheels[0] if wheels else []:
        x=[w[k] for w in wheels if isinstance(w[k],(int,float))];fields[k]=dict(valid=len(x),invalid=len(wheels)-len(x),nonzero=sum(v!=0 for v in x),range=[min(x),max(x)] if x else None)
    repeated=[r for r in ts if r['repeated_after'] is not None]
    mismatches=[r for r in repeated if r['repeated_after']['data']!=r['telemetry_after']['data']]
    speed_diff=[abs(r['body']['long_speed']-r['telemetry_after']['data']['speed']) for r in ts]
    commands={k:sum(abs(r['requested'][k]-r['telemetry_after']['data'][k])>1e-5 for r in ts) for k in ['throttle','brake','steer']}
    meta=dict(kind='offline_analysis_of_actual_CARLA',source='Privileged',carla_version='0.9.16',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),true_engine_wheel_api=bool(ts))
    write('reports/m6f_actuation_causal_probe.json',dict(**meta,trials=trials,fields=fields,body_vs_telemetry_speed_abs=quant(speed_diff),command_readback_mismatches=commands,
        repeat_no_tick={'n':len(repeated),'changed':len(mismatches)},scope='finite interventions, not complete root-cause proof; debug torque is not brake actuator torque; no substep measurement'))
    overhead=[]
    for r in all_rows:
        if r['phase'].startswith('overhead_'):
            t=r['timeline'];overhead.append(dict(block=r['phase'],on=r['true_engine_wheel_api'],work_ms=1000*(t['end']-t['start']),tick_ms=1000*(t['tick_end']-t['apply_end']),deadline=r['deadline_missed']))
    write('reports/m6f_runtime_overhead.json',dict(**meta,telemetry_rpc_ms=quant([r[k]['rpc_ms'] for r in ts for k in ['telemetry_before','telemetry_after']]),
        paired_blocks={b:dict(on=g[0]['on'],work_ms=quant([x['work_ms'] for x in g]),tick_ms=quant([x['tick_ms'] for x in g]),deadline_misses=sum(x['deadline'] for x in g)) for b in dict.fromkeys(x['block'] for x in overhead) for g in [[x for x in overhead if x['block']==b]]},
        controls=[dict(trial=t['path'],mode=t['config']['mode']) for t in trials],strict_20Hz='Open_not_driver_regression',limitation='Extra RPC benchmark includes two calls and bracketing/serialization; no Planner, no shared GPU model'))
    sham=[t for t in trials if '/sham/' in t['path']]
    write('reports/m6f_physics_sham_comparison.json',dict(**meta,status='Executed' if sham else 'NotRun',trials=sham,design='same-version original/sham/half-brake; order reversed second repeat',P1_status='NoGo_historical_unchanged'))

if __name__=='__main__':main()
