"""Dev-only fixed-bin maxima; never refit after opening validation."""
import json
from pathlib import Path
import numpy as np
from simulator.m6b_motion import LowSpeedController
from simulator.m6c_envelope import StoppingEnvelope
from tools.m6c_common import write,provenance,asset


def main():
    report=json.loads(Path('reports/m6c_low_speed_dynamics.json').read_text());protocol=json.loads(Path('configs/m6c/protocol.json').read_text())['envelope'];edges=protocol['speed_bins']
    all_samples=[];startup=[];records=[]
    for trial in report['trials']:
        task=trial['task'];rows=[json.loads(s) for s in Path(trial['trace']['path']).read_text().splitlines()];start=trial['probe_start']
        if start is None:continue
        first=next((k for k in range(start,len(rows)) if rows[k]['speed']>.2),None)
        if task['mode'] in ['startup','warm_start']:
            startup.append(dict(task=task,time_to_first_moving=None if first is None else rows[first]['t']-rows[start]['t'],
                initial_gear=rows[start]['applied']['gear'],first_moving_gear=None if first is None else rows[first]['applied']['gear'],
                max_acceleration=float(np.diff([r['long_speed'] for r in rows]).max()/.05),metrics=trial['metrics'],
                gear_transitions=[dict(t=r['t']-rows[0]['t'],speed=r['speed'],gear=r['applied']['gear']) for i,r in enumerate(rows) if i and r['applied']['gear']!=rows[i-1]['applied']['gear']],trace=trial['trace']))
        if task['mode']!='stop':continue
        end=next((k for k in range(start,len(rows)) if rows[k]['phase']=='hold'),None)
        action={0.:'coast',.01:'brake_001',.05:'brake_005',1.:'emergency'}[task['brake']]
        xy=np.array([r['position'][:2] for r in rows]);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        record=dict(name=trial['name'],split=task['split'],action=action,spawn=task['spawn'],seed=task['seed'],
            initial_speed=rows[start]['speed'],grade=rows[start]['road']['grade'],stopped=end is not None,
            observed_distance_m=float(s[(end if end is not None else len(rows)-1)]-s[start]),
            stop_time_s=None if end is None else rows[end]['t']-rows[start]['t'],final_speed=rows[-1]['speed'],trace=trial['trace'])
        records.append(record)
        if end is None:continue
        for i in range(start,end):
            all_samples.append(dict(split=task['split'],action=action,speed=rows[i]['speed'],grade=rows[i]['road']['grade'],distance=float(s[end]-s[i]),duration=rows[end]['t']-rows[i]['t'],name=trial['name']))
    tables={}
    for action in protocol['actions']:
        table=[];prior=None
        for lo,hi in zip(edges[:-1],edges[1:]):
            values=[v for v in all_samples if v['split']=='development' and v['action']==action and lo<v['speed']<=hi]
            if not values:table.append(None);continue
            cell=dict(max_distance_m=max(v['distance'] for v in values),max_duration_s=max(v['duration'] for v in values),min_grade=min(v['grade'] for v in values),samples=len(values))
            if prior:
                cell['max_distance_m']=max(cell['max_distance_m'],prior['max_distance_m']);cell['max_duration_s']=max(cell['max_duration_s'],prior['max_duration_s']);cell['min_grade']=min(cell['min_grade'],prior['min_grade'])
            table.append(cell);prior=cell
        # Unobserved holes inside measured speed range get next higher measured
        # maximum, never extrapolate above max supported interval.
        for i in range(len(table)-2,-1,-1):
            if table[i] is None and table[i+1] is not None:table[i]=dict(table[i+1],samples=0,filled_from_higher_bin=True)
        tables[action]=table
    model=dict(speed_edges=edges,tables=tables,**{k:protocol[k] for k in ['response_delay_s','unmodeled_distance_m','grade_uncertainty']},
        fit_split='development_only',development_positions=[27,126],source='Privileged',confidence='Empirical_upper_observed_bounds_not_distribution_free_guarantee')
    write('configs/m6c/envelope_model.json',model)
    estimator=StoppingEnvelope(model)
    for r in records:
        predictions={}
        for variant in ['A','B','C']:
            if variant=='A':value=LowSpeedController.coast_distance(r['initial_speed']);reason=None
            else:
                e=estimator.estimate(r['initial_speed'],r['grade'],r['action'],variant);value=e.distance_m;reason=e.reason
            predictions[variant]=dict(distance_m=value,unsupported_reason=reason,
                upper_bound_failed=(value is not None and r['observed_distance_m']>value+1e-6),
                actual_stop_error_m=None if value is None or not r['stopped'] else value-r['observed_distance_m'],
                no_stop_finite_claim=(not r['stopped'] and value is not None))
        r['predictions']=predictions
    summaries={}
    for split in ['development','validation']:
        summaries[split]={v:dict(trials=sum(r['split']==split for r in records),finite=sum(r['split']==split and r['predictions'][v]['distance_m'] is not None for r in records),
          bound_failures=sum(r['split']==split and r['predictions'][v]['upper_bound_failed'] for r in records),
          censored_finite_claims=sum(r['split']==split and r['predictions'][v]['no_stop_finite_claim'] for r in records)) for v in ['A','B','C']}
    write('reports/m6c_coast_brake_envelope.json',dict(**provenance('actual_CARLA_dev_only_fit_independent_road_validation'),source='Privileged',model=asset('configs/m6c/envelope_model.json'),summary=summaries,trials=records,
        limits='A is old coast model, including brake trials only as an action-mismatched reference; compare coast-to-coast separately. C adds predeclared delay/grade uncertainty. Val opens after frozen model write. Censored distance is a lower bound, not actual stop distance.'))
    write('reports/m6c_startup_modes.json',dict(**provenance('actual_CARLA_startup_mode_diagnostics'),source='Privileged',trials=startup,
        limits='manual1 vs auto changes commanded initialization only, not Physics. Gear association does not isolate drivetrain/contact causes; no candidate gear change authorized automatically.'))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
