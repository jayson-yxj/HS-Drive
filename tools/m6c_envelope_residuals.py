"""Check remaining stopping distance at every held-out observed speed, no refit."""
import json
from pathlib import Path
import numpy as np
from simulator.m6c_envelope import StoppingEnvelope
from tools.m6c_common import write,provenance,asset


def main():
    model=json.loads(Path('configs/m6c/envelope_model.json').read_text());estimator=StoppingEnvelope(model);rows=[]
    d=json.loads(Path('reports/m6c_low_speed_dynamics.json').read_text())
    for trial in d['trials']:
        task=trial['task']
        if task['mode']!='stop' or task['split']!='validation' or trial['probe_start'] is None:continue
        r=[json.loads(s) for s in Path(trial['trace']['path']).read_text().splitlines()];start=trial['probe_start'];end=next((k for k in range(start,len(r)) if r[k]['phase']=='hold'),None)
        action={0.:'coast',.01:'brake_001',.05:'brake_005',1.:'emergency'}[task['brake']]
        xy=np.array([q['position'][:2] for q in r]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];stats={}
        for variant in ['B','C']:
            events=[];supported=0;failures=0;res=[]
            for i in range(start,end if end is not None else len(r)):
                value=estimator.estimate(r[i]['speed'],r[i]['road']['grade'],action,variant)
                if value.distance_m is None:continue
                supported+=1;actual=float(arc[end if end is not None else -1]-arc[i]);diff=value.distance_m-actual;res.append(diff)
                if diff< -1e-6:
                    failures+=1
                    if len(events)<5:events.append(dict(frame=r[i]['frame'],speed=r[i]['speed'],grade=r[i]['road']['grade'],predicted=value.distance_m,observed_remaining_lower_bound=actual))
            stats[variant]=dict(supported_samples=supported,underbound_samples=failures,min_predicted_minus_observed_m=min(res) if res else None,examples=events)
        rows.append(dict(name=trial['name'],action=action,stopped=end is not None,stats=stats,trace=trial['trace']))
    write('reports/m6c_envelope_residual_validation.json',dict(**provenance('offline_every_frame_on_independent_CARLA_validation_roads_no_refit'),source='Privileged',trials=rows,
        model=asset('configs/m6c/envelope_model.json'),limits='Within-trajectory samples strongly correlated; count trial/position support separately. Censored remaining travel is lower bound. No post-validation model revision.'))
    for v in ['B','C']:print(v,sum(q['stats'][v]['underbound_samples'] for q in rows),sum(q['stats'][v]['supported_samples'] for q in rows))

if __name__=='__main__':main()
