"""Offline actual vs planned bounds and HOLD drift; never suppress raw peaks."""
import json
from pathlib import Path
import numpy as np
from tools.m6b_common import write,provenance,asset
from tools.m47_common import dynamics_metrics


def main():
    out=[]
    for p in sorted(Path('reports/m6b_trials').glob('final_*/result.json')):
        d=json.loads(p.read_text());r=[json.loads(s) for s in (p.parent/'trace.jsonl').read_text().splitlines()]
        speed=np.array([q['measurement']['speed'] for q in r]);xyz=np.array([q['measurement']['position'] for q in r])
        modes=[q['command'].get('mode') for q in r];holds=[];start=None
        for k in range(len(r)+1):
            hold=k<len(r) and modes[k]=='HOLD'
            if hold and start is None:start=k
            elif not hold and start is not None:
                last=k-1;sub=r[start:k]
                holds.append(dict(start_frame=r[start]['frame'],end_frame=r[last]['frame'],entry_speed=float(speed[start]),
                    entry_signed_remaining=r[start]['current_remaining_m'],duration_s=(last-start)*.05,
                    max_displacement_m=float(np.linalg.norm(xyz[start:k]-xyz[start],axis=1).max()),
                    max_speed=float(speed[start:k].max()),dynamics=dynamics_metrics([q['measurement'] for q in sub]) if len(sub)>=8 else None))
                start=None
        out.append(dict(name=p.parent.name,source='Privileged',max_actual_speed_mps=float(speed.max()),
            requested_speed_limit_mps=d['config']['speed'],over_speed_limit_ticks=int((speed>d['config']['speed']+.05).sum()),
            overspeed_tolerance_for_reporting_mps=.05,hold_intervals=holds,trace=asset(p.parent/'trace.jsonl')))
    write('reports/m6b_motion_envelope.json',dict(**provenance('offline_actual_CARLA_motion_bound_audit'),source='Privileged',trials=out,
        limitation='0.05m/s is reporting resolution, not a new acceptance threshold. Planner cruise cap need not be an attainable instantaneous physical speed cap; actual overshoot retained. HOLD drift intervals not reverse parking validation.'))

if __name__=='__main__':main()
