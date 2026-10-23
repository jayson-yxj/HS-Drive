"""Frozen guard replay on actual historical poses; no simulated recovery claims."""
import json,math
from pathlib import Path
import numpy as np
from simulator.m6d_state import SignedMotionState
from simulator.m6e_guard import DisplacementGuard,GuardConfig
from tools.m6e_common import write,asset,provenance


def inspect(path,config):
    rows=[json.loads(s) for s in path.read_text().splitlines()];g=DisplacementGuard(config);out=[]
    for q in rows:
        s=SignedMotionState(**q['signed']);p=np.array(q['measurement']['position'][:2])*[1,-1]
        out.append(g.update(s,p,[math.cos(s.yaw_rad),math.sin(s.yaw_rad)],s.timestamp_s))
    risk=next((i for i,x in enumerate(out) if x['risk']),None);em=next((i for i,x in enumerate(out) if x['emergency']),None)
    # Negatives: separate fresh, exact stationary hold phases without target reset
    # within each segment. Noisy/sideslip controls are synthetic tests separately.
    controls=[]
    phases=dict.fromkeys(q.get('phase') for q in rows)
    for phase in phases:
        if not phase or not (phase.startswith('hold') or phase=='guard_hold'):continue
        r=[q for q in rows if q['phase']==phase];v=[q['signed']['speed_norm_mps'] for q in r]
        if max(v,default=100)>.02:continue
        p=np.array([q['measurement']['position'][:2] for q in r])*[1,-1]
        if np.linalg.norm(p-p[0],axis=1).max()>.001:continue
        h=DisplacementGuard(config);events=[]
        for q,pos in zip(r,p):
            s=SignedMotionState(**q['signed']);d=h.update(s,pos,[math.cos(s.yaw_rad),math.sin(s.yaw_rad)],s.timestamp_s)
            if d['risk'] or d['emergency']:events.append(d)
        controls.append(dict(phase=phase,frames=len(r),false_risk_or_emergency=len(events),max_drift_m=float(np.linalg.norm(p-p[0],axis=1).max())))
    def event(i):
        return None if i is None else dict(decision=out[i],measurement=rows[i]['measurement'],phase=rows[i].get('phase'))
    return dict(name=path.parent.name,trace=asset(path),frames=len(rows),first_risk=event(risk),first_emergency=event(em),
        risk_to_emergency_s=None if risk is None or em is None else rows[em]['measurement']['t']-rows[risk]['measurement']['t'],stationary_controls=controls,
        protected_trajectory='NotRun_offline_replay_cannot_infer_intervention_outcome')


def main():
    cfg=GuardConfig(**json.loads(Path('configs/m6e/protocol.json').read_text())['guard'])
    records=[inspect(p,cfg) for p in sorted(Path('reports/m6d_probe_artifacts').glob('*/trace.jsonl'))]
    positives=[r for r in records if r['name'].startswith('guard_diagnostic')]
    controls=[c for r in records for c in r['stationary_controls']]
    write('reports/m6e_guard_historical_replay.json',dict(**provenance('offline_M6D_frozen_actual_poses'),records=records,
        known_failure_detection=sum(r['first_emergency'] is not None for r in positives),known_failures=len(positives),
        stationary_control_segments=len(controls),stationary_false_decisions=sum(c['false_risk_or_emergency'] for c in controls),
        limits='Thresholds selected using historical development; historical diagnostic failures are regression, not blind generalization. Stationary quiet segments selected from observed measurements; not a global false-positive rate.'))
    print('known failures detected',sum(r['first_emergency'] is not None for r in positives),'/',len(positives),'stationary false',sum(c['false_risk_or_emergency'] for c in controls))
if __name__=='__main__':main()
