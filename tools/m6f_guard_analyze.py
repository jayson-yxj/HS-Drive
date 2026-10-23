"""Audit frozen G1 independently of telemetry backend; raw protected dynamics."""
import json
from pathlib import Path
import numpy as np
from tools.m6f_analyze import write,quant
from tools.m47_common import dynamics_metrics


def main():
    trials=[]
    for p in sorted(Path('reports/m6f_guard_0915').glob('p*/result.json')):
        rs=[json.loads(s) for s in (p.parent/'trace.jsonl').read_text().splitlines()];release=[i for i,r in enumerate(rs) if r['phase'] in ['guard_release','protected_stop']]
        if not release:continue
        start=release[0];group=rs[start:];measure=[r['measurement'] for r in group];first=measure[0];yaw=np.radians(first['rotation'][2]);axis=np.array([np.cos(yaw),np.sin(yaw)])
        points=np.array([r['position'][:2] for r in measure]);position=(points-points[0])@axis
        events=json.loads(p.read_text())['guard_events'];interventions=[i for i,r in enumerate(group) if r['protection_applied']]
        protected=None
        if interventions:
            i=interventions[0];x=measure[max(0,i-2):];displacement=(points[i:]-points[i])@axis
            protected=dict(raw=dynamics_metrics(x),net_body_forward_travel_m=float(displacement[-1]),max_backward_after_emergency_m=float(max(0,-displacement.min())),path_length_after_m=float(np.linalg.norm(np.diff(points[i:],axis=0),axis=1).sum()),terminal_speed_mps=measure[-1]['speed'])
        quiet=[r for r in rs if r['phase']=='guard_hold'];initial_risk=quiet[0]['guard']['risk'] if quiet else False
        quiet_events=sum(r['guard']['risk'] and not initial_risk for r in quiet)
        trials.append(dict(path=str(p.parent),result=json.loads(p.read_text()),road=json.loads((p.parent/'road.json').read_text()),raw_release=dynamics_metrics(measure),max_release_backward_m=float(max(0,-position.min())),
            guard_events=events,protected=protected,quiet_hold_risk_ticks=quiet_events,release_emergency=len(interventions)>0,
            natural_dangerous_observation=any(r['guard']['reason']=='RollbackDisplacementConfirmed' for r in group),
            interpretation='Guard threshold crossing is corroborated observation, not independent ground-truth sensitivity denominator; small/nonnegative motion is negative support only'))
    write('reports/m6f_guard_validation.json',dict(kind='actual_CARLA0915_plus_synthetic_regression_tests',source='Privileged',carla_version='0.9.15',true_engine_wheel_api=False,
       frozen_guard='simulator/m6e_guard.py unchanged',trials=trials,actual_trials=len(trials),natural_positive_trials=sum(x['natural_dangerous_observation'] for x in trials),
       status='Partial_independent_positive_support_requires_review',threshold_search=False,assigned_velocity=False,default_promoted=False,
       synthetic_support='M6E tests: jitter/lateral/slow forward/recovery/goal/gear/nonmonotonic/gap/stale; tests do not certify live sensitivity',
       limitations=['Two independent spawn positions, two deterministic repeats, one map; not statistical generalization','Emergency/raw protected jerk is not normal comfort','No automatic guard release policy']))

if __name__=='__main__':main()
