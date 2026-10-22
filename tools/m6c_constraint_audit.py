"""Post-run physical feasibility support and uncontrolled rollback violations."""
import json
from pathlib import Path
import numpy as np
from tools.m6c_common import write,provenance,asset


def main():
    out=[]
    for path in sorted(Path('reports/m6c_trials').glob('*/result.json')):
        d=json.loads(path.read_text());r=[json.loads(s) for s in (path.parent/'trace.jsonl').read_text().splitlines()];violations=[];holds=[]
        for i,q in enumerate(r):
            signed=q['measurement']['long_speed'];decision=q['command'].get('envelope_decision')
            if signed<-.05:
                violations.append(dict(frame=q['frame'],signed_speed=signed,norm_speed=q['measurement']['speed'],mode=q['command'].get('mode'),geometry=q['geometry'],decision=decision,reason=q['reason']))
            if q['command'].get('transition_reason')=='near_goal_and_measured_settled':holds.append(dict(frame=q['frame'],actual_remaining=q['geometry']['available_clearance_to_safe_stop'],speed=q['measurement']['speed']))
        out.append(dict(name=path.parent.name,source='Privileged',config=d['config'],trace=asset(path.parent/'trace.jsonl'),
            valid_D_contract=d['config']['arm']!='D' or (path.parent.name.endswith('D2') or path.parent.name.startswith('runtime_') and path.parent.name.endswith('2')),
            unsupported_backward_ticks=len(violations),examples=violations[:3],
            finite_envelope_during_backward_ticks=sum(v['decision'] is not None and v['decision']['estimated_stopping_distance_m'] is not None for v in violations),
            hold_entries=holds,hold_farther_than_true_point35=sum(v['actual_remaining']>.35+1e-6 for v in holds),
            speed_max=max(q['measurement']['speed'] for q in r),nominal_max=d['config']['speed'],
            boundary_ticks=sum(q['boundary'] for q in r)))
    write('reports/m6c_motion_support_audit.json',dict(**provenance('offline_actual_CARLA_motion_contract_limits'),source='Privileged',trials=out,
        limits='Forward-only empirical table cannot certify reverse rolling even when speed norm is in training range. Unsupported negative signed speed is a candidate failure, not a new filtered exclusion. HOLD distance uses actual path/OBB available margin.'))

if __name__=='__main__':main()
