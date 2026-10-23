from dataclasses import replace
from pathlib import Path
import json,math
import numpy as np
import pytest
from simulator.m6d_state import SignedMotionState
from simulator.m6e_guard import DisplacementGuard


def state(i,v=0.,lateral=0.,route=None):
    return SignedMotionState(v,lateral,math.hypot(v,lateral),v if route is None else route,1,0,1,False,False,0,0,i*.05,i+1)


def test_exact_recorded_m6d_failure_and_no_immediate_tiny_negative_brake():
    path=Path('reports/m6d_probe_artifacts/guard_diagnostic_p27_s201_warm_guarded_auto_015/trace.jsonl')
    rows=[json.loads(s) for s in path.read_text().splitlines()];guard=DisplacementGuard();decisions=[]
    for r in rows:
        s=SignedMotionState(**r['signed']);p=np.array(r['measurement']['position'][:2])*[1,-1];t=[math.cos(s.yaw_rad),math.sin(s.yaw_rad)]
        decisions.append(guard.update(s,p,t,s.timestamp_s))
    first=next(i for i,x in enumerate(decisions) if x['emergency'])
    assert decisions[first]['reason']=='RollbackDisplacementConfirmed'
    assert .008<=decisions[first]['net_backward_m']<.012
    assert any(x['risk'] and not x['emergency'] for x in decisions[:first])
    assert decisions[first-1]['timestamp_s']<decisions[first]['timestamp_s']

@pytest.mark.parametrize('case',['noise','lateral','slope_hold','forward','goal_advance'])
def test_no_false_emergency_on_controls(case):
    guard=DisplacementGuard()
    for i in range(100):
        if case=='noise':s=state(i,(-1)**i*.01);p=[(-1)**i*.0001,0]
        elif case=='lateral':s=state(i,0,.1,0);p=[0,i*.005]
        elif case=='forward':s=state(i,.2);p=[i*.01,0]
        else:s=replace(state(i),slope=.02,signed_goal_distance_m=1+i if case=='goal_advance' else 0);p=[0,0]
        d=guard.update(s,p,[1,0],s.timestamp_s);assert not d['risk'] and not d['emergency']


def test_frame_loss_nonmonotonic_and_stale_fail_closed():
    for s,reason in [(state(3),'SnapshotGap'),(state(0),'NonmonotonicSnapshot')]:
        g=DisplacementGuard();g.update(state(0),[0,0],[1,0],0)
        assert g.update(s,[0,0],[1,0],s.timestamp_s)['reason']==reason
    assert DisplacementGuard().update(state(0),[0,0],[1,0],.2)['emergency']


def test_gear_and_target_changes_do_not_reset_risk_or_clear_emergency():
    g=DisplacementGuard()
    for i in range(8):
        s=replace(state(i,-.08 if i%2 else .01),gear=-1 if i%2 else 1,signed_goal_distance_m=100*i)
        d=g.update(s,[-i*.002,0],[1,0],s.timestamp_s)
    assert d['emergency']
    s=state(8,.2);assert g.update(s,[1,0],[1,0],s.timestamp_s)['emergency']


def test_risk_clears_only_after_forward_recovery_and_hysteresis():
    g=DisplacementGuard()
    for i,p in enumerate([0,-.002,-.004]):d=g.update(state(i,-.04),[p,0],[1,0],i*.05)
    assert d['risk'] and not d['emergency']
    for i in range(3,30):d=g.update(state(i,.1),[.001*(i-3),0],[1,0],i*.05)
    assert not d['risk'] and not d['emergency']
