import json
from pathlib import Path
import pytest
from tools.m8j13_common import atomic
from simulator.m8j13_snapshot_age import SnapshotAge,ledger_entries,consumed_completion


def test_pending_confirmation_continues_and_accumulates_without_refresh(tmp_path):
    row=dict(sequence_id='r',frame_id=10,timestamp_s=1.,input_capture_wall_s=10.)
    atomic(tmp_path/'current_snapshot.json',dict(frame=10,sim_s=1.))
    atomic(tmp_path/'tick_pending.json',dict(frame=11,pending=True))
    clock=[10.1]
    def sleep(dt):
        clock[0]+=dt
        if clock[0]>=10.12:
            atomic(tmp_path/'current_snapshot.json',dict(frame=11,sim_s=1.05));atomic(tmp_path/'tick_pending.json',dict(frame=11,pending=False))
    age=SnapshotAge(tmp_path,clock=lambda:clock[0],sleep=sleep)
    unconfirmed=age(row,10.05,phase='pending')
    assert unconfirmed['sim_age_s'] is None and unconfirmed['simulation_expired'] is None
    assert unconfirmed['sim_age_range_s']==[0.,.05] and unconfirmed['reason']=='SimulationSnapshotUnconfirmed'
    ok=age(row,10.05,wait_for_snapshot=True,phase='confirmed')
    assert ok['reason'] is None and ok['snapshot_confirmed'] and ok['snapshot_wait_ms']>=19
    assert ok['wall_age_s']>=.12 and row['input_capture_wall_s']==10.
    assert age.cumulative_wait_ms(10)==ok['snapshot_wait_ms']
    clock[0]=10.31;late=age(row,10.05,wait_for_snapshot=True)
    assert late['reason']=='PlanWallExpired' and late['snapshot_wait_ms']==0


def test_missing_snapshot_confirmation_and_expired_during_pending(tmp_path):
    row=dict(sequence_id='r',frame_id=10,timestamp_s=1.,input_capture_wall_s=10.)
    atomic(tmp_path/'current_snapshot.json',dict(frame=10,sim_s=1.))
    clock=[10.29]
    age=SnapshotAge(tmp_path,clock=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt))
    v=age(row,10.1,wait_for_snapshot=True)
    assert v['wall_expired'] and not v['snapshot_confirmed'] and v['simulation_expired'] is None
    atomic(tmp_path/'tick_pending.json',dict(frame=10,pending=False));clock[0]=10.32
    assert age(row,10.1,wait_for_snapshot=True)['reason']=='PlanWallExpired'


def test_empty_ledger_and_zero_offline_recheck(tmp_path):
    assert ledger_entries(tmp_path/'not-created.jsonl')==[]
    c=json.loads(Path('configs/m8j13_per_camera_recheck.json').read_text());old=json.loads(Path('configs/m8j13_per_camera.json').read_text())
    assert c['offline_forward_cap']==0 and c['total_forward_cap']==24
    allowed={'private_root','offline_forward_cap','total_forward_cap','equivalence_evidence','admission_evidence','scope'}
    assert all(c[k]==v for k,v in old.items() if k not in allowed)
    assert json.loads(Path(c['equivalence_evidence']).read_text())['status']=='Passed'


def test_consumer_requires_checks_and_current_age_and_flags_only():
    row=dict(sequence_id='r',frame_id=10,timestamp_s=1.,input_capture_wall_s=10.)
    flags=dict(status='ShadowComplete',model_forward=True,finite=True,trajectory_valid=True,result_expired=False)
    assert consumed_completion(flags,row,12,1.1,10.2,10.21)['consumable_fresh']
    assert not consumed_completion(flags,row,17,1.35,10.2,10.21)['consumable_fresh']
    assert not consumed_completion(flags,row,12,1.1,10.29,10.31)['consumable_fresh']
    assert not consumed_completion(dict(flags,trajectory_valid=False),row,12,1.1,10.2,10.21)['consumable_fresh']
    with pytest.raises(ValueError):consumed_completion(dict(flags,throttle=1.),row,12,1.1,10.2,10.21)


def test_report_empty_run_does_not_invent_denominator(tmp_path):
    from tools.m8j13_recheck_report import summarize,ages
    root=tmp_path/'live';root.mkdir()
    s,_=summarize(root,{'candidate_count':120})
    assert s['attempts']==s['success']==s['HS_PID']['fresh']==0
    assert s['reached_candidates']==0 and s['not_reached']==120
    assert s['timings']['all']['end_to_HS_PID_ms'] is None
    result=ages([dict(reason=None,wall_expired=False,simulation_expired=False,snapshot_confirmed=True),dict(reason='SimulationSnapshotUnconfirmed',wall_expired=False,simulation_expired=None,snapshot_confirmed=False),dict(reason='PlanWallExpired',wall_expired=True,simulation_expired=False,snapshot_confirmed=True)])
    assert result['fresh']==1 and result['wall_expired']==1 and result['simulation_unconfirmed']==1 and result['simulation_expired']==0
