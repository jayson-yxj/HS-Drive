import json
from pathlib import Path
import numpy as np
import pytest
from datasets.m8j12_contract import CAMERAS
from simulator.m8j13_partial_transport import PartialSlots
from simulator.m8j13_transport import SIZE


def fixture(tmp_path):
    s=PartialSlots(tmp_path/'slots',create=True);identity=dict(run='r',episode=1,generation='g',sensors={n:i for i,n in enumerate(CAMERAS)})
    token=s.begin('request',identity,10,1.,10.)
    stamps={n:dict(run='r',episode=1,sensor_generation='g',camera=n,actor_id=i,frame=10,timestamp_s=1.,transform=np.eye(4).tolist()) for i,n in enumerate(CAMERAS)}
    row=dict(sequence_id='r',frame_id=10,timestamp_s=1.,input_capture_wall_s=10.,imu_frame=10,imu_timestamp_s=1.,ego_world_matrix=np.eye(4).tolist(),speed_mps=0.,camera_frames={n:10 for n in CAMERAS},camera_timestamps_s={n:1. for n in CAMERAS},camera_world_matrices={n:np.eye(4).tolist() for n in CAMERAS})
    snap=dict(frame=10,sim_s=1.,pose=np.eye(4).tolist(),speed=0.)
    return s,identity,token,stamps,row,snap


def test_partial_camera_ready_does_not_mean_whole_input_ready(tmp_path):
    s,i,t,st,r,snap=fixture(tmp_path)
    with pytest.raises(RuntimeError,match='CameraNotPublished'):s.read_camera(t,CAMERAS[0])
    s.publish_camera(t,st[CAMERAS[0]],bytes([7])*(SIZE//6))
    im,_=s.read_camera(t,CAMERAS[0]);assert im.shape==(396,704,3) and np.all(im==7)
    with pytest.raises(RuntimeError,match='IncompleteObservation'):s.committed_row(t)
    with pytest.raises(RuntimeError,match='Incomplete'):s.commit(t,r,snap,10.2,1.2)
    for n in CAMERAS[1:]:s.publish_camera(t,st[n],bytes([8])*(SIZE//6))
    s.commit(t,r,snap,10.2,1.2);assert s.committed_row(t)==r
    s.close()


def test_duplicate_generation_clock_and_timeout_rejected(tmp_path):
    s,i,t,st,r,snap=fixture(tmp_path);n=CAMERAS[0];raw=bytes(SIZE//6)
    for field,value,match in [('sensor_generation','old','Generation'),('frame',9,'Clock'),('timestamp_s',1.05,'Clock'),('actor_id',99,'Instance')]:
        stamp=dict(st[n]);stamp[field]=value
        with pytest.raises(ValueError,match=match):s.publish_camera(t,stamp,raw)
    for n in CAMERAS:s.publish_camera(t,st[n],raw)
    with pytest.raises(RuntimeError,match='Duplicate'):s.publish_camera(t,st[n],raw)
    with pytest.raises(RuntimeError,match='InputWaitExpired'):s.commit(t,r,snap,10.251,1.2)
    with pytest.raises(RuntimeError,match='SimulationExpired'):s.commit(t,r,snap,10.2,1.35)
    bad=dict(snap,speed=1.)
    with pytest.raises(ValueError,match='Snapshot'):s.commit(t,r,bad,10.2,1.2)
    s.close()


def test_cancelled_half_input_capacity_reuse_cannot_revive(tmp_path):
    s,i,t,st,r,snap=fixture(tmp_path);t2=s.begin('second',i,11,1.05,10.05)
    assert s.begin('third',i,12,1.1,10.1) is None
    s.cancel(t)
    with pytest.raises(RuntimeError,match='Cancelled'):s.committed_row(t)
    s.release('request');t3=s.begin('third',i,12,1.1,10.1)
    with pytest.raises(RuntimeError,match='Stale'):s.header(t)
    assert s.header(t3)['frame']==12
    s.close();assert not s.root.exists()


def test_actual_relay_queues_without_transforming_callback_payload(tmp_path):
    from queue import Queue
    from types import SimpleNamespace as NS
    import time
    from tools.m8j13_camera_live import EarlyRelay
    s,i,t,st,r,snap=fixture(tmp_path);s.release('request')
    class Inbox:
        overflow=0
        def __init__(self):self.q=Queue()
        def drain(self):
            while not self.q.empty():yield self.q.get_nowait()
        def close(self):pass
    inbox=Inbox();relay=EarlyRelay(inbox,s,i,tmp_path);relay.reserve_frame(10,10.,0)
    payload=NS(stamp=NS(record=lambda:st[CAMERAS[0]]),bgra=bytes(SIZE//6),copied_monotonic_s=10.1)
    inbox.q.put(payload);until=time.monotonic()+1
    while relay.queue.empty() and time.monotonic()<until:time.sleep(.005)
    assert list(relay.drain())==[payload]
    assert relay.frames[10]['status']=='Partial'
    relay.reject(10);assert relay.frames[10]['status']=='Cancelled'
    relay.close();s.close()


def test_frozen_camera_budget_and_original_deadlines():
    c=json.loads(Path('configs/m8j13_per_camera.json').read_text())
    assert c['max_starts']==1 and c['max_active_ticks']==800 and c['total_forward_cap']==32
    assert c['offline_forward_cap']==8 and c['max_forward_attempts']==24
    assert c['candidate_count']==120 and c['candidate_interval_ticks']==4 and c['cpu_camera_units']==2
    assert c['input_wait_wall_s']==.25 and c['max_plan_age_wall_s']==c['max_plan_age_sim_s']==.3
    assert not c['model_actuation_allowed'] and not c['training']


def test_live_tuple_transform_matches_json_list_without_tolerance(tmp_path):
    s,i,t,st,r,snap=fixture(tmp_path)
    for n in CAMERAS:
        st[n]['transform']=tuple(tuple(v for v in row) for row in st[n]['transform'])
        r['camera_world_matrices'][n]=st[n]['transform']
        s.publish_camera(t,st[n],bytes(SIZE//6))
    assert isinstance(r['camera_world_matrices'][CAMERAS[0]],tuple)
    serialized=json.loads((s.root/('0.%s.json'%CAMERAS[0])).read_text())
    assert isinstance(serialized['stamp']['transform'],list)
    changed=dict(r);changed['camera_world_matrices']=dict(r['camera_world_matrices'])
    m=np.asarray(r['camera_world_matrices'][CAMERAS[0]]).copy();m[0,3]=1e-12;changed['camera_world_matrices'][CAMERAS[0]]=m.tolist()
    with pytest.raises(ValueError,match='RowCameraBinding'):s.commit(t,changed,snap,10.2,1.2)
    s.commit(t,r,snap,10.2,1.2)
    assert s.committed_row(t)['camera_world_matrices'][CAMERAS[0]]==np.eye(4).tolist()
    s.close()


def test_pending_or_missing_snapshot_never_qualifies_for_current_age():
    from simulator.m8j13_shadow_contract import snapshot_is_current
    a=dict(pending=False,frame=10,wall_s=1.)
    assert snapshot_is_current(a,dict(a),dict(frame=10))
    assert not snapshot_is_current(None,None,dict(frame=10))
    assert not snapshot_is_current(dict(a,pending=True),dict(a,pending=True),dict(frame=10))
    assert not snapshot_is_current(a,dict(a,frame=11),dict(frame=11))
    assert not snapshot_is_current(a,a,dict(frame=9))
