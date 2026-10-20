import numpy as np
from datasets.schema import SensorSample
from simulator.synchronizer import FrameSynchronizer


def sample(name, frame, timestamp=None):
    return SensorSample(name, frame, frame*.05 if timestamp is None else timestamp,
                        1, 1, bytes([1,2,3,255]), np.eye(4))


def test_out_of_order_delay_snapshot_join():
    sync = FrameSynchronizer(('a','b'), timeout_ticks=4)
    for frame in range(100, 103):
        sync.add_snapshot(frame,frame*.05,{'pose_frame':frame},[])
    for name, frame in [('b',102),('a',100),('b',100),('a',102)]:
        sync.callback(name)(sample(name,frame))
    frames = sync.drain()
    assert [f.frame_id for f in frames] == [100,102]
    assert [f.ego['pose_frame'] for f in frames] == [100,102]


def test_missing_and_completely_absent_frames_are_reported():
    events = []
    sync = FrameSynchronizer(('a','b'),timeout_ticks=3,on_event=events.append)
    for frame in range(10,18):
        sync.add_snapshot(frame,frame*.05,{},[])
        if frame == 10:
            for name in ('a','b'):
                sync.callback(name)(sample(name,frame))
        if frame == 14:
            sync.callback('a')(sample('a',frame))
        sync.drain()
    dropped = {e['frame_id']: e for e in events if e['reason']=='dropped_frame'}
    assert dropped[12]['missing'] == ['a','b']
    assert dropped[14]['missing'] == ['b']


def test_overflow_duplicate_and_late_are_bounded():
    sync = FrameSynchronizer(('a','b'),queue_size=2,timeout_ticks=2,max_frames=2)
    sync.add_snapshot(1,.05,{},[])
    for _ in range(3):
        sync.callback('a')(sample('a',1))
    sync.drain()
    assert sync.stats['queue_overflow']==1
    assert sync.stats['duplicate_sensor']==1
    for f in range(2,10):
        sync.add_snapshot(f,f*.05,{},[])
        sync.drain()
    sync.callback('b')(sample('b',1))
    sync.drain()
    assert sync.stats['late_sensor']==1
    assert len(sync.pending)<=2
    assert len(sync.snapshots)<=3


def test_mixed_timestamps_never_commit():
    sync = FrameSynchronizer(('a','b'))
    sync.add_snapshot(1,.05,{},[])
    sync.callback('a')(sample('a',1))
    sync.callback('b')(sample('b',1,.15))
    assert sync.drain()==[]
    assert sync.stats['invalid_frame']==1


def test_tick_gap_detected():
    import pytest
    sync = FrameSynchronizer(('a',))
    sync.add_snapshot(1,.05,{},[])
    with pytest.raises(RuntimeError):
        sync.add_snapshot(3,.15,{},[])


def test_nonfinite_timestamp_rejected():
    sync=FrameSynchronizer(('a',))
    sync.add_snapshot(1,.05,{},[])
    sync.callback('a')(sample('a',1,float('nan')))
    assert not sync.drain()
    assert sync.stats['invalid_frame']==1


def test_later_complete_waits_for_earlier_then_recovers():
    sync=FrameSynchronizer(('a','b'),timeout_ticks=4)
    sync.add_snapshot(10,.5,{'f':10},[])
    sync.callback('a')(sample('a',10))
    assert sync.drain()==[]
    sync.add_snapshot(11,.55,{},[])
    sync.add_snapshot(12,.6,{'f':12},[])
    for name in ('a','b'):
        sync.callback(name)(sample(name,12))
    assert sync.drain()==[]
    sync.callback('b')(sample('b',10))
    assert [f.frame_id for f in sync.drain()]==[10,12]


def test_delayed_warmup_frames_cannot_set_capture_phase():
    sync=FrameSynchronizer(('a','b'))
    sync.begin_interval(100)
    sync.add_snapshot(100,5.0,{},[])
    sync.callback('a')(sample('a',99))
    for name in ('a','b'):sync.callback(name)(sample(name,100))
    assert [b.frame_id for b in sync.drain()]==[100]
    assert sync.stats['pre_interval_sensor']==1
    assert sync.stats['dropped_frame']==0


def test_explicit_cadence_preserves_ids_and_discards_only_unselected_ticks():
    sync = FrameSynchronizer(('a','b'), cadence_ticks=2)
    sync.begin_interval(100, explicit_cadence=True)
    received = []
    for frame in range(100,105):
        sync.add_snapshot(frame,frame*.05,{'pose_frame':frame},[])
        for name in ('a','b'):sync.callback(name)(sample(name,frame))
        received.extend(sync.drain())
    assert [b.frame_id for b in received] == [100,102,104]
    assert [b.ego['pose_frame'] for b in received] == [100,102,104]
    assert sync.stats['subsampled_sensor'] == 4
    assert sync.stats['dropped_frame'] == 0


def test_explicit_cadence_rejects_missing_first_frame_and_shifted_sensor():
    events=[]
    sync = FrameSynchronizer(('a','b'), cadence_ticks=2, timeout_ticks=3,on_event=events.append)
    sync.begin_interval(100, explicit_cadence=True)
    received=[]
    for frame in range(100,106):
        sync.add_snapshot(frame,frame*.05,{},[])
        # b at 101 must not fill the missing b at 100, nor change the phase.
        if frame==100:sync.callback('a')(sample('a',frame))
        if frame==101:sync.callback('b')(sample('b',frame))
        if frame>=102:
            for name in ('a','b'):sync.callback(name)(sample(name,frame))
        received.extend(sync.drain())
    assert [b.frame_id for b in received] == [102,104]
    assert any(e['reason']=='dropped_frame' and e['frame_id']==100 for e in events)
