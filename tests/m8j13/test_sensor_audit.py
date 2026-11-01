from types import SimpleNamespace as S
import pytest
from simulator.m8j13_sensor_audit import RawInbox, check_group


def image(frame=12, timestamp=.6):
    matrix = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    return S(frame=frame, timestamp=timestamp, transform=S(get_matrix=lambda:matrix))


def test_entry_values_survive_mutable_object_changes():
    inbox = RawInbox('run',7,'generation'); obj = image()
    inbox.callback('front',23)(obj)
    obj.frame=99; obj.timestamp=123.; obj.transform.get_matrix()[0][3]=80.
    stamp,_ = next(inbox.drain())
    assert (stamp.frame, stamp.timestamp_s, stamp.transform[0][3]) == (12,.6,0.)


def test_close_rejects_old_callback_and_clears_queue():
    inbox=RawInbox('run',7,'generation'); callback=inbox.callback('front',23)
    callback(image()); inbox.close(); callback(image())
    assert list(inbox.drain()) == []
    assert inbox.stats == {'close_queue_discarded':1,'old_generation_rejected':1}


def test_overflow_is_visible_without_blocking():
    inbox=RawInbox('run',7,'generation',capacity=1); cb=inbox.callback('front',23)
    cb(image()); cb(image()); assert inbox.stats['queue_overflow']==1


def record():
    inbox=RawInbox('run',7,'generation'); inbox.callback('front',23)(image())
    return next(inbox.drain())[0].record()


@pytest.mark.parametrize('field,value,error',[
    ('run','other','OldGeneration'), ('episode',8,'OldGeneration'),
    ('sensor_generation','old','OldGeneration'), ('actor_id',24,'WrongSensorInstance'),
    ('frame',13,'MixedFrame'), ('timestamp_s',.65,'MixedSimulationTimestamp')])
def test_invalid_provenance_is_rejected(field,value,error):
    r=record();r[field]=value
    out=check_group([r],dict(frame=12,elapsed_seconds=.6),('run',7,'generation'),{'front':23})
    assert not out['valid'] and error in out['errors']


def test_conflicting_duplicate_never_selects_first_as_valid():
    a=record();b=dict(a,timestamp_s=.65)
    out=check_group([a,b],dict(frame=12,elapsed_seconds=.6),('run',7,'generation'),{'front':23})
    assert not out['valid'] and 'DuplicateCamera' in out['errors']
    assert out['sources']=={} and len(out['candidates'])==2


def test_late_image_uses_original_snapshot_not_current_state():
    r=record();r['receive_monotonic_s']=90000.
    original=dict(frame=12,elapsed_seconds=.6,ego={'pose':'original'})
    out=check_group([r],original,('run',7,'generation'),{'front':23})
    assert out['valid'] and out['snapshot']['ego']['pose']=='original'


def test_missing_group_is_in_denominator():
    out=check_group([],dict(frame=12,elapsed_seconds=.6),('run',7,'generation'),{'front':23})
    assert not out['valid'] and out['errors']==['MissingCamera']


def test_live_rig_latches_duplicate_even_after_first_delivery():
    from simulator.m8j13_carla import Rig
    delivered=[]
    rig=Rig.__new__(Rig)
    rig.raw_inbox=RawInbox('run',7,'generation');rig.raw_records=[];rig.arrival_records=[]
    rig.raw_seen={};rig.raw_fault=None;rig.events=[];rig.sensor_ids={'front':23}
    rig.drop_camera=None;rig.sync=S(callback=lambda name:delivered.append)
    cb=rig.raw_inbox.callback('front',23)
    cb(image());rig.drain_raw();assert len(delivered)==1
    cb(image(timestamp=.65))
    with pytest.raises(ValueError,match='DuplicateRawSensorFrame'):rig.drain_raw()
    with pytest.raises(ValueError,match='DuplicateRawSensorFrame'):rig.drain_raw()
    assert len(rig.raw_records)==2 and rig.events[0]['previous_timestamp_s']==.6
