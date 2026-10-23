import copy
import pytest
from diagnostics.m6g_events import command_events,flatten,braking_metrics


def rows():
    out=[]
    for i in range(6):
        req=dict(throttle=0.,brake=.01 if i>=2 else 0.,steer=0.,hand_brake=False,reverse=False)
        applied=dict(req);applied['brake']=.01 if i>=3 else 0.
        b=dict(frame=i+1,t=(i+1)*.05,long_speed=1.-i*.1,velocity=[1.,0,0],position=[0,0,0],long_acceleration_api=-2.,applied=applied)
        before=dict(b,frame=i,t=i*.05,long_speed=1.1-i*.1)
        out.append(dict(requested=req,body=b,body_before=before,phase='test',telemetry_after=None,timeline=dict(apply=i,apply_end=i+.001,end=i+.002)))
    return out


def test_delayed_request_is_aligned_to_first_applied_not_request():
    r=rows();e=command_events(r)[1]
    assert e['request_index']==2 and e['applied_index']==3 and e['delay_outer_ticks']==1


def test_censor_superseded_request_and_no_forward_matching_past_change():
    r=rows();r[3]['requested']['brake']=.2;r[4]['requested']['brake']=.2;r[5]['requested']['brake']=.2
    assert command_events(r)[1]['applied_index'] is None


def test_rebuild_invalidates_first_derivative_and_following_jerk():
    s=flatten(rows(),rebuild=True)
    assert s[0]['acceleration_50ms_mps2'] is None and s[1]['jerk_50ms_mps3'] is None
    assert s[2]['jerk_50ms_mps3']==pytest.approx(0,abs=1e-10)


def test_reject_frame_gap_and_wrong_telemetry():
    r=rows();r[2]['body']['frame']+=1
    with pytest.raises(ValueError):flatten(r)
    r=rows();r[2]['telemetry_after']=dict(data={},associated_frame=8,frame_before=8,frame_after=8)
    with pytest.raises(ValueError):flatten(r)


def test_missing_telemetry_not_invented():
    assert all(x['telemetry'] is None for x in flatten(rows()))


def test_constant_command_not_duplicate_events():
    r=rows();assert len(command_events(r))==2
