from types import SimpleNamespace as S
import numpy as np
from simulator.m8j13_candidate_audit import CandidateInbox, deadline_decision, marker_pixels


def test_callback_independently_copies_pixels_before_mutation():
    raw=bytearray([1,2,3,255]);obj=S(frame=3,timestamp=.15,width=1,height=1,
        raw_data=raw,transform=S(get_matrix=lambda:np.eye(4)))
    inbox=CandidateInbox(('test',1,'generation'));inbox.callback('front',42)(obj)
    raw[0]=99;payload=next(inbox.drain())
    assert payload.bgra==bytes([1,2,3,255]) and payload.stamp.frame==3


def test_wall_and_sim_expiry_reject_without_actuation():
    for d in [deadline_decision(1.,100.,1.,100.31),deadline_decision(1.,100.,1.31,100.1)]:
        assert not d['accepted'] and d['stop_command']==dict(steer=0.,throttle=0.,brake=1.)
        assert not d['actuated']
    assert deadline_decision(1.,100.,1.1,100.1)['accepted']


def test_fixed_marker_discriminates_snapshot_side():
    cfg=dict(roi_halfwidth_px=6,roi_halfheight_px=6,red_min=65,red_channel_ratio=1.3,min_red_pixels=20,opposite_max_ratio=.5)
    a=np.zeros((100,100,4),np.uint8);a[45:55,60:70,2]=255
    good=marker_pixels(a.tobytes(),100,100,np.eye(4),[10,3,0],[10,-3,0],90,cfg)
    bad=marker_pixels(a.tobytes(),100,100,np.eye(4),[10,-3,0],[10,3,0],90,cfg)
    assert good['matched'] and not bad['matched']
