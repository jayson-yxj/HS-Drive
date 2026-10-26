import copy
import math

import pytest

from datasets.m8j5_supervision import classify_object, digest, hs_to_petr_center, validate_signoff


def object_fixture():
    return dict(geometry_status='Valid',model_class='car',label_status='Unknown',positive_supervision_candidate=False,
        yaw_only_comparable=True,center_hs_m=[10,0,1],dimensions_lwh_m=[4,2,2],yaw_ccw_rad=.2,
        reason='RenderIdentityConflict',views={'front':dict(status='Unknown',withdrawn_match={'render_token':1},
            projection=dict(status='Valid',bbox_xyxy=[600,400,800,700]))})


def test_identity_unknown_is_not_geometry_invalid():
    axes=classify_object(object_fixture())
    assert axes['identity_validity']=='Unknown' and axes['geometry_validity']=='Valid'
    assert axes['supervision_eligibility']['ignore'] and not axes['supervision_eligibility']['negative']


def test_auto_supported_still_requires_real_signoff():
    obj=object_fixture();v=obj['views']['front'];v.pop('withdrawn_match');v['status']='Valid'
    axes=classify_object(obj)
    assert axes['supervision_eligibility']['provisional_positive_candidate']
    assert not axes['supervision_eligibility']['positive']
    assert 'HumanSignoffMissing' in axes['supervision_eligibility']['reasons']


def test_no_synthetic_signoff_can_pass_missing_fields():
    with pytest.raises(ValueError):validate_signoff({'candidate_id':'a','candidate_sha256':'x'}, {})


def test_signoff_binding_and_representation_checks_synthetic_only():
    obj=object_fixture();obj['views']['front'].pop('withdrawn_match');obj['views']['front']['status']='Valid'
    c=dict(candidate_id='SYNTHETIC',object=obj,trust=classify_object(obj));c['candidate_sha256']=digest(c)
    e=dict(candidate_id='SYNTHETIC',candidate_sha256=c['candidate_sha256'],decision='ApprovePositive',reviewer='SYNTHETIC_TEST',
        reviewed_at='2026-10-26T00:00:00+00:00',review_source='SYNTHETIC_UNIT_TEST_NOT_ACTUAL_SIGNOFF',
        geometry_confirmed=True,class_confirmed=True,identity_confirmed=True,visible_confirmed=True,notes='')
    assert validate_signoff(c,e)
    assert validate_signoff(c,dict(e,reviewed_at='2026-10-26T00:00:00.123Z'))
    for patch in ({'candidate_sha256':'other'}, {'reviewer':None}, {'identity_confirmed':False}, {'visible_confirmed':'yes'}):
        with pytest.raises(ValueError):validate_signoff(c,dict(e,**patch))
    c['object']['yaw_only_comparable']=False
    with pytest.raises(ValueError):validate_signoff(c,e)
    c['candidate_sha256']=digest({k:v for k,v in c.items() if k!='candidate_sha256'})
    e['candidate_sha256']=c['candidate_sha256']
    with pytest.raises(ValueError):validate_signoff(c,e)


def test_center_reference_not_bottom_and_missing_velocity():
    o=object_fixture();b=hs_to_petr_center(o)
    assert b[:6]==[10,0,1,2,4,2] and b[7:]==[0,0]
    assert abs((b[6]+.2+math.pi/2+math.pi)%(2*math.pi)-math.pi)<1e-8


def test_review_gate_empty_real_record_does_not_invent_golden():
    from tools.m8j5_review_gate import review_gate
    r=review_gate([dict(candidate_id='pending')],[])
    assert r['golden_count']==r['safe_negative_count']==0
    assert r['pending_ids']==['pending']
    with pytest.raises(ValueError):review_gate([], [dict(candidate_id='missing')])
