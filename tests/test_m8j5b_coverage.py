import copy
import numpy as np
import pytest

from datasets.m8j5b_coverage import coverage_view,positive_eligibility


def fixture():
    sem=np.zeros((40,60),np.uint8);ins=np.zeros((40,60,4),np.uint8)
    sem[10:20,10:20]=14;ins[10:20,10:20,2]=14
    ins[10:20,10:20,1]=99 # Token intentionally differs from Actor.
    obj=dict(actor_id='123',source_class='vehicle',converted={'geometry_status':'Valid'},
        views={'CAM_FRONT':{'raw_projection':dict(status='Valid',polygon=[[8,8],[22,8],[22,22],[8,22]])}})
    return sem,ins,obj


def test_complete_visible_source_support_not_background_certificate():
    sem,ins,o=fixture();r=coverage_view(sem,ins,[o],'CAM_FRONT')
    assert not r['ignore'].any() and r['support'].sum()==100
    assert r['records'][0]['actor_identity']=='Unknown'
    assert r['stats']['safe_negative_pixels_certified']==0


def test_missing_target_ignored_without_inventing_3d():
    sem,ins,o=fixture();r=coverage_view(sem,ins,[],'CAM_FRONT',0)
    assert r['ignore'].sum()==100 and r['missing'].sum()==100
    assert not r['records'][0]['reliable_3d_positive']


def test_semantic_only_or_disagreement_not_background():
    sem,ins,o=fixture();ins[:]=0;r=coverage_view(sem,ins,[o],'CAM_FRONT',0)
    assert r['ignore'].sum()==100 and r['stats']['disagreement_foreground_pixels']==100


def test_single_pixel_kept_and_margin_clips_border():
    sem=np.zeros((12,12),np.uint8);ins=np.zeros((12,12,4),np.uint8)
    sem[0,0]=12;ins[0,0,2]=12
    r=coverage_view(sem,ins,[],'CAM_FRONT',4)
    assert r['ignore'].sum()==25 and len(r['records'])==1


def test_supported_region_partial_uncovered_pixels_still_ignore():
    sem,ins,o=fixture();o['views']['CAM_FRONT']['raw_projection']['polygon']=[[10,10],[15,10],[15,20],[10,20]]
    r=coverage_view(sem,ins,[o],'CAM_FRONT',0)
    assert r['support'].sum()==60 and r['ignore'].sum()==40


def test_train_rider_semantics_not_silently_dropped():
    sem,ins,o=fixture();sem[10:20,10:20]=17;ins[10:20,10:20,2]=17
    r=coverage_view(sem,ins,[o],'CAM_FRONT',0)
    assert r['ignore'].sum()==100 and r['records'][0]['semantic']==17


def test_bad_encoding_rejected():
    sem,ins,o=fixture()
    with pytest.raises(ValueError):coverage_view(np.stack([sem]*3,-1),ins,[o],'CAM_FRONT')
    with pytest.raises(ValueError):coverage_view(sem,ins[:,:,:3],[o],'CAM_FRONT')


def test_real_four_known_gaps_and_geometry_remain_separate():
    import json
    from tools.m8j5b_build_coverage import ROOT
    if not (ROOT/'case_reproduction.json').exists():pytest.skip('Private payload unavailable')
    rows=json.loads((ROOT/'case_reproduction.json').read_text())
    assert len(rows)==4 and all(r['region']['status']=='MissingSourceSupportCandidate' for r in rows)
    f=json.loads((ROOT/'sidecar/00124.json').read_text())
    assert not any(o['source_class']=='walker' for o in f['objects'])
    assert not f['frame_negative_enabled'] and not f['training_authorized']
