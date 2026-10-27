"""Finite checks of new aggregation, using synthetic boxes only."""
import numpy as np
import pytest
from tools.m8j10_report import paired_classes,anomalies
from tools.m8j8_report import paired


def record(xyz):
    gt=np.array([[0,0,0,2,4,1,0,0,0],[10,0,0,2,4,1,0,0,0]],dtype=np.float32)
    boxes=gt.copy()
    boxes[:,:3]=xyz
    return dict(boxes=boxes,scores=np.array([.9,.8]),labels=np.array([0,0]),gt=gt,gt_labels=np.array([0,0]))


def test_reference_candidate_pairing_not_matched_population_average():
    a=record([[.8,0,0],[15,0,0]])
    c=record([[.2,0,0],[10.1,0,0]])
    result=paired_classes([a],[c],['car','pedestrian'])
    assert result['car']['common']==1 and result['car']['B_only']==1
    assert result['car']['paired_errors']['xy_m_B_minus_A']['mean']<0
    assert result['pedestrian']['common']==0
    assert result['pedestrian']['paired_errors']=={}


def test_changed_GT_not_comparable():
    a=record([[0,0,0],[10,0,0]]);c=record([[0,0,0],[10,0,0]])
    c['gt'][0,0]=1
    with pytest.raises(AssertionError):paired([a],[c],['car'])


def test_vertical_center_and_height_are_different():
    r=record([[0,0,2],[10,0,0]])
    r['boxes'][0,5]=3
    s=anomalies([r],['car'])
    assert s['counts']['matched']==2
    assert s['counts']['abs_dz_gt2m']==1
    assert s['counts']['abs_dh_gt2m']==0
    assert s['counts']['abs_dh_gt1m']==1
