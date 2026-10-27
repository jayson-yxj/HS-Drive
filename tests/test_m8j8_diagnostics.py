import numpy as np
from training.m8j8_diagnostics import vertical_errors,duplicate_groups
from training.m8j7_metrics import match


def record():
    gt=np.array([[0.,0.,-1.,1.,2.,2.,0.,0.,0.]])
    boxes=np.repeat(gt,3,axis=0);boxes[:,0]=[0.,.1,.2]
    return dict(boxes=boxes,scores=np.array([.9,.8,.7]),labels=np.array([0,0,1]),
                gt=gt,gt_labels=np.array([0]))


def test_center_and_height_are_independent():
    r=record();r['boxes'][0,2]=-.5;r['boxes'][0,5]=3.
    e=vertical_errors(r,[(0,0)])
    assert e['signed_dz_m']['mean']==1. and e['signed_dh_m']['mean']==1.


def test_half_height_shift_preserves_center():
    r=record();r['boxes'][0,2]=-2;r['boxes'][0,5]=4.
    e=vertical_errors(r,[(0,0)])
    assert e['abs_dz_m']['mean']==0. and e['abs_dh_m']['mean']==2.


def test_duplicates_and_cross_class_query_are_distinct():
    r=record();d=duplicate_groups(r,np.array([11,12,11]))
    assert d['duplicate_predictions']==1
    assert d['details'][0]['query_index']==12
    assert d['details'][0]['matched_query_indices']==[11]
    assert len(d['cross_class_same_query_groups'])==1


def test_ambiguous_neighboring_ground_truth_is_reported():
    r=record();r['gt']=np.repeat(r['gt'],2,axis=0);r['gt'][1,0]=.5;r['gt_labels']=np.array([0,0])
    r['labels'][:]=0
    d=duplicate_groups(r,np.arange(3))
    assert d['duplicate_predictions']==1 and d['ambiguous_duplicate_predictions']==1


def test_empty_matches_not_fake_zero_error():
    e=vertical_errors(record(),[])
    assert e['abs_dh_m']['mean'] is None and e['abs_dz_m']['count']==0


def test_official_encoding_decode_center_and_corner_contract():
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox,denormalize_bbox
    center=torch.tensor([[1.,2.,3.,.7,.9,1.6,-1.,0.,0.]])
    gt=LiDARInstance3DBoxes(center,box_dim=9,origin=(.5,.5,.5))
    target=torch.cat((gt.gravity_center,gt.tensor[:,3:]),dim=1)
    encoded=normalize_bbox(target,[-51.2,-51.2,-5,51.2,51.2,3])
    assert torch.allclose(encoded[:,4],center[:,2])
    assert torch.allclose(encoded[:,5].exp(),center[:,5])
    decoded=denormalize_bbox(encoded,None)
    decoded[:,2]-=decoded[:,5]/2
    obj=LiDARInstance3DBoxes(decoded,box_dim=9)
    assert torch.allclose(obj.gravity_center,gt.gravity_center)
    assert torch.allclose(obj.tensor,gt.tensor)
    assert torch.allclose(obj.corners[:,:,2].min(1).values,obj.tensor[:,2])
    assert torch.allclose(obj.corners[:,:,2].max(1).values,obj.tensor[:,2]+obj.tensor[:,5])
