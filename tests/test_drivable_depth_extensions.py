import numpy as np
import pytest
from datasets.drivable import classify_lane,grid_centers,validate_arrays


def test_lane_semantics_unknown_boundary_height_junction():
    args=dict(junction=False,height_error=.1,edge_clearance=1.)
    assert classify_lane('Driving',**args)==1
    assert classify_lane('Sidewalk',**args)==0
    assert classify_lane('Parking',**args)==255
    assert classify_lane('Driving',**{**args,'junction':True})==255
    assert classify_lane('Driving',**{**args,'height_error':4})==255
    assert classify_lane('Sidewalk',**{**args,'edge_clearance':.49})==255
    cfg=dict(x_range_m=[-1,1],y_range_m=[-1,1],resolution_m=1.)
    p=grid_centers(cfg);np.testing.assert_allclose(p[1,0],[.5,-.5,0,1])
    label=np.array([[0,1,255]],np.uint8);valid=label!=255;rules=np.zeros_like(valid)
    validate_arrays(label,valid,rules,(1,3))
    with pytest.raises(ValueError):validate_arrays(label,np.ones_like(valid),rules,(1,3))
    with pytest.raises(ValueError):validate_arrays(label,valid,np.ones_like(valid),(1,3))


def test_masked_loss_does_not_learn_unknown_and_balances_ranges():
    torch=pytest.importorskip('torch')
    from training.losses import masked_drivable_loss,masked_depth_mean
    x=torch.tensor([0.,0.,5.],requires_grad=True);labels=torch.tensor([0.,1.,255.]);valid=labels!=255
    loss=masked_drivable_loss(x,labels,valid);loss.backward()
    assert x.grad[0]>0 and x.grad[1]<0 and x.grad[2]==0
    x.grad.zero_();masked_drivable_loss(x,labels,torch.zeros_like(valid)).backward();assert x.grad.abs().sum()==0
    errors=torch.tensor([1.,1.,1.,9.]);depth=torch.tensor([3.,3.,3.,70.]);v=torch.ones(4,dtype=torch.bool)
    assert masked_depth_mean(errors,depth,v,False)==3
    assert masked_depth_mean(errors,depth,v,True)==5


def test_static_labels_remain_supervision_only(m3_index):
    torch=pytest.importorskip('torch')
    from datasets.training import HSDriveDataset
    ds=HSDriveDataset(m3_index,'train',output_wh=(8,4),feature_stride=2)
    item=ds[0]
    assert not item.targets['drivable_valid'].any()
    assert not item.targets['drivable_rule_valid'].any()
    assert 'drivable_static' not in item.inputs.as_model_kwargs()


def test_drivable_head_backward_without_gt_forward_argument():
    torch=pytest.importorskip('torch');pytest.importorskip('torchvision')
    from models.depth_bev import DepthBEV
    from training.losses import masked_drivable_loss
    import yaml
    torch.set_num_threads(2)
    model=DepthBEV(yaml.safe_load(open('configs/bev.yaml')),drivable=True)
    bev=torch.randn(1,128,160,120,requires_grad=True)
    logits=model.drivable_head(bev)[:,0];labels=torch.zeros_like(logits);labels[:,:80]=1
    loss=masked_drivable_loss(logits,labels,torch.ones_like(labels,dtype=torch.bool));loss.backward()
    assert torch.isfinite(bev.grad).all() and bev.grad.abs().sum()>0
    assert model.drivable_head.weight.grad.abs().sum()>0


def test_raster_uses_hs_left_and_rejects_other_height(monkeypatch):
    import sys,types
    from datasets.drivable import rasterize
    class Location:
        def __init__(self,x,y,z):self.x,self.y,self.z=x,y,z
    monkeypatch.setitem(sys.modules,'carla',types.SimpleNamespace(Location=Location,LaneType=types.SimpleNamespace(Any=1)))
    class Map:
        def get_waypoint(self,p,**kw):
            center=Location(p.x,p.y,0 if p.x<10 else 4)
            t=types.SimpleNamespace(location=center,get_right_vector=lambda:Location(0,1,0))
            return types.SimpleNamespace(transform=t,is_junction=False,lane_width=4,lane_type='Driving' if p.y<0 else 'Sidewalk')
    cfg=dict(x_range_m=[-1,1],y_range_m=[-1,1],resolution_m=1.)
    result=rasterize(Map(),np.eye(4),cfg)
    np.testing.assert_array_equal(result['drivable_static'],[[0,1],[0,1]])
    pose=np.eye(4);pose[0,3]=20
    assert not rasterize(Map(),pose,cfg)['drivable_valid'].any()


def test_index_binds_static_labels_to_raw_frame_and_bev(m3_registry,tmp_path):
    import json,yaml
    from datasets.index import build_index
    from datasets.drivable import SCHEMA
    from simulator.recorder import atomic_json,file_hash
    registry=yaml.safe_load(m3_registry.read_text());scene=registry['scenes'][0]
    scene['drivable_labels']='static/train_a';root=tmp_path/'raw'/scene['drivable_labels'];root.mkdir(parents=True)
    geometry=tmp_path/'bev.yaml';geometry.write_text('shape: [160, 120]\n');registry['drivable_bev_config']='bev.yaml'
    labels=np.full((160,120),255,np.uint8);labels[:80]=1;labels[80:120]=0;valid=labels!=255
    refs={}
    for frame in sorted((tmp_path/'raw/train_a/frames').glob('*.json')):
        f=json.loads(frame.read_text());path=root/(frame.stem+'.npz')
        np.savez_compressed(path,drivable_static=labels,drivable_valid=valid,drivable_rule_valid=np.zeros_like(valid))
        refs[str(f['frame_id'])]={'path':path.name,'sha256':file_hash(path),'raw_frame_sha256':file_hash(frame)}
    manifest={'schema_version':SCHEMA,'status':'complete','raw_manifest_sha256':file_hash(tmp_path/'raw/train_a/manifest.json'),
        'bev_config_sha256':file_hash(geometry),'frames':refs}
    atomic_json(root/'manifest.json',manifest);m3_registry.write_text(yaml.safe_dump(registry))
    index=build_index(m3_registry,tmp_path/'index_labeled')
    row=next(r for r in index['samples'] if r['scene_id']=='train_a')
    inp=json.loads((tmp_path/'index_labeled'/row['input']['path']).read_text())
    target=json.loads((tmp_path/'index_labeled'/row['target']['path']).read_text())
    assert 'drivable' not in inp and target['drivable']['schema_version']==SCHEMA
    try:
        import torch
    except ImportError:
        torch=None
    if torch is not None:
        from datasets.training import HSDriveDataset
        from unittest.mock import patch
        ds=HSDriveDataset(tmp_path/'index_labeled','train',output_wh=(8,4),feature_stride=2)
        item=ds[0];assert item.targets['drivable_valid'].sum()==120*120
        with patch('numpy.load',side_effect=AssertionError('inference opened label')):
            inference=HSDriveDataset(tmp_path/'index_labeled','train',supervision=False,output_wh=(8,4),feature_stride=2)[0]
            for k in item.inputs:assert torch.equal(item.inputs[k],inference.inputs[k])
    refs['100']['raw_frame_sha256']='wrong';atomic_json(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Drivable frame identity'):build_index(m3_registry,tmp_path/'bad_index')


def test_drivable_prior_confusion_metrics():
    from tools.eval_drivable_prior import metrics
    result=metrics(np.array([[4,2],[1,3]]))
    assert result['valid_cells']==10
    assert result['iou_positive']==.5
    assert result['all_positive_iou']==.4
    assert metrics(np.zeros((2,2),np.int64))['iou_positive'] is None
