import numpy as np
import pytest
from datasets.road_supervision import project_ribbons,validate_road_arrays
from simulator.traffic_rules import signal_stop_target,validate_traffic,SCHEMA


def base():
    shape=(160,120);return dict(drivable_static=np.full(shape,255,np.uint8),drivable_valid=np.zeros(shape,bool),drivable_rule_valid=np.zeros(shape,bool))


def config():return dict(x_range_m=[-20,60],y_range_m=[-30,30],resolution_m=.5)


def ribbons(z=0,direction=(1,0,0)):
    return dict(polygons=np.array([[[0,-2,z],[0,2,z],[10,2,z],[10,-2,z]]],float),headings=np.array([direction],float),junction=np.array([True]))


def test_junction_is_ribbon_not_bounding_box_or_permission():
    out=project_ribbons(ribbons(),np.eye(4),config(),base())
    assert out['junction_support'][50,60] and out['drivable_static'][50,60]==1
    assert out['drivable_static'][50,75]==255
    assert not out['drivable_rule_valid'].any()
    np.testing.assert_allclose(out['lane_direction'][:,50,60],[1,0])
    assert out['lane_direction_valid'][50,60]


def test_hs_direction_conflict_vertical_layer_and_mask_validation():
    r=ribbons(direction=(0,1,0));out=project_ribbons(r,np.eye(4),config(),base())
    np.testing.assert_allclose(out['lane_direction'][:,50,60],[0,-1])
    a=ribbons();b=ribbons(direction=(-1,0,0))
    merged={k:np.concatenate([a[k],b[k]]) for k in a}
    out=project_ribbons(merged,np.eye(4),config(),base())
    assert out['drivable_valid'][50,60] and not out['lane_direction_valid'][50,60]
    b=ribbons(z=5);merged={k:np.concatenate([a[k],b[k]]) for k in a}
    out=project_ribbons(merged,np.eye(4),config(),base())
    assert out['vertical_ambiguity'][50,60] and out['drivable_static'][50,60]==255
    out['drivable_rule_valid'][0,0]=True
    with pytest.raises(ValueError):validate_road_arrays(out)


@pytest.mark.parametrize('state,expected',[('Red',(1,True)),('Green',(0,True)),('Yellow',(-1,False)),('Off',(-1,False)),('Unknown',(-1,False))])
def test_signal_indication_does_not_assert_full_permission(state,expected):
    assert signal_stop_target(state,applicable=True,stable=True)==expected
    assert signal_stop_target(state,applicable=False,stable=True)==(-1,False)
    assert signal_stop_target(state,applicable=True,stable=False)==(-1,False)


def test_traffic_mixed_frames_and_unknown_rejected():
    x=dict(schema_version=SCHEMA,frame_id=8,timestamp_s=.4,full_traffic_permission_valid=False,ego_stop_valid=True,ego_stop_required=1,ego_signal_state=0)
    validate_traffic(x,8,.4)
    with pytest.raises(ValueError):validate_traffic(x,10,.4)
    with pytest.raises(ValueError):validate_traffic(x,8,.5)
    with pytest.raises(ValueError):validate_traffic({**x,'full_traffic_permission_valid':True},8,.4)
    with pytest.raises(ValueError):validate_traffic({**x,'ego_stop_valid':False},8,.4)
    with pytest.raises(ValueError,match='truth table'):validate_traffic({**x,'ego_stop_required':0},8,.4)
    with pytest.raises(ValueError,match='truth table'):validate_traffic({**x,'ego_signal_state':2},8,.4)


def test_road_overlay_not_read_during_inference(m3_index):
    torch=pytest.importorskip('torch')
    from datasets.training import HSDriveDataset
    # Deliberately nonexistent overlay: inference must not even open its manifest.
    ds=HSDriveDataset(m3_index,'train',supervision=False,road_overlay='/absent/privileged',output_wh=(8,4),feature_stride=2)
    assert ds[0].targets is None


def test_far_metric_selection_checks_missing_bands():
    pytest.importorskip('torch');pytest.importorskip('torchvision')
    from training.runner import selection_score
    bands=('1.5-10m','10-30m','30-60m','60-80m')
    report={'depth':{b:{'abs_rel':x} for b,x in zip(bands,[.1,.2,.3,.4])}}
    assert selection_score(report,{'selection_metric':'depth_macro_abs_rel'})==.25
    report['depth'][bands[-1]]['abs_rel']=None
    with pytest.raises(ValueError):selection_score(report,{'selection_metric':'depth_macro_abs_rel'})


def test_traffic_survives_index_target_but_never_input(m3_registry,tmp_path):
    import yaml,json
    from simulator.recorder import atomic_json,file_hash
    from datasets.index import build_index
    raw=tmp_path/'raw'
    for directory in raw.iterdir():
        for path in (directory/'frames').glob('*.json'):
            f=json.loads(path.read_text());rel='labels/traffic_rules/'+path.name
            traffic=dict(schema_version=SCHEMA,frame_id=f['frame_id'],timestamp_s=f['timestamp_s'],full_traffic_permission_valid=False,
                ego_stop_valid=True,ego_stop_required=1,ego_signal_state=0)
            atomic_json(directory/rel,traffic);f['labels']['traffic_rules']=rel;f['sha256'][rel]=file_hash(directory/rel);atomic_json(path,f)
    out=tmp_path/'rules_index';index=build_index(m3_registry,out)
    row=index['samples'][0];inp=json.loads((out/row['input']['path']).read_text());target=json.loads((out/row['target']['path']).read_text())
    assert 'traffic_rules' in target and 'traffic_rules' not in inp
    try:import torch
    except ImportError:return
    from datasets.training import HSDriveDataset
    ds=HSDriveDataset(out,'train',output_wh=(8,4),feature_stride=2);item=ds[0]
    assert item.targets['signal_stop_valid'].item() and item.targets['signal_stop_required'].item()==1
    assert 'signal_state' not in item.inputs.as_model_kwargs()
    from unittest.mock import patch
    with patch('datasets.training.checked_json',side_effect=lambda root,ref:json.loads((root/ref['path']).read_text()) if ref['path'].startswith('inputs/') else (_ for _ in ()).throw(AssertionError('target read'))):
        assert HSDriveDataset(out,'train',supervision=False,output_wh=(8,4),feature_stride=2)[0].targets is None


def test_context_head_receptive_field_and_old_checkpoint_compatibility():
    torch=pytest.importorskip('torch');pytest.importorskip('torchvision')
    import yaml
    from models.depth_bev import DepthBEV
    torch.set_num_threads(2)
    model=DepthBEV(yaml.safe_load(open('configs/bev.yaml')),drivable=True,road_context=True)
    x=torch.rand(1,128,160,120,requires_grad=True);y=model.drivable_head(x)
    assert y.shape==(1,1,160,120)
    y[0,0,80,60].backward()
    assert x.grad[0,:,80,60].abs().sum()>0 and x.grad[0,:,75,60].abs().sum()>0


def test_signal_probe_restores_global_freeze_and_all_states(monkeypatch):
    import sys,types
    from simulator.traffic_rules import TrafficRules
    events=[]
    class Light:
        def __init__(self,i,state):self.id=i;self.state=state
        def get_stop_waypoints(self):return []
        def get_state(self):return self.state
        def is_frozen(self):return False
        def freeze(self,value):events.append(('global_freeze',value))
        def set_state(self,value):self.state=value
        def get_group_traffic_lights(self):return [self]
    a,b=Light(1,'Green'),Light(2,'Yellow')
    class Actors:
        def filter(self,p):return [a,b]
    world=types.SimpleNamespace(get_map=lambda:None,get_actors=lambda:Actors())
    monkeypatch.setitem(sys.modules,'carla',types.SimpleNamespace(TrafficLightState=types.SimpleNamespace(Red='Red',Yellow='Yellow',Green='Green')))
    rules=TrafficRules(world,probe=a);assert a.state=='Red' and b.state=='Yellow'
    rules.set_probe_phase(40);assert a.state=='Green'
    rules.close();assert a.state=='Green' and b.state=='Yellow'
    assert events==[('global_freeze',True),('global_freeze',False)]


def test_signal_rule_payload_is_json_serializable_after_numpy_geometry():
    import json
    applicable=bool(np.float64(1)>.8)
    value,valid=signal_stop_target('Red',applicable=applicable,stable=True)
    json.dumps({'applies_to_current_lane':applicable,'stop_required':value,'stop_valid':valid},allow_nan=False)


def test_ribbon_center_mask_does_not_expand_to_touched_pixel():
    r=ribbons();r['polygons'][:,:,1]=np.array([[-.1,.1,.1,-.1]])
    out=project_ribbons(r,np.eye(4),config(),base())
    # Y centers are +/-0.25: a 0.2m stripe contains neither center.
    assert not out['drivable_valid'].any()


def test_overlay_checksum_and_raw_identity_fail_closed(m3_index,tmp_path):
    torch=pytest.importorskip('torch')
    import json
    from simulator.recorder import atomic_json,file_hash
    from datasets.training import HSDriveDataset
    from datasets.road_supervision import SCHEMA as ROAD_SCHEMA
    index=json.loads((m3_index/'index.json').read_text());overlay=tmp_path/'road';overlay.mkdir();samples={}
    for row in index['samples']:
        data=project_ribbons(ribbons(),np.eye(4),config(),base());rel=row['sample_id']+'.npz';path=overlay/rel;path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**data);samples[row['sample_id']]={'path':rel,'sha256':file_hash(path),'raw_frame_sha256':row['raw_frame_sha256']}
    m=dict(schema_version=ROAD_SCHEMA,status='complete',index_sha256=file_hash(m3_index/'index.json'),bev_config_sha256=index['drivable_bev_config_sha256'],samples=samples)
    atomic_json(overlay/'manifest.json',m)
    ds=HSDriveDataset(m3_index,'train',road_overlay=overlay,output_wh=(8,4),feature_stride=2);item=ds[0]
    assert item.targets['junction_support'].any() and item.targets['lane_direction_valid'].any()
    ref=samples[ds.windows[0][0]['sample_id']];(overlay/ref['path']).write_bytes(b'bad')
    with pytest.raises(ValueError,match='Road overlay sample'):ds[0]


def test_nonfinite_traffic_time_cannot_pass_frame_guard():
    record=dict(schema_version=SCHEMA,frame_id=1,timestamp_s=float('nan'),full_traffic_permission_valid=False,ego_stop_valid=False,ego_stop_required=-1,ego_signal_state=-1)
    with pytest.raises(ValueError):validate_traffic(record,1,.1)
