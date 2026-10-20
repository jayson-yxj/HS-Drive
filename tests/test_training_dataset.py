import json
from pathlib import Path
import pytest
import numpy as np
torch=pytest.importorskip('torch')
from datasets.training import HSDriveDataset,make_dataloader
from datasets.inputs import LearnedInputs
from simulator.recorder import file_hash,atomic_json


def dataset(path,**kwargs):
    return HSDriveDataset(path,'train',output_wh=(8,4),feature_stride=2,**kwargs)


def test_shapes_rgb_order_geometry_and_temporal_boundaries(m3_index):
    ds=dataset(m3_index,load_objects=True);item=ds[0]
    assert item.inputs['images'].shape==(6,3,4,8)
    np.testing.assert_allclose(item.inputs['images'][0,:,0,0],[140/255,60/255,20/255],atol=1e-7)
    assert item.targets['depth_m'].shape==(6,2,4)
    assert item.targets['objects']==[{'actor_id':1,'is_ego':True}]
    assert item.inputs['route_command'].item()==-1
    assert torch.equal(item.inputs['ego_pose'],torch.eye(4,dtype=torch.float64))
    sequence=dataset(m3_index,sequence_length=3)
    assert len(sequence)==4  # two scenes, each has 4 frames -> 2 causal windows
    for w in sequence.windows:
        assert len({r['scene_id'] for r in w})==1
        assert np.allclose(np.diff([r['timestamp_s'] for r in w]),.1)
    batch=next(iter(make_dataloader(sequence,batch_size=2)))
    assert batch.inputs['images'].shape==(2,3,6,3,4,8)
    assert batch.targets['depth_bin'].shape==(2,3,6,2,4)


def test_inference_succeeds_with_all_supervision_files_absent(m3_index,monkeypatch):
    import shutil
    for name in ('targets',):shutil.rmtree(m3_index/name)
    for scene in (m3_index.parent/'raw').iterdir():
        shutil.rmtree(scene/'depth');shutil.rmtree(scene/'labels')
        shutil.rmtree(scene/'frames')  # inference needs no mixed raw metadata
    def forbidden(*args,**kwargs):raise AssertionError('Inference touched NPZ supervision')
    monkeypatch.setattr(np,'load',forbidden)
    item=dataset(m3_index,supervision=False)[0]
    assert item.targets is None and 'depth' not in list(item.inputs)
    with pytest.raises(FileNotFoundError):dataset(m3_index)[0]


def test_forbidden_reads_and_nested_payload_are_rejected_and_logged(m3_index,caplog):
    inputs=dataset(m3_index,supervision=False)[0].inputs
    with pytest.raises(KeyError,match='Forbidden'):inputs['depth']
    assert 'Forbidden learned input access: depth' in caplog.text
    with pytest.raises(ValueError):LearnedInputs({**inputs,'actor_gt':torch.zeros(1)})
    with pytest.raises(TypeError):LearnedInputs({**inputs,'ego_state':{'depth':torch.zeros(1)}})
    with pytest.raises(TypeError):LearnedInputs({**inputs,'images':[torch.zeros(1)]})


def test_altering_targets_cannot_change_inputs(m3_index):
    ds=dataset(m3_index);before=ds[0]
    row=ds.windows[0][0]
    record=json.loads((m3_index/row['target']['path']).read_text())
    depth_path=ds.raw_root/record['depth']['front']['path']
    with np.load(depth_path) as data:depth=data['depth_m'].copy()
    np.savez_compressed(depth_path,depth_m=depth*2)
    after=dataset(m3_index,supervision=False)[0]
    for name in before.inputs:assert torch.equal(before.inputs[name],after.inputs[name])
    with pytest.raises(ValueError,match='Depth checksum'):ds[0]


def test_corrupt_rgb_and_calibration_fail_closed(m3_index):
    ds=dataset(m3_index,supervision=False);row=ds.windows[0][0]
    record=json.loads((m3_index/row['input']['path']).read_text())
    (ds.raw_root/record['cameras']['front']['rgb']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='RGB checksum'):ds[0]
    (m3_index/row['input']['path']).write_text('{}')
    with pytest.raises(ValueError,match='Checksum'):ds[0]


def test_seeded_workers_produce_same_order_and_values(m3_index):
    ds=dataset(m3_index,supervision=False)
    def collect(workers):
        return [(batch.metadata[0]['sample_id'],batch.inputs['images'].sum().item())
                for batch in make_dataloader(ds,workers=workers,shuffle=True,seed=17)]
    assert collect(0)==collect(2)==collect(0)


def test_sequence_does_not_bridge_missing_tick(m3_index):
    index_path=m3_index/'index.json';index=json.loads(index_path.read_text())
    row=[r for r in index['samples'] if r['scene_id']=='train_a'][2];row['timestamp_s']+=.05
    atomic_json(index_path,index)
    manifest=json.loads((m3_index/'manifest.json').read_text());manifest['index_sha256']=file_hash(index_path)
    atomic_json(m3_index/'manifest.json',manifest)
    split=json.loads((m3_index/'splits/train.json').read_text());split['index_sha256']=file_hash(index_path)
    atomic_json(m3_index/'splits/train.json',split)
    ds=dataset(m3_index,sequence_length=3)
    assert all(w[0]['scene_id']!='train_a' for w in ds.windows)


def test_loader_config_is_consumed_and_unknown_fields_rejected(m3_index,tmp_path):
    import yaml
    from datasets.training import loader_from_config
    config=yaml.safe_load(Path('configs/train_data.yaml').read_text())
    config.update(output_wh=[8,4],feature_stride=2,workers=0)
    path=tmp_path/'loader.yaml';path.write_text(yaml.safe_dump(config))
    loader=loader_from_config(m3_index,'val',path,supervision=False)
    assert next(iter(loader)).targets is None
    assert isinstance(loader.sampler,torch.utils.data.SequentialSampler)
    config['depth_as_input']=True;path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):loader_from_config(m3_index,'train',path)


def test_exported_kwargs_can_only_contain_approved_tensors(m3_index):
    item=dataset(m3_index,load_objects=True)[0]
    kwargs=item.inputs.as_model_kwargs()
    assert all(isinstance(v,torch.Tensor) for v in kwargs.values())
    for forbidden in ('targets','objects','map','scene_id','depth_bin','depth_m','future_trajectory'):
        assert forbidden not in kwargs
    with pytest.raises(ValueError):LearnedInputs({**kwargs,'targets':item.targets})
