import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from datasets.index import build_index,safe_path
from datasets.split import assign_splits,verify_spatial_separation


def test_index_is_reproducible_and_never_splits_frames(m3_registry):
    first=build_index(m3_registry,m3_registry.parent/'one')
    second=build_index(m3_registry,m3_registry.parent/'two')
    assert first==second
    for scene in first['scenes']:
        assert {r['split'] for r in first['samples'] if r['scene_id']==scene['scene_id']}=={scene['split']}
    assert first['spatial_separation']
    with pytest.raises(FileExistsError):build_index(m3_registry,m3_registry.parent/'one')


def test_groups_and_unseen_maps_cannot_leak():
    scenes=[{'scene_id':'a','group_id':'same','map':'Town03','split':'train'},
            {'scene_id':'b','group_id':'same','map':'Town03','split':'val'},
            {'scene_id':'c','group_id':'test','map':'Town10','split':'test'}]
    with pytest.raises(ValueError,match='Group crosses'):
        assign_splits(scenes,seed=7,val_fraction=.2,test_maps=['Town10'])
    scenes[1]['group_id']='other';scenes[2]['split']='train'
    with pytest.raises(ValueError,match='Held-out'):
        assign_splits(scenes,seed=7,val_fraction=.2,test_maps=['Town10'])


def test_nearby_routes_not_just_recording_names_are_checked():
    scenes=[{'scene_id':'a','map':'Town03','split':'train','positions_xy':[[0,0],[100,0]]},
            {'scene_id':'b','map':'Town03','split':'val','positions_xy':[[95,0]]}]
    with pytest.raises(ValueError,match='Nearby routes'):
        verify_spatial_separation(scenes,50)


@pytest.mark.parametrize('rel',['../escape','/absolute'])
def test_references_cannot_escape_root(tmp_path,rel):
    with pytest.raises(ValueError,match='escapes'):safe_path(tmp_path,rel)


@pytest.mark.parametrize('defect',['checksum','incomplete','missing_seed','duplicate_source','duplicate_content'])
def test_index_refuses_invalid_sources_without_publishing(m3_registry,defect):
    root=m3_registry.parent;registry=yaml.safe_load(m3_registry.read_text())
    manifest=root/'raw/train_a/manifest.json';m=json.loads(manifest.read_text())
    if defect=='checksum':
        next((root/'raw/train_a/rgb/front').glob('*.png')).write_bytes(b'broken')
    elif defect=='incomplete':m['status']='interrupted'
    elif defect=='missing_seed':m.pop('seed')
    elif defect=='duplicate_source':registry['scenes'][1]['path']='train_a'
    else:
        import shutil
        shutil.rmtree(root/'raw/val');shutil.copytree(root/'raw/train_a',root/'raw/val')
    manifest.write_text(json.dumps(m));m3_registry.write_text(yaml.safe_dump(registry))
    with pytest.raises(ValueError):build_index(m3_registry,root/'invalid')
    assert not (root/'invalid').exists() and not (root/'invalid.building').exists()


def test_missing_checksum_reference_is_not_silently_accepted(m3_registry):
    path=next((m3_registry.parent/'raw/train_a/frames').glob('*.json'))
    sample=json.loads(path.read_text());sample['sha256'].pop(sample['cameras']['front']['depth'])
    path.write_text(json.dumps(sample))
    with pytest.raises(ValueError,match='checksum references'):
        build_index(m3_registry,m3_registry.parent/'invalid')


def test_group_hash_split_is_stable_under_reordering():
    scenes=[{'scene_id':f's{i}','group_id':f'g{i}','map':'Town03'} for i in range(30)]
    scenes.append({'scene_id':'test','group_id':'test','map':'Town10'})
    args={'seed':7,'val_fraction':.3,'test_maps':['Town10']}
    assert assign_splits(scenes,**args)==assign_splits(scenes[::-1],**args)
