import ast
import json
from pathlib import Path
import numpy as np
import pytest
from datasets.m8j4_petr_input import A,validate_arrays


def test_real_dimensions_and_crop_projection():
    images=np.zeros((6,900,1600,3),np.uint8)
    k=np.tile([[1142.5,0,800],[0,1142.5,450],[0,0,1]],(6,1,1))
    t=np.tile(np.eye(4),(6,1,1));validate_arrays(images,k,t)
    assert np.allclose(A@k[0],[[1005.4,0,704],[0,1005.4,116],[0,0,1]])
    with pytest.raises(ValueError):validate_arrays(images[:5],k,t)
    k[0,0,0]=np.nan
    with pytest.raises(ValueError):validate_arrays(images,k,t)


def test_forward_source_isolation_and_limited_budget():
    text=Path('tools/m8j4_forward.py').read_text()
    tree=ast.parse(text)
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert 'datasets.m8j4_gt' not in imports and 'tools.m8j4_sidecar' not in imports
    assert 'allow_pickle=False' in text and 'sys.addaudithook' in text
    assert 'requires_grad_(False)' in text and 'torch.no_grad()' in text
    cfg=json.loads(Path('configs/m8j4_protocol.json').read_text())
    assert cfg['model']['timed_calls']==3 and cfg['model']['training_steps']==0
    assert cfg['limits']['fake_time'] is False


def test_private_sidecar_invariants():
    root=Path('/root/gpufree-data/hs-drive-private/m8j4-single-frame')
    if not (root/'reviewed_sidecar').exists():pytest.skip('private payload not available')
    seen=set();anomalies=0;positives=0
    for p in sorted((root/'reviewed_sidecar').glob('*.json')):
        f=json.loads(p.read_text());seen.add(f['ordinal'])
        assert f['timestamp_s'] is None and f['source_frame'] is None
        assert f['background_negative_valid'] is False and f['occupancy_free_valid'] is False
        for o in f['objects']:
            assert o['label_status'] in ('Valid','Invalid','Unknown')
            if o['reason']=='HiddenBelowGround_NoTrainingTarget':
                anomalies+=1;assert not o['positive_supervision_candidate']
            if o['positive_supervision_candidate']:
                positives+=1
                assert any(v['status']=='Valid' for v in o['views'].values())
            for v in o['views'].values():
                if v['status']=='Valid':
                    assert v['match']['render_token']!=30564
                    assert v['human_review_required']
    assert len(seen)==214 and anomalies==23 and positives==711


def test_common_positive_region_does_not_make_unknown_a_negative():
    from tools.m8j4_final_audit import common_region
    obj=dict(positive_supervision_candidate=True, center_hs_m=[10,0,1],
             views={'front':dict(status='Valid', projection=dict(bbox_xyxy=[700,400,800,600]))})
    assert common_region(obj)
    obj['views']['front']['status']='Unknown'
    assert not common_region(obj)
    obj['views']['front']['status']='Valid'
    obj['views']['front']['projection']['bbox_xyxy']=[700,0,800,200]
    assert not common_region(obj)
    obj['views']['front']['projection']['bbox_xyxy']=[700,400,800,600]
    obj['center_hs_m']=[70,0,1]
    assert not common_region(obj)
