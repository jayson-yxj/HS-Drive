"""New checkpoint contract checks, CPU/synthetic, no private assets."""
import copy
import random

import numpy as np
import pytest
import torch

from training.m8j6_retry_state import (parameter_mapping, snapshot, fingerprint, assert_exact,
    assert_frozen_unchanged, rng_state, restore_rng)


def fixture():
    model = torch.nn.Module()
    shared = torch.nn.Linear(3,2)
    model.head = torch.nn.ModuleList([shared,shared])
    model.backbone = torch.nn.BatchNorm1d(3).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-5, weight_decay=0.)
    return model,optimizer,lambda n:n.startswith('head.')


def test_alias_map_and_no_duplicate_optimizer():
    m,o,p=fixture();a=parameter_mapping(m,o,p)
    r=next(r for r in a['parameters'] if r['canonical_name']=='head.0.weight')
    assert r['aliases']==['head.0.weight','head.1.weight'] and r['trainable']
    assert a['optimizer_unique_count']==a['canonical_trainable_count']==2
    assert not set(a['trainable_state_names']) & set(a['frozen_state_names'])


@pytest.mark.parametrize('target',['backbone.weight','backbone.running_mean'])
def test_true_frozen_parameter_or_buffer_corruption_rejected(target):
    m,o,p=fixture();a=parameter_mapping(m,o,p);s=snapshot(m.state_dict())
    with torch.no_grad():m.state_dict()[target].add_(1)
    with pytest.raises(AssertionError,match='FrozenChanged'):
        assert_frozen_unchanged(m,s,a['frozen_state_names'])


def test_shared_trainable_change_allowed_without_copying_weights():
    m,o,p=fixture();a=parameter_mapping(m,o,p);s=snapshot(m.state_dict())
    with torch.no_grad():m.head[0].weight.add_(1)
    assert_frozen_unchanged(m,s,a['frozen_state_names'])
    assert m.head[0].weight is m.head[1].weight
    assert not torch.equal(m.head[0].weight,s['head.0.weight'])


def test_conflicting_alias_policy_rejected():
    m,o,_=fixture()
    with pytest.raises(ValueError,match='ConflictingAliasPolicy'):
        parameter_mapping(m,o,lambda n:n.startswith('head.0.'))


def test_duplicate_optimizer_parameter_rejected():
    m,o,p=fixture();o.param_groups[0]['params'].append(o.param_groups[0]['params'][0])
    with pytest.raises(ValueError,match='DuplicateOptimizerParameter'):parameter_mapping(m,o,p)


def test_missing_optimizer_parameter_rejected():
    m,o,p=fixture();o.param_groups[0]['params'].pop()
    with pytest.raises(ValueError,match='OptimizerEligibilityMismatch'):parameter_mapping(m,o,p)


def test_frozen_parameter_in_optimizer_rejected():
    m,o,p=fixture();o.param_groups[0]['params'].append(m.backbone.weight)
    with pytest.raises(ValueError,match='OptimizerEligibilityMismatch'):parameter_mapping(m,o,p)


def test_snapshot_is_independent_for_model_and_optimizer():
    m,o,p=fixture();m.head[0](torch.ones(1,3)).sum().backward();o.step()
    state=snapshot(dict(model=m.state_dict(),optimizer=o.state_dict()));h=fingerprint(state)
    with torch.no_grad():m.head[0].weight.add_(2)
    next(iter(o.state.values()))['exp_avg'].add_(3)
    assert fingerprint(state)==h
    assert fingerprint(dict(model=m.state_dict(),optimizer=o.state_dict()))!=h


def test_new_instance_restore_names_moments_aliases(tmp_path):
    m,o,p=fixture();m.head[0](torch.ones(1,3)).sum().backward();o.step()
    a=parameter_mapping(m,o,p);s=snapshot(dict(model=m.state_dict(),optimizer=o.state_dict()))
    path=tmp_path/'own_synthetic.pth';torch.save(s,path);saved=torch.load(path)
    n,q,_=fixture();n.load_state_dict(saved['model']);q.load_state_dict(saved['optimizer'])
    assert m.head[0].weight is not n.head[0].weight
    assert n.head[0].weight is n.head[1].weight
    assert parameter_mapping(n,q,p)==a
    assert_exact(s,dict(model=n.state_dict(),optimizer=q.state_dict()))


def test_optimizer_moment_corruption_detected():
    m,o,_=fixture();m.head[0](torch.ones(1,3)).sum().backward();o.step()
    s=snapshot(o.state_dict());bad=snapshot(s);next(iter(bad['state'].values()))['exp_avg'].add_(.5)
    with pytest.raises(AssertionError,match='TensorRestoreMismatch'):assert_exact(s,bad)


def test_cpu_rng_roundtrip_without_cuda():
    random.seed(875606);np.random.seed(875606);torch.manual_seed(875606)
    s=rng_state();a=(random.random(),np.random.rand(),torch.rand(4))
    restore_rng(s);b=(random.random(),np.random.rand(),torch.rand(4))
    assert_exact(a,b)


def test_copying_shared_weights_changes_cross_process_contract():
    m,o,p=fixture();a=parameter_mapping(m,o,p)
    n,_,_=fixture();n.head[1]=copy.deepcopy(n.head[0])
    q=torch.optim.AdamW(n.head.parameters(),lr=1e-5,weight_decay=0.)
    b=parameter_mapping(n,q,p)
    with pytest.raises(AssertionError):assert_exact(a,b,'SharedRelationship')


def test_official_model_policy_matches_frozen_parameter_set():
    import os
    from pathlib import Path
    if not os.environ.get('M8J6_RUNTIME'):pytest.skip('Frozen official PETR environment required')
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_model
    from training.m8j6_retry_state import head_policy
    rt=Path(os.environ['M8J6_RUNTIME'])
    cfg=mmcv.Config.fromfile(str(rt/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    cfg.model.img_backbone.pretrained=None
    m=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg')).eval()
    for n,p in m.named_parameters():p.requires_grad_(p.requires_grad and n.startswith('pts_bbox_head.'))
    o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.)
    mapping=parameter_mapping(m,o,head_policy)
    assert mapping['canonical_trainable_count']==141
    assert len(mapping['trainable_state_names'])==221 and len(mapping['frozen_state_names'])==337
