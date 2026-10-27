"""Object-based optimizer policy and stable cross-process checkpoint evidence."""
import hashlib
import json
import random

import numpy as np
import torch

from training.m8j6_frozen_state import partition_state, assert_frozen_unchanged


def parameter_mapping(model, optimizer, policy):
    """Validate each alias's declared policy, actual flag and optimizer slot.

    IDs are only local dictionary keys; public/cross-process rows use names.
    Frozen buffers remain in partition_state's frozen set.
    """
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        entry = groups.setdefault(id(parameter), dict(parameter=parameter, aliases=[]))
        entry['aliases'].append(name)
    membership = {}
    for gi, group in enumerate(optimizer.param_groups):
        for slot, parameter in enumerate(group['params']):
            if id(parameter) in membership:
                raise ValueError('DuplicateOptimizerParameter')
            if id(parameter) not in groups:
                raise ValueError('ForeignOptimizerParameter')
            membership[id(parameter)] = dict(group=gi, slot=slot)
    rows = []
    for identity, entry in groups.items():
        p = entry['parameter']; aliases = entry['aliases']
        decisions = [policy(name) for name in aliases]
        if any(type(v) is not bool for v in decisions):
            raise ValueError('PolicyMustReturnBool')
        if len(set(decisions)) != 1:
            raise ValueError('ConflictingAliasPolicy:' + ','.join(aliases))
        trainable = decisions[0]
        if trainable != p.requires_grad:
            raise ValueError('ActualRequiresGradMismatch:' + aliases[0])
        if (identity in membership) != trainable:
            raise ValueError('OptimizerEligibilityMismatch:' + aliases[0])
        rows.append(dict(canonical_name=aliases[0], aliases=aliases, trainable=trainable,
                         shape=list(p.shape), dtype=str(p.dtype), numel=p.numel(),
                         optimizer=membership.get(identity)))
    train_aliases, frozen = partition_state(model)
    assert not train_aliases & frozen
    assert train_aliases | frozen == set(model.state_dict())
    return dict(parameters=rows, trainable_state_names=sorted(train_aliases), frozen_state_names=sorted(frozen),
                buffer_names=sorted(n for n, _ in model.named_buffers(remove_duplicate=False)),
                canonical_trainable_count=sum(r['trainable'] for r in rows),
                optimizer_unique_count=len(membership), policy_conflicts=0, cross_process_identity='NamesAndAliases_NotObjectID')


def head_policy(name):
    # The original config deliberately makes code_weights nontrainable.
    return name.startswith('pts_bbox_head.') and name != 'pts_bbox_head.code_weights'


def snapshot(value):
    """Independent CPU copies, including optimizer moments and buffers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    return value


def fingerprint(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous()
            h.update(b'tensor'); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(b'array'); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=lambda k:(type(k).__name__,str(k))):
                visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode()); h.update(str(len(v)).encode())
            for x in v: visit(x)
        else:
            h.update(type(v).__name__.encode()); h.update(json.dumps(v, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def assert_exact(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError('TensorRestoreMismatch:' + path)
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a,b):
            raise AssertionError('ArrayRestoreMismatch:' + path)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b): raise AssertionError('SchemaRestoreMismatch:' + path)
        for k in a: assert_exact(a[k], b[k], path + '.' + str(k))
    elif isinstance(a, (tuple, list)):
        if type(a) != type(b) or len(a) != len(b): raise AssertionError('SequenceRestoreMismatch:' + path)
        for i,(x,y) in enumerate(zip(a,b)): assert_exact(x,y,path + '.' + str(i))
    elif a != b:
        raise AssertionError('ValueRestoreMismatch:' + path)


def rng_state():
    return snapshot(dict(python=random.getstate(), numpy=np.random.get_state(),
                         torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
    assert_exact(state, rng_state(), 'RNG')


def optimizer_summary(optimizer):
    state = optimizer.state_dict()
    return dict(sha256=fingerprint(state), entries=len(state['state']),
                step_values=sorted({int(s['step']) for s in state['state'].values()}),
                first_moment_l2=sum(float(s['exp_avg'].double().square().sum()) for s in state['state'].values())**.5,
                lr=[g['lr'] for g in optimizer.param_groups],
                weight_decay=[g['weight_decay'] for g in optimizer.param_groups])
