"""Identify frozen state by Parameter identity, including state_dict aliases."""


def partition_state(model):
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    aliases = {n for n, p in model.named_parameters(remove_duplicate=False) if id(p) in trainable_ids}
    state_names = set(model.state_dict())
    return aliases, state_names - aliases


def assert_frozen_unchanged(model, before, frozen_names):
    current = model.state_dict()
    if set(current) != set(before):
        raise AssertionError('StateSchemaChanged')
    import torch
    changed = [n for n in sorted(frozen_names) if not torch.equal(current[n].detach().cpu(), before[n])]
    if changed:
        raise AssertionError('FrozenChanged:' + ','.join(changed))
    return len(frozen_names)
