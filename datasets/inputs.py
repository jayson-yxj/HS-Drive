"""Audited, tensor-only boundary passed to learned models (no label objects)."""
import logging
from collections.abc import Mapping
import torch

INPUT_FIELDS = frozenset(('images','intrinsics','cam_to_ego','ego_pose','ego_state',
                          'route_command','timestamps_s','valid_cameras'))
_LOG = logging.getLogger('hsdrive.input_audit')


def validate_tensor_inputs(inputs):
    if set(inputs) != INPUT_FIELDS:
        _LOG.error('Rejected model input keys: %s', sorted(set(inputs)^INPUT_FIELDS))
        raise ValueError('Learned model input field set mismatch')
    for key,value in inputs.items():
        if not isinstance(value,torch.Tensor):
            _LOG.error('Rejected non-tensor/nested payload in model field %s', key)
            raise TypeError(f'Only tensors allowed in model input field: {key}')
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite model input: {key}')
    images=inputs['images']
    if images.ndim not in (4,5,6) or images.shape[-4:-2] != (6,3) or images.dtype != torch.float32:
        raise ValueError('Expected float32 [...,6,3,H,W] images')
    leading=images.shape[:-4]
    shapes={'intrinsics':(6,3,3),'cam_to_ego':(6,4,4),'ego_pose':(4,4),
            'ego_state':(7,),'route_command':(),'timestamps_s':(),'valid_cameras':(6,)}
    for key,tail in shapes.items():
        if inputs[key].shape != leading+tail:
            raise ValueError(f'Input shape mismatch: {key}')
    if inputs['route_command'].dtype != torch.int64 or inputs['valid_cameras'].dtype != torch.bool:
        raise TypeError('Route must be int64 and camera validity bool')
    for key in ('intrinsics','cam_to_ego','ego_pose','ego_state','timestamps_s'):
        if not inputs[key].dtype.is_floating_point:
            raise TypeError(f'Expected floating point tensor: {key}')


class LearnedInputs(Mapping):
    """Only approved tensors are stored; forbidden access raises and is logged."""
    def __init__(self,values):
        validate_tensor_inputs(values)
        self._values=dict(values)

    def __getitem__(self,key):
        if key not in INPUT_FIELDS:
            _LOG.error('Forbidden learned input access: %s', key)
            raise KeyError(f'Forbidden learned input access: {key}')
        return self._values[key]

    def __iter__(self):return iter(self._values)
    def __len__(self):return len(self._values)

    def to(self,device,*,non_blocking=False):
        return LearnedInputs({k:v.to(device,non_blocking=non_blocking) for k,v in self.items()})

    def pin_memory(self):
        return LearnedInputs({k:v.pin_memory() for k,v in self.items()})

    def as_model_kwargs(self):
        validate_tensor_inputs(self)
        return dict(self)
