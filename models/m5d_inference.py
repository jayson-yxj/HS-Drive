"""Single-frame inference wrapper; privileged evidence is never an input."""

from torch import nn
from datasets.inputs import validate_tensor_inputs
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cached_forward


class VisibleExistenceModel(nn.Module):
    def __init__(self, encoder, head, stride=16):
        super().__init__()
        if stride not in (8, 16):
            raise ValueError("stride must be 8 or 16")
        self.encoder, self.head, self.stride = encoder, head, stride

    def forward(self, **inputs):
        validate_tensor_inputs(inputs)
        if self.stride == 16:
            bev = self.encoder(**inputs)["bev_current"]
        else:
            if self.training:
                raise ValueError("P8 wrapper is inference-only; use matched training runner")
            bev = cached_forward(self.encoder, cache_p8(self.encoder, inputs))
        return self.head(bev)
