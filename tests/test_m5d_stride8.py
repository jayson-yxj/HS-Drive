import pytest
import torch
from models.m5d_stride8 import p8_features
from models.image_encoder import ImageEncoder


def test_p8_uses_real_unpooled_signal_and_preserves_state():
    torch.set_num_threads(2);torch.manual_seed(81);net=ImageEncoder().eval();images=torch.randn(1,6,3,64,96);before={k:v.clone() for k,v in net.state_dict().items()}
    with torch.no_grad():p8=p8_features(net,images);p16=net(images)
    assert p8.shape==(1,6,64,8,12) and p16.shape==(1,6,64,4,6)
    interp=torch.nn.functional.interpolate(p16.flatten(0,1),scale_factor=2,mode='bilinear',align_corners=False).reshape_as(p8)
    assert (p8-interp).abs().mean()>.01
    assert all(torch.equal(v,before[k]) for k,v in net.state_dict().items())
