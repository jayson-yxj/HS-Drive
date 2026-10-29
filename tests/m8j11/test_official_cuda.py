"""Optional official candidate runtime tests; no data/model checkpoint required."""
import numpy as np
import pytest
import torch
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from datasets.m8j11_contract import to_legacy_boxes

def test_official_rc6_corner_definition():
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    x=np.array([[1,2,-.5,4,2,1.5,.7,3,-2]],dtype=np.float32)
    native=LiDARInstance3DBoxes(torch.from_numpy(x),box_dim=9)
    old=to_legacy_boxes(x)
    b=Box([1,2,.25],old[0,3:6],Quaternion(axis=[0,0,1],angle=-float(old[0,6])-np.pi/2))
    d=np.linalg.norm(native.corners[0].numpy()[:,None]-b.corners().T[None],axis=-1)
    assert d.min(1).max()<1e-5

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA required')
def test_official_fp32_attention_and_qkv_layout():
    from projects.mmdet3d_plugin.models.utils.petr_transformer import PETRMultiheadAttention
    torch.manual_seed(11);a=PETRMultiheadAttention(256,8,fp16=False).cuda().eval()
    q=torch.randn(5,1,256,device='cuda');k=torch.randn(7,1,256,device='cuda');v=torch.randn_like(k)
    with torch.no_grad():
        got=a(q,k,v)
        w=a.attn.in_proj_weight;b=a.attn.in_proj_bias
        qp,kp,vp=[torch.nn.functional.linear(x,ww,bb).permute(1,0,2).reshape(1,-1,8,32).transpose(1,2) for x,ww,bb in zip((q,k,v),w.chunk(3),b.chunk(3))]
        raw=((qp@kp.transpose(-2,-1))/(32**.5)).softmax(-1)@vp
        ref=torch.nn.functional.linear(raw.transpose(1,2).reshape(1,5,256).transpose(0,1),a.attn.out_proj.weight,a.attn.out_proj.bias)+q
    torch.testing.assert_close(got,ref,atol=1e-5,rtol=1e-5)
    assert got.dtype==torch.float32
    assert set(a.attn.state_dict())=={'in_proj_weight','in_proj_bias','out_proj.weight','out_proj.bias'}
