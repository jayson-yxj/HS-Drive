"""Explicit dual-RGB research Candidate. Does not register with Oracle driving."""
import json
from pathlib import Path
import torch
from torch import nn
from datasets.inputs import validate_tensor_inputs
from tools.m5i_data import load
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5j_evidence import evidence
from simulator.recorder import file_hash

CAMERAS=('front','front_left','front_right','rear_left','rear_right','rear')


def validate_packet(inputs, camera_names, now_s=None, max_age_s=.25):
    validate_tensor_inputs(inputs)
    if tuple(camera_names)!=CAMERAS:raise ValueError('Camera order mismatch')
    im=inputs['images']
    if im.shape!=(1,6,3,384,640):raise ValueError('Candidate v1 requires B1 six384x640')
    if im.min()<0 or im.max()>1:raise ValueError('RGB outside [0,1]')
    if not inputs['valid_cameras'].all():raise ValueError('All six cameras required')
    ts=inputs['timestamps_s']
    if (ts<=0).any():raise ValueError('Invalid simulation timestamp')
    if now_s is not None:
        if not torch.isfinite(torch.tensor(now_s)) or now_s<float(ts[0]) or now_s-float(ts[0])>max_age_s:raise ValueError('Future or stale timestamp')
    K=inputs['intrinsics'];T=inputs['cam_to_ego'];R=T[...,:3,:3]
    if (K[...,0,0]<=0).any() or (K[...,1,1]<=0).any() or (torch.linalg.det(K).abs()<1e-8).any():raise ValueError('Invalid calibration K')
    eye=torch.eye(3,device=R.device).expand_as(R)
    if not torch.allclose(R@R.transpose(-1,-2),eye,atol=1e-4) or not torch.allclose(torch.linalg.det(R),torch.ones_like(R[...,0,0]),atol=1e-4):raise ValueError('Invalid rigid camera transform')
    if not torch.allclose(T[...,3,:],torch.tensor([0,0,0,1.],device=T.device).expand_as(T[...,3,:])):raise ValueError('Invalid homogeneous transform')


class MobileBEVCandidate(nn.Module):
    def __init__(self, *, enable=False, visual='mobile_gt', seed=101, manifest='configs/m5j_models.json'):
        super().__init__()
        if not enable:raise ValueError('Explicit Candidate opt-in required')
        cfg=json.loads(Path(manifest).read_text())
        if cfg['schema']!='hsd-m5j-dual-v1':raise ValueError('Version mismatch')
        r=next((r for r in cfg['pixel']if r['name']==visual and r['seed']==seed),None)
        if r is None:raise ValueError('Unknown visual candidate')
        if file_hash(cfg['depth']['checkpoint'])!=cfg['depth']['sha256']:raise ValueError('Depth SHA mismatch')
        self.pixel,_=load(r);self.depth=encoder_load();self.version=cfg['schema'];self.visual=visual;self.seed=seed
        self.eval()
        for p in self.parameters():p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, *, camera_names=CAMERAS, now_s=None, **inputs):
        validate_packet(inputs,camera_names,now_s)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pixel=self.pixel(**inputs).float().sigmoid()
            # Retain historical floor256 cache semantics for exact controlled comparisons.
            pixel=torch.clamp((pixel*256).floor(),max=255)/256
            original=cache_p8(self.depth,inputs)
        spatial=evidence(self.depth.splat,pixel[:,:,4::8,4::8],original['probs'],original['K'],original['T'],original['image_hw'])
        return dict(pixel_foreground=pixel,depth_probability=original['probs'],evidence=spatial,
                    timestamp_s=inputs['timestamps_s'].clone(),schema=self.version)
