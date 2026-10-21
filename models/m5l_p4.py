"""Real frozen Mobile shallow S4 + trained semantic P8 at explicit pixel coordinates.

No depth-trained P4 model is claimed. Optional head starts from P8 probability;
training remains gated by observed strict support. Default encoder untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F

def sample_map(feature,uv,*,stride,center):
    """uv [N,R,2] native image pixels; conv feature centers or nominal depth centers."""
    n,c,h,w=feature.shape
    xy=(uv.float()-center)/stride
    grid=torch.stack((2*(xy[...,0]+.5)/w-1,2*(xy[...,1]+.5)/h-1),-1)[:,:,None]
    return F.grid_sample(feature.float(),grid,align_corners=False,padding_mode='border')[:,:,:,0].transpose(1,2)

class P4LocalFeatures(nn.Module):
    def __init__(self,mobile):
        super().__init__();self.mobile=mobile
        for p in self.mobile.parameters():p.requires_grad_(False)
    @torch.no_grad()
    def forward(self,images,uv):
        x=(images.flatten(0,1)-self.mobile.mean)/self.mobile.std;levels=[];shallow=None
        for i,layer in enumerate(self.mobile.features):
            x=layer(x)
            if i==3:shallow=x
            if i in [6,12,16]:levels.append(x)
        semantic=self.mobile.fuse(levels)
        # Torchvision padded odd convs have receptive center 0+stride*i.
        return torch.cat((sample_map(shallow,uv,stride=4,center=0),sample_map(semantic,uv,stride=8,center=0)),dim=-1)

class P4LocalDepth(nn.Module):
    def __init__(self):
        super().__init__();self.hidden=nn.Linear(24+64+48,32);self.delta=nn.Linear(32,48)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def forward(self,rgb_features,base_probability):
        if rgb_features.shape[-1]!=88 or base_probability.shape[-1]!=48:raise ValueError('Feature contract')
        base=base_probability.float().clamp_min(1e-8).log()
        return base+self.delta(torch.relu(self.hidden(torch.cat((rgb_features.float(),base/10),-1))))
