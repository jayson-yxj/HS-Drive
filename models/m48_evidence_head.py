"""Small independent sigmoid evidence heads; old Encoder/Splat unchanged."""
import torch
from torch import nn

class EvidenceHead(nn.Module):
    def __init__(self,channels=128,width=64):
        super().__init__()
        self.shared=nn.Sequential(nn.Conv2d(channels,width,3,padding=1),nn.GELU(),
                                  nn.Conv2d(width,width,3,padding=1),nn.GELU())
        self.surface=nn.Conv2d(width,4,1)
        self.free=nn.Conv2d(width,4,1)
        self.semantic=nn.Conv2d(width,12,1)
    def forward(self,bev):
        x=self.shared(bev)
        return dict(surface=self.surface(x),free=self.free(x),semantic=self.semantic(x).reshape(-1,3,4,*x.shape[-2:]))

class PositionPrior(nn.Module):
    """Per-cell logits, stronger than a small coordinate MLP; same training masks."""
    def __init__(self):
        super().__init__();self.logits=nn.Parameter(torch.zeros(1,20,160,120))
    def forward(self,bev):
        p=self.logits.expand(len(bev),-1,-1,-1)
        return dict(surface=p[:,:4],free=p[:,4:8],semantic=p[:,8:].reshape(-1,3,4,160,120))
