"""Opt-in zero-initialized residual; only frozen RGB quantities are inputs."""
import torch
from torch import nn

class DepthResidual(nn.Module):
    def __init__(self, *, mobile_condition=False):
        super().__init__();self.mobile_condition=mobile_condition
        self.hidden=nn.Sequential(nn.Conv2d(64+48+1,32,1),nn.ReLU())
        self.delta=nn.Conv2d(32,48,1);nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def forward(self, features, original_logits, foreground):
        if features.ndim!=4 or features.shape[1]!=64 or original_logits.shape[1]!=48 or foreground.shape!=features[:,:1].shape:raise ValueError('RGB feature/logit/foreground shapes')
        q=foreground if self.mobile_condition else torch.zeros_like(foreground)
        x=torch.cat((features.float(),original_logits.float()/10,q.float()),1)
        return original_logits.float()+self.delta(self.hidden(x)).float()
