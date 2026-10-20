"""Torchvision ResNet-18 and lightweight stride-16 pyramid (V1.3)."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class ImageEncoder(nn.Module):
    def __init__(self, channels=64, weights_path=None):
        super().__init__()
        self.backbone=resnet18(weights=None)
        if weights_path:
            path=Path(weights_path)
            if not hashlib.sha256(path.read_bytes()).hexdigest().startswith('f37072fd'):
                raise ValueError('Expected official ResNet18 IMAGENET1K_V1 weight hash prefix')
            self.backbone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        self.backbone.fc=nn.Identity()
        self.lateral=nn.ModuleList([nn.Conv2d(c,channels,1) for c in (128,256,512)])
        self.output=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(8,channels),nn.ReLU())
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.train(True)

    def train(self,mode=True):
        super().train(mode)
        # Batch size 1 scene, correlated six views: use fixed pretrained BN stats.
        # Affine parameters still train; from-scratch runs retain default stats.
        for module in self.backbone.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        return self

    def forward(self,images):
        b,n,c,h,w=images.shape
        x=(images.reshape(b*n,c,h,w)-self.mean)/self.std
        r=self.backbone
        x=r.maxpool(r.relu(r.bn1(r.conv1(x))))
        x=r.layer1(x);p2=r.layer2(x);p3=r.layer3(p2);p4=r.layer4(p3)
        f=self.lateral[1](p3)
        f=f+F.interpolate(self.lateral[2](p4),size=f.shape[-2:],mode='bilinear',align_corners=False)
        f=f+F.adaptive_avg_pool2d(self.lateral[0](p2),f.shape[-2:])
        f=self.output(f)
        return f.reshape(b,n,*f.shape[1:])
