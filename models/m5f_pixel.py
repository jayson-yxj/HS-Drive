"""Opt-in RGB-only foreground decoder; default visual/depth/BEV unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from datasets.inputs import validate_tensor_inputs


def backbone_levels(encoder,images):
    b,n,c,h,w=images.shape;r=encoder.backbone;x=(images.flatten(0,1)-encoder.mean)/encoder.std
    x=r.maxpool(r.relu(r.bn1(r.conv1(x))));x=r.layer1(x);a=r.layer2(x);b=r.layer3(a);c=r.layer4(b)
    return [a,b,c]


def pyramid(encoder,levels,stride):
    a,b,c=levels
    if stride==16:
        f=encoder.lateral[1](b)+F.interpolate(encoder.lateral[2](c),size=b.shape[-2:],mode='bilinear',align_corners=False)+F.adaptive_avg_pool2d(encoder.lateral[0](a),b.shape[-2:])
    elif stride==8:
        f=encoder.lateral[0](a)+F.interpolate(encoder.lateral[1](b),size=a.shape[-2:],mode='bilinear',align_corners=False)+F.interpolate(encoder.lateral[2](c),size=a.shape[-2:],mode='bilinear',align_corners=False)
    else:raise ValueError('only native S16/P8')
    return encoder.output(f)


class PixelHead(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),nn.Conv2d(32,1,1))
    def forward(self,features,size=(384,640)):
        return F.interpolate(self.layers(features),size=size,mode='bilinear',align_corners=False)[:,0]


class ForegroundModel(nn.Module):
    def __init__(self,encoder,head,stride=16):
        super().__init__();self.encoder=encoder;self.head=head;self.stride=stride
    def forward(self,**inputs):
        validate_tensor_inputs(inputs);images=inputs['images'];b,n=images.shape[:2]
        features=pyramid(self.encoder,backbone_levels(self.encoder,images),self.stride)
        return self.head(features,images.shape[-2:]).reshape(b,n,*images.shape[-2:])
