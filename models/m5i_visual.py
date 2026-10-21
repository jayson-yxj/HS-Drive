"""Opt-in MobileNetV3 multiscale RGB branch; production encoder unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large
from models.m5f_pixel import PixelHead
from datasets.inputs import validate_tensor_inputs

class MobilePixel(nn.Module):
    def __init__(self,weights_path=None):
        super().__init__();base=mobilenet_v3_large(weights=None)
        if weights_path:base.load_state_dict(torch.load(weights_path,weights_only=True))
        self.features=base.features;self.lateral=nn.ModuleList([nn.Conv2d(c,64,1)for c in [40,112,960]]);self.output=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU());self.head=PixelHead();self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        for p in self.features.parameters():p.requires_grad_(False)
        self.eval()
    def levels(self,images):
        x=(images.flatten(0,1)-self.mean)/self.std;levels=[]
        for i,l in enumerate(self.features):
            x=l(x)
            if i in [6,12,16]:levels.append(x)
        return levels
    def fuse(self,levels):return self.output(sum(F.interpolate(layer(x),size=levels[0].shape[-2:],mode='bilinear',align_corners=False)for layer,x in zip(self.lateral,levels)))
    def forward(self,**inputs):
        validate_tensor_inputs(inputs);im=inputs['images'];b,n=im.shape[:2];return self.head(self.fuse(self.levels(im)),im.shape[-2:]).reshape(b,n,*im.shape[-2:])

def align_teacher(feature,native_hw,resized_hw,student_hw=(48,80),stride=8):
    """Align conv receptive centers: native u=8j -> resized (u+.5)*scale-.5.

    Teacher FPN P3 conv centers at8k in resized padded image. Padding excluded
    through actual resized extents; bilinear sample align_corners=False.
    """
    h,w=student_hw;ih,iw=native_hw;rh,rw=resized_hw;fh,fw=feature.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=feature.device),torch.arange(w,device=feature.device),indexing='ij');u=((xx.float()*stride+.5)*rw/iw-.5)/8;v=((yy.float()*stride+.5)*rh/ih-.5)/8;grid=torch.stack((2*(u+.5)/fw-1,2*(v+.5)/fh-1),-1)[None].expand(len(feature),-1,-1,-1);return F.grid_sample(feature.float(),grid,align_corners=False,padding_mode='border')
