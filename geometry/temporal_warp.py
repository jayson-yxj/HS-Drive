"""Planar HS-ego memory warp; inverse sampling, explicit valid support and resets."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


def check_pose(pose):
    if pose.ndim!=3 or pose.shape[-2:]!=(4,4) or not torch.isfinite(pose).all():raise ValueError('Pose must be finite Bx4x4')
    r=pose[:,:3,:3];eye=torch.eye(3,device=pose.device).expand_as(r)
    if not torch.allclose(r.transpose(1,2)@r,eye,atol=1e-4) or not torch.allclose(torch.linalg.det(r),torch.ones(len(pose),device=pose.device),atol=1e-4):raise ValueError('Nonrigid pose')
    if not torch.allclose(pose[:,3],torch.tensor([0,0,0,1.],device=pose.device).expand(len(pose),4),atol=1e-5):raise ValueError('Pose last row')


def warp_bev(previous,previous_pose,current_pose,*,previous_valid=None,x_range=(-20.,60.),y_range=(-30.,30.),resolution=.5,max_tilt_rad=.05):
    """[B,C,X,Y], centers and align_corners=False. SE2 only; tilt rejected.

    Forward maps previous points to current. grid_sample needs its inverse:
    current output cell -> previous input cell. Invalid interpolation zeroed.
    """
    if previous.ndim!=4 or not previous.is_floating_point():raise ValueError('BEV must B,C,X,Y float')
    b,c,nx,ny=previous.shape
    if nx!=round((x_range[1]-x_range[0])/resolution) or ny!=round((y_range[1]-y_range[0])/resolution):raise ValueError('Grid shape mismatch')
    with torch.autocast(device_type=previous.device.type,enabled=False):
        pp=previous_pose.to(previous.device,dtype=torch.float32);cp=current_pose.to(previous.device,dtype=torch.float32);check_pose(pp);check_pose(cp)
        if len(pp)!=b or len(cp)!=b:raise ValueError('Batch mismatch')
        forward=torch.linalg.inv(cp)@pp;inverse=torch.linalg.inv(forward)
        tilt=torch.acos(forward[:,2,2].clamp(-1,1))
        if (tilt>max_tilt_rad).any() or (forward[:,2,3].abs()>.5).any():raise ValueError('Planar memory cannot represent tilt/elevation change')
        x=torch.arange(nx,device=previous.device)*resolution+x_range[0]+resolution/2;y=torch.arange(ny,device=previous.device)*resolution+y_range[0]+resolution/2
        xx,yy=torch.meshgrid(x,y,indexing='ij');points=torch.stack((xx,yy,torch.zeros_like(xx),torch.ones_like(xx)),-1)
        source=torch.einsum('bij,hwj->bhwi',inverse,points)
        grid=torch.stack((2*(source[...,1]-y_range[0])/(y_range[1]-y_range[0])-1,2*(source[...,0]-x_range[0])/(x_range[1]-x_range[0])-1),-1)
        support=torch.ones(b,1,nx,ny,device=previous.device) if previous_valid is None else previous_valid.to(previous.device,dtype=torch.float32)
        if support.shape!=(b,1,nx,ny):raise ValueError('Valid mask must B,1,X,Y')
        weight=F.grid_sample(support,grid,align_corners=False,mode='bilinear',padding_mode='zeros')
        valid=weight>=1-1e-5
        aligned=F.grid_sample(previous.float()*support,grid,align_corners=False,mode='bilinear',padding_mode='zeros')
        return aligned*valid,valid,forward


@dataclass
class BEVMemory:
    hidden: torch.Tensor
    pose: torch.Tensor
    timestamp_s: float
    scene_id: str
    valid: torch.Tensor

    def align(self,current_pose,timestamp_s,scene_id,*,max_dt=.5,max_translation=5.,max_yaw_rad=.6,**grid):
        if scene_id!=self.scene_id:return None,'scene_change'
        dt=timestamp_s-self.timestamp_s
        # CARLA float time steps accumulate ~1e-8 s around an exact 0.5 s gap.
        if not math.isfinite(dt) or not 0<dt<=max_dt+1e-6:return None,'timestamp_gap_or_reversal'
        try:
            check_pose(current_pose.float());relative=torch.linalg.inv(current_pose.float())@self.pose.float()
            yaw=torch.atan2(relative[:,1,0],relative[:,0,0]).abs()
            if (relative[:,:3,3].norm(dim=-1)>max_translation).any() or (yaw>max_yaw_rad).any():return None,'localization_jump'
            result=warp_bev(self.hidden,self.pose,current_pose,previous_valid=self.valid,**grid)
        except (ValueError,RuntimeError):return None,'invalid_pose_or_nonplanar'
        return result,'aligned'
