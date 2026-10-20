"""V1.3 single-frame RGB-only Depth + height-aware BEV baseline."""
import torch
from torch import nn
from datasets.inputs import validate_tensor_inputs
from models.image_encoder import ImageEncoder
from models.view_transform import HeightSplat


class DepthBEV(nn.Module):
    def __init__(self,bev_config,*,weights_path=None,chunk_points=32768,drivable=False,road_context=False):
        super().__init__()
        self.encoder=ImageEncoder(64,weights_path)
        self.camera_mlp=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,64))
        self.depth_head=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU(),nn.Conv2d(64,48,1))
        self.context_head=nn.Conv2d(64,64,1)
        self.splat=HeightSplat(bev_config,chunk_points)
        if len(self.splat.depth_centers)!=48 or self.splat.nz!=4:raise ValueError('V1.3 baseline requires 48 depth/4 height bins')
        self.bev_encoder=nn.Sequential(nn.Conv2d(64*4+4,128,3,padding=1),nn.GroupNorm(8,128),nn.ReLU(),
                                       nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128),nn.ReLU())
        # Predict log(1+sampled-surface-count); not semantic occupancy.
        self.surface_head=nn.Conv2d(128,4,1)
        if road_context and not drivable:raise ValueError('Road context requires Drivable head')
        if road_context:
            layers=[nn.Conv2d(128,64,1),nn.GroupNorm(8,64),nn.ReLU()]
            for dilation in (1,2,4,8):
                layers.extend([nn.Conv2d(64,64,3,padding=dilation,dilation=dilation),nn.GroupNorm(8,64),nn.ReLU()])
            layers.append(nn.Conv2d(64,1,1));self.drivable_head=nn.Sequential(*layers)
        else:self.drivable_head=nn.Conv2d(128,1,1) if drivable else None

    def forward(self,*,images,intrinsics,cam_to_ego,ego_pose,ego_state,route_command,timestamps_s,valid_cameras):
        validate_tensor_inputs(dict(images=images,intrinsics=intrinsics,cam_to_ego=cam_to_ego,
            ego_pose=ego_pose,ego_state=ego_state,route_command=route_command,timestamps_s=timestamps_s,valid_cameras=valid_cameras))
        if images.ndim!=5 or images.shape[-2]%32 or images.shape[-1]%32:
            raise ValueError('Single frame [B,6,3,H,W], dimensions divisible by 32 required')
        if not valid_cameras.all():raise ValueError('Incomplete camera rig: baseline refuses missing inputs')
        b,n,_,ih,iw=images.shape
        features=self.encoder(images);_,_,c,h,w=features.shape
        camera=torch.stack((intrinsics[...,0,0]/iw,intrinsics[...,1,1]/ih,
                            intrinsics[...,0,2]/iw,intrinsics[...,1,2]/ih),-1)
        extrinsic=torch.cat((cam_to_ego[...,:3,:3].flatten(-2),cam_to_ego[...,:3,3]/10),-1)
        conditioning=self.camera_mlp(torch.cat((camera,extrinsic),-1)).reshape(b*n,c,1,1)
        flat=features.flatten(0,1)
        logits=self.depth_head(flat+conditioning).reshape(b,n,48,h,w)
        context=self.context_head(flat).reshape(b,n,64,h,w)
        probabilities=logits.float().softmax(2)
        height,mass=self.splat(context,probabilities,intrinsics,cam_to_ego,(ih,iw))
        bev=self.bev_encoder(torch.cat((height.flatten(1,2),torch.log1p(mass)),1))
        result={'depth_logits':logits,'depth_probs':probabilities,'height_bev':height,
                'bev_current':bev,'surface_log_counts':self.surface_head(bev),'splat_mass':mass}
        if self.drivable_head is not None:result['drivable_logits']=self.drivable_head(bev)[:,0]
        return result
