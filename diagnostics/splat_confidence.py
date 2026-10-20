"""Opt-in M4.5 aggregation research. Mass is accumulated evidence, not occupancy."""
import torch
from models.view_transform import HeightSplat,feature_uv,unproject


def aggregate_sum(weighted_sum,mass,mode='baseline',tau=None,eps=1e-6):
    if mode=='baseline':return weighted_sum/mass.clamp_min(eps).unsqueeze(1)
    if mode=='sum':return weighted_sum
    if mode=='gated':
        if tau is None or not 0<float(tau)<float('inf'):raise ValueError('Finite positive tau required')
        return weighted_sum/(mass+tau).unsqueeze(1)
    raise ValueError('Unknown aggregation')


class ConfidenceSplat(HeightSplat):
    """Same buffers/checkpoint shape; default delegates to unchanged original path."""
    def __init__(self,config,chunk_points=32768,mode='baseline',tau=None):
        super().__init__(config,chunk_points);self.mode=mode;self.tau=tau
        if mode not in ('baseline','sum','gated'):raise ValueError('Invalid mode')
        if mode=='gated' and (tau is None or not 0<tau<float('inf')):raise ValueError('Invalid tau')

    def forward(self,context,depth_probs,intrinsics,cam_to_ego,image_hw):
        if self.mode=='baseline':return super().forward(context,depth_probs,intrinsics,cam_to_ego,image_hw)
        b,n,c,h,w=context.shape;d=depth_probs.shape[2]
        if depth_probs.shape!=(b,n,len(self.depth_centers),h,w):raise ValueError('Depth/context mismatch')
        with torch.autocast(device_type=context.device.type,enabled=False):
            uv=feature_uv(h,w,image_hw,context.device).expand(b,n,d,h,w,2)
            depths=self.depth_centers.view(1,1,d,1,1).expand(b,n,d,h,w)
            points=unproject(uv,depths,intrinsics,cam_to_ego);ranks,valid=self.ranks(points)
            cells=self.nz*self.nx*self.ny;ranks=ranks+torch.arange(b,device=context.device).view(b,1,1,1,1)*cells
            rays=torch.arange(b*n*h*w,device=context.device).reshape(b,n,1,h,w).expand(b,n,d,h,w)
            ids=ranks[valid];rays=rays[valid];weight=depth_probs.float()[valid]
            features=context.float().permute(0,1,3,4,2).reshape(-1,c)
            result=features.new_zeros(b*cells,c);mass=features.new_zeros(b*cells)
            for start in range(0,len(ids),self.chunk_points):
                sl=slice(start,start+self.chunk_points);result.index_add_(0,ids[sl],features[rays[sl]]*weight[sl,None]);mass.index_add_(0,ids[sl],weight[sl])
            mass=mass.reshape(b,self.nz,self.nx,self.ny)
            weighted=result.reshape(b,self.nz,self.nx,self.ny,c).permute(0,4,1,2,3)
            return aggregate_sum(weighted,mass,self.mode,self.tau),mass


def feature_buckets(height,mass,oracle):
    """Per-layer x observed support x mass bucket. Zero samples != Free."""
    buckets=[(-1,1e-6),(1e-6,1e-3),(1e-3,1e-2),(1e-2,1e-1),(1e-1,float('inf'))]
    l1=height.float().abs().sum(1);norm=height.float().norm(dim=1);total=float(l1.sum());result={}
    for z in range(mass.shape[1]):
        for observed in (False,True):
            for i,(lo,hi) in enumerate(buckets):
                m=(mass[:,z]>lo)&(mass[:,z]<=hi)&((oracle[:,z]>0)==observed)
                result[f'z{z}/observed{int(observed)}/bucket{i}']={'voxels':int(m.sum()),'mass':float(mass[:,z][m].sum()),'feature_l1':float(l1[:,z][m].sum()),'feature_norm_sum':float(norm[:,z][m].sum())}
    weak=(mass>1e-6)&(mass<=1e-3)
    return {'groups':result,'feature_l1_total':total,'weak_mass_feature_fraction':float(l1[weak].sum())/max(total,1e-12)}
