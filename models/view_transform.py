"""FP32 camera geometry and differentiable, chunked height-aware scatter-splat.

No complete B*N*D*C*H*W lifted feature volume is materialized. Grid indexing
is [height, X, Y], matching the Oracle implementation; bounds are half-open.
"""
import torch
from torch import nn
from datasets.labels import depth_bin_edges


def feature_uv(height,width,image_hw,device):
    ih,iw=image_hw
    v=(torch.arange(height,device=device,dtype=torch.float32)+.5)*ih/height-.5
    u=(torch.arange(width,device=device,dtype=torch.float32)+.5)*iw/width-.5
    vv,uu=torch.meshgrid(v,u,indexing='ij')
    return torch.stack((uu,vv),-1)


def unproject(uv,depth,intrinsics,cam_to_ego):
    """uv=[B,N,...,2], depth=[B,N,...]; no autocast allowed for geometry."""
    b,n=depth.shape[:2]
    shape=depth.shape
    homogeneous=torch.cat((uv.float(),torch.ones_like(uv[...,:1])),dim=-1).reshape(b,n,-1,3)
    optical=torch.einsum('bnij,bnpj->bnpi',torch.linalg.inv(intrinsics.float()),homogeneous)
    optical=optical*depth.float().reshape(b,n,-1,1)
    points=torch.einsum('bnij,bnpj->bnpi',cam_to_ego[...,:3,:3].float(),optical)
    points=points+cam_to_ego[...,:3,3].float().unsqueeze(-2)
    return points.reshape(*shape,3)


class HeightSplat(nn.Module):
    def __init__(self,config,chunk_points=32768):
        super().__init__()
        if chunk_points<1:raise ValueError('chunk_points must be positive')
        self.chunk_points=chunk_points
        self.resolution=float(config['resolution_m'])
        self.x_range=tuple(config['x_range_m']);self.y_range=tuple(config['y_range_m'])
        self.nx=round((self.x_range[1]-self.x_range[0])/self.resolution)
        self.ny=round((self.y_range[1]-self.y_range[0])/self.resolution)
        self.register_buffer('z_edges',torch.tensor(config['z_edges_m'],dtype=torch.float32))
        net=config['network'];edges=depth_bin_edges(net['depth_bins'],*net['depth_range_m'])
        self.register_buffer('depth_centers',torch.tensor((edges[1:]+edges[:-1])/2,dtype=torch.float32))
        self.nz=len(config['z_edges_m'])-1
        if self.resolution<=0 or self.nx<1 or self.ny<1 or not torch.all(self.z_edges.diff()>0):
            raise ValueError('Invalid BEV grid')

    def ranks(self,points):
        x,y,z=points.unbind(-1)
        ix=torch.floor((x-self.x_range[0])/self.resolution).long()
        iy=torch.floor((y-self.y_range[0])/self.resolution).long()
        iz=torch.bucketize(z.contiguous(),self.z_edges,right=True)-1
        valid=(torch.isfinite(points).all(-1)&(x>=self.x_range[0])&(x<self.x_range[1])&
               (y>=self.y_range[0])&(y<self.y_range[1])&(iz>=0)&(iz<self.nz))
        return (iz*self.nx+ix)*self.ny+iy,valid

    def forward(self,context,depth_probs,intrinsics,cam_to_ego,image_hw):
        b,n,c,h,w=context.shape;d=depth_probs.shape[2]
        if depth_probs.shape!=(b,n,len(self.depth_centers),h,w):raise ValueError('Depth/context shape mismatch')
        with torch.autocast(device_type=context.device.type,enabled=False):
            uv=feature_uv(h,w,image_hw,context.device).expand(b,n,d,h,w,2)
            depths=self.depth_centers.view(1,1,d,1,1).expand(b,n,d,h,w)
            points=unproject(uv,depths,intrinsics,cam_to_ego)
            ranks,valid=self.ranks(points)
            cells=self.nz*self.nx*self.ny
            ranks=ranks+torch.arange(b,device=context.device).view(b,1,1,1,1)*cells
            ray_ids=torch.arange(b*n*h*w,device=context.device).reshape(b,n,1,h,w).expand(b,n,d,h,w)
            ids=ranks[valid];rays=ray_ids[valid];weights=depth_probs.float()[valid]
            features=context.float().permute(0,1,3,4,2).reshape(-1,c)
            output=features.new_zeros((b*cells,c));mass=features.new_zeros(b*cells)
            for start in range(0,len(ids),self.chunk_points):
                sl=slice(start,start+self.chunk_points)
                output.index_add_(0,ids[sl],features[rays[sl]]*weights[sl,None])
                mass.index_add_(0,ids[sl],weights[sl])
            # Weighted average retains depth-dependent mass as a separate feature.
            output=output/mass.clamp_min(1e-6).unsqueeze(-1)
            height=output.reshape(b,self.nz,self.nx,self.ny,c).permute(0,4,1,2,3)
            return height,mass.reshape(b,self.nz,self.nx,self.ny)

    def observed_counts(self,depth,uv,valid,intrinsics,cam_to_ego):
        """Loss/evaluation only: sampled visible surfaces, NOT occupancy/free GT."""
        with torch.autocast(device_type=depth.device.type,enabled=False):
            points=unproject(uv,depth,intrinsics,cam_to_ego)
            ranks,inside=self.ranks(points)
            valid=valid&inside;b=depth.shape[0];cells=self.nz*self.nx*self.ny
            ranks=ranks+torch.arange(b,device=depth.device).view(b,1,1,1)*cells
            result=torch.zeros(b*cells,device=depth.device)
            result.index_add_(0,ranks[valid],torch.ones_like(ranks[valid],dtype=torch.float32))
            return result.reshape(b,self.nz,self.nx,self.ny)
