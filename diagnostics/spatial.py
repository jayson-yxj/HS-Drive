"""Matched-ray four-height splat experiments. No semantic/free occupancy labels."""
import torch
from torch.nn import functional as F
from models.view_transform import unproject


def ray_geometry(splat,uv,depth,intrinsics,cam_to_ego):
    pts=unproject(uv,depth,intrinsics,cam_to_ego);rank,inside=splat.ranks(pts)
    xy=(pts[...,0]>=splat.x_range[0])&(pts[...,0]<splat.x_range[1])&(pts[...,1]>=splat.y_range[0])&(pts[...,1]<splat.y_range[1])
    return pts,rank,inside,xy


def project(context,weights,points,splat):
    """One batch: context [R,C], weights [R,D], points [R,D,3]."""
    rank,inside=splat.ranks(points);cells=splat.nz*splat.nx*splat.ny
    ids=rank[inside];rays=torch.arange(len(context),device=context.device)[:,None].expand_as(weights)[inside]
    w=weights.float()[inside];features=context.float();mass=torch.zeros(cells,device=context.device)
    values=torch.zeros(cells,context.shape[1],device=context.device)
    for start in range(0,len(ids),splat.chunk_points):
        sl=slice(start,start+splat.chunk_points);mass.index_add_(0,ids[sl],w[sl]);values.index_add_(0,ids[sl],features[rays[sl]]*w[sl,None])
    values=values/mass.clamp_min(1e-6)[:,None]
    height=values.reshape(splat.nz,splat.nx,splat.ny,-1).permute(3,0,1,2)[None]
    return height,mass.reshape(1,splat.nz,splat.nx,splat.ny)


def downstream(model,height,mass):
    bev=model.bev_encoder(torch.cat((height.flatten(1,2),torch.log1p(mass)),1))
    return {'bev':bev,'surface':model.surface_head(bev),'road':model.drivable_head(bev)[:,0]}


def compare_mass(pred,oracle):
    p=pred.float();q=oracle.float();positive=q>0;empty=~positive;n=float(q.sum());total=float(p.sum())
    error=(torch.log1p(p)-torch.log1p(q)).abs()
    return {'oracle_count':n,'predicted_mass':total,'mass_ratio':total/n if n else None,
        'l1_mass_per_oracle_point':float((p-q).abs().sum())/n if n else None,
        'normalized_spatial_tv':float((p/p.sum().clamp_min(1e-9)-q/q.sum().clamp_min(1e-9)).abs().sum()/2) if n and total else None,
        'mass_fraction_on_gt_support':float(p[positive].sum())/total if total else None,
        'log_count_mae_all':float(error.mean()),'log_count_mae_gt_positive':float(error[positive].mean()) if positive.any() else None,
        'log_count_mae_zero_sample_cells':float(error[empty].mean()) if empty.any() else None,
        'layer_mass':[float(x) for x in p.sum((-1,-2))[0]],'layer_oracle_count':[float(x) for x in q.sum((-1,-2))[0]]}


def compare_features(pred,oracle,p_mass,q_mass):
    p=pred.float();q=oracle.float();mask=(p_mass>1e-6)&(q_mass>0)
    diff=(p-q).abs();union=(p_mass>1e-6)|(q_mass>0)
    common_cos=F.cosine_similarity(p,q,dim=1)
    return {'relative_l1_union':float(diff.permute(0,2,3,4,1)[union].sum()/q.abs().permute(0,2,3,4,1)[union].sum().clamp_min(1e-9)),
        'cosine_common_support':float(common_cos[mask].mean()) if mask.any() else None,'common_cells':int(mask.sum())}


def spatial_probability_metrics(p,bin_points,gt_points,gt_rank,inside_gt,splat):
    rank,inside=splat.ranks(bin_points);distance=(bin_points-gt_points[:,None]).norm(dim=-1)
    xy_distance=(bin_points[...,:2]-gt_points[:,None,:2]).norm(dim=-1)
    iz=torch.bucketize(bin_points[...,2].contiguous(),splat.z_edges,right=True)-1
    gt_iz=torch.bucketize(gt_points[...,2].contiguous(),splat.z_edges,right=True)-1
    values={'mass_projected_inside_bev':(p*inside).sum(-1),
            'mass_on_exact_gt_voxel':(p*(inside&inside_gt[:,None]&(rank==gt_rank[:,None]))).sum(-1),
            'expected_3d_error_m':(p*distance).sum(-1)}
    for radius in (1.,3.):
        values[f'mass_xy_within_{int(radius)}m_same_height']=(p*(inside&(iz==gt_iz[:,None])&(xy_distance<=radius))).sum(-1)
    return {k:v.cpu().numpy() for k,v in values.items()}
