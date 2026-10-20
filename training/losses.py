"""Labels are consumed here, never by DepthBEV.forward."""
import torch
from torch.nn import functional as F


def logits_at_target_uv(logits,uv,image_hw):
    b,n,d,h,w=logits.shape;ih,iw=image_hw
    grid=uv.float().clone();grid[...,0]=2*(grid[...,0]+.5)/iw-1;grid[...,1]=2*(grid[...,1]+.5)/ih-1
    sampled=F.grid_sample(logits.float().flatten(0,1),grid.flatten(0,1),mode='bilinear',
                           padding_mode='border',align_corners=False)
    return sampled.reshape(b,n,d,*uv.shape[2:4])


def compute_losses(model,output,inputs,targets,*,surface_weight=.2,depth_log_weight=0.,depth_range_balance=False,drivable_weight=0.):
    logits=logits_at_target_uv(output['depth_logits'],targets['depth_uv'],inputs['images'].shape[-2:])
    labels=targets['depth_bin'];valid=targets['depth_valid']
    if not valid.any():raise ValueError('No valid depth supervision')
    labels=torch.where(valid,labels,torch.full_like(labels,-100))
    ce=F.cross_entropy(logits.flatten(0,1),labels.flatten(0,1),ignore_index=-100,reduction='none').reshape_as(labels)
    prediction=(logits.softmax(2)*model.splat.depth_centers[None,None,:,None,None]).sum(2)
    depth_gt=targets['depth_m'].float().clamp_min(1.5)
    regression=F.smooth_l1_loss(prediction.clamp_min(1.5).log(),depth_gt.log(),reduction='none',beta=.2)
    depth=masked_depth_mean(ce,depth_gt,valid,depth_range_balance)
    depth_log=masked_depth_mean(regression,depth_gt,valid,depth_range_balance)
    with torch.no_grad():
        counts=model.splat.observed_counts(targets['depth_m'],targets['depth_uv'],valid,
                                           inputs['intrinsics'],inputs['cam_to_ego'])
    errors=(output['surface_log_counts'].float()-torch.log1p(counts)).square()
    positive=counts>0
    pos=errors[positive].mean() if positive.any() else errors.sum()*0
    neg=errors[~positive].mean() if (~positive).any() else errors.sum()*0
    surface=.5*(pos+neg)
    drivable=depth*0
    if drivable_weight:
        if 'drivable_logits' not in output:raise ValueError('Drivable loss requires head')
        drivable=masked_drivable_loss(output['drivable_logits'],targets['drivable_static'],targets['drivable_valid'])
    return {'total':depth+surface_weight*surface+depth_log_weight*depth_log+drivable_weight*drivable,
            'depth_ce':depth,'surface_mse':surface,'depth_log':depth_log,'drivable_bce_dice':drivable},counts


def masked_depth_mean(errors,depth,valid,balance):
    if not balance:return errors[valid].mean()
    groups=[valid & (depth>=lo) & (depth<hi) for lo,hi in ((1.5,10),(10,30),(30,60),(60,80))]
    return torch.stack([errors[m].mean() for m in groups if m.any()]).mean()


def masked_drivable_loss(logits,labels,valid):
    if not valid.any():return logits.sum()*0
    x=logits.float()[valid];y=labels.float()[valid]
    if not ((y==0)|(y==1)).all():raise ValueError('Invalid supervised Drivable class')
    bce=F.binary_cross_entropy_with_logits(x,y,reduction='none')
    groups=[bce[y==c].mean() for c in (0,1) if (y==c).any()]
    probability=x.sigmoid()
    dice=1-(2*(probability*y).sum()+1)/(probability.sum()+y.sum()+1)
    return torch.stack(groups).mean()+dice


def loss_options(config):
    return dict(surface_weight=config['surface_loss_weight'],
                depth_log_weight=config.get('depth_log_loss_weight',0.),
                depth_range_balance=config.get('depth_range_balance',False),
                drivable_weight=config.get('drivable_loss_weight',0.))
