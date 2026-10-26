"""Opt-in coverage-aware PETR loss. All spatial evidence is supervision-only.

Unknown image evidence vetoes background gradients, never creates a 3D Box.
Finite detached full-query envelopes are projected with the actual crop K/T.
An image free of semantic foreground is NOT a safe 3D negative certificate.
"""
from dataclasses import dataclass
import warnings

import torch

from training.m8j5_ignore_loss import (
    Supervision, IgnoreAwarePETRLoss, query_envelopes, certified_negative_mask,
)


@dataclass
class CoverageSupervision(Supervision):
    # [V,3,4] optical, preprocessed image pixel projection, not CARLA camera axes.
    image_projection: torch.Tensor = None
    # [V,H+1,W+1] summed-area table of boolean Ignore pixels.
    ignore_integral: torch.Tensor = None
    frame_negative_enabled: bool = False
    coverage_ref: str = ''

    def validate(self, num_classes):
        super().validate(num_classes)
        if self.image_projection is None or self.ignore_integral is None:
            raise ValueError('Missing coverage calibration or mask')
        m=self.image_projection;ii=self.ignore_integral
        if m.ndim!=3 or m.shape[1:]!=(3,4) or ii.ndim!=3 or len(m)!=len(ii) or not len(m):
            raise ValueError('Coverage view shape mismatch')
        if min(ii.shape[1:])<2 or not torch.isfinite(m).all() or not torch.isfinite(ii).all():
            raise ValueError('Nonfinite/empty coverage')
        if not self.coverage_ref or type(self.frame_negative_enabled) is not bool:
            raise ValueError('Missing coverage evidence')
        if ii[:,0].any() or ii[:,:,0].any():raise ValueError('Integral mask origin invalid')
        pixels=ii[:,1:,1:]-ii[:,:-1,1:]-ii[:,1:,:-1]+ii[:,:-1,:-1]
        if not ((pixels==0)|(pixels==1)).all():raise ValueError('Ignore integral must encode boolean pixels')
        if self.complete_annotation and (not self.frame_negative_enabled or ii[:,-1,-1].any()):
            raise ValueError('Complete annotation contradicts incomplete coverage')


def integral_mask(mask):
    if mask.ndim!=3 or mask.dtype!=torch.bool:raise ValueError('Expected boolean VHW mask')
    return torch.nn.functional.pad(mask.to(torch.int64).cumsum(1).cumsum(2),(1,0,1,0))


def projected_ignore(pred, spec, margin=.5):
    """Conservative rectangle of projected full AABB, with any-pixel veto.

    No score gate, 2D-to-3D invention, or predicted-depth occupancy certificate.
    Near-plane intersections and boxes with no in-crop projection are unknown.
    """
    bounds, valid=query_envelopes(pred,margin)
    device=pred.device
    bits=torch.tensor([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)],device=device,dtype=torch.bool)
    corners=torch.where(bits[None],bounds[:,None,3:],bounds[:,None,:3])
    points=torch.cat([corners,torch.ones_like(corners[:,:,:1])],-1)
    matrix=spec.image_projection.to(pred)
    q=torch.einsum('vij,nkj->vnki',matrix,points)
    z=q[:,:,:,2];front=z>.1
    crossing=(front.any(-1)&~front.all(-1)).any(0)
    uv=q[:,:,:,:2]/z.clamp(min=.1)[...,None]
    ii=spec.ignore_integral.to(device);height,width=ii.shape[1]-1,ii.shape[2]-1
    low=uv.min(2).values;high=uv.max(2).values
    visible=front.all(-1)&(high[:,:,0]>=0)&(high[:,:,1]>=0)&(low[:,:,0]<width)&(low[:,:,1]<height)
    # Huge/nonfinite extents are never background; clamp before integer cast.
    safe_low=torch.nan_to_num(low,nan=-1e8,posinf=1e8,neginf=-1e8)
    safe_high=torch.nan_to_num(high,nan=1e8,posinf=1e8,neginf=-1e8)
    x0=safe_low[:,:,0].floor().clamp(0,width).long();y0=safe_low[:,:,1].floor().clamp(0,height).long()
    x1=(safe_high[:,:,0].ceil()+1).clamp(0,width).long();y1=(safe_high[:,:,1].ceil()+1).clamp(0,height).long()
    views=torch.arange(len(ii),device=device)[:,None]
    area=ii[views,y1,x1]-ii[views,y0,x1]-ii[views,y1,x0]+ii[views,y0,x0]
    hit=((area>0)&visible).any(0)
    no_view=~visible.any(0)
    # The isolated Torch 2.0 build does not support tuple dims for Tensor.all.
    invalid=~valid | ~torch.isfinite(q).all(3).all(2).all(0)
    veto=hit|crossing|no_view|invalid
    return veto,dict(image_ignore_overlap=int(hit.sum()),near_plane_ambiguous=int(crossing.sum()),
        no_in_crop_projection=int(no_view.sum()),invalid_envelope=int(invalid.sum()),
        image_veto_count=int(veto.sum()))


class CoverageAwarePETRLoss(IgnoreAwarePETRLoss):
    def __init__(self,head,envelope_margin=.5,high_ignore_fraction=.95):
        super().__init__(head,envelope_margin)
        if not 0<=high_ignore_fraction<=1:raise ValueError('Invalid warning threshold')
        self.high_ignore_fraction=high_ignore_fraction

    def layer_loss(self,cls,box,supervision):
        from mmdet.core import reduce_mean
        from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
        h=self.head
        if cls.ndim!=3 or box.shape!=(*cls.shape[:2],10) or cls.shape[2]!=h.num_classes or len(supervision)!=len(cls) or not len(cls) or cls.shape[1]==0:
            raise ValueError('Prediction shape mismatch')
        if not torch.isfinite(cls).all() or not torch.isfinite(box).all():raise ValueError('Nonfinite prediction')
        all_labels=[];all_lw=[];targets=[];weights=[];counts=[]
        for c,p,spec in zip(cls,box,supervision):
            if not isinstance(spec,CoverageSupervision):raise ValueError('Coverage contract required')
            spec.validate(h.num_classes)
            assignment=h.assigner.assign(p.detach(),c.detach(),spec.boxes,spec.labels,None)
            pos=torch.nonzero(assignment.gt_inds>0,as_tuple=False).squeeze(1)
            assigned=assignment.gt_inds[pos]-1
            labels=torch.full((len(p),),h.num_classes,device=p.device,dtype=torch.long)
            labels[pos]=spec.labels[assigned]
            target=p.new_zeros((len(p),9));target[pos]=spec.boxes[assigned]
            bw=torch.zeros_like(p);bw[pos]=1
            image_veto,detail=projected_ignore(p,spec,self.margin)
            if spec.complete_annotation:
                # Parity only for a genuine exhaustive all-class certificate.
                neg=torch.ones(len(p),device=p.device,dtype=torch.bool)
            else:
                neg=certified_negative_mask(p,spec,self.margin)&~image_veto
                if not spec.frame_negative_enabled:neg[:]=False
            neg[pos]=False
            lw=p.new_zeros(len(p));lw[neg]=1;lw[pos]=1
            if len(pos):bw[pos,8:10]*=spec.velocity_valid[assigned,None].to(bw)
            npos=len(pos);nneg=int(neg.sum());nignore=len(p)-npos-nneg
            alerts=[]
            if len(p) and nignore/len(p)>=self.high_ignore_fraction:alerts.append('HighIgnoreFraction')
            if nneg==0:alerts.append('NoNegativeSupervision')
            if alerts:warnings.warn('Coverage supervision: '+','.join(alerts),RuntimeWarning,stacklevel=2)
            counts.append(dict(positive=npos,negative=nneg,ignore=nignore,query_count=len(p),
                ignore_fraction=nignore/max(1,len(p)),warnings=alerts,
                frame_negative_enabled=spec.frame_negative_enabled,
                state=torch.where(lw==0,-1,torch.where(labels<h.num_classes,1,0)).detach().cpu().tolist(),
                positive_in_image_veto=int(image_veto[pos].sum()),**detail))
            all_labels.append(labels);all_lw.append(lw);targets.append(target);weights.append(bw)
        labels=torch.cat(all_labels);lw=torch.cat(all_lw);target=torch.cat(targets);bw=torch.cat(weights)
        npos=sum(x['positive'] for x in counts);nneg=sum(x['negative'] for x in counts)
        avg=npos+nneg*h.bg_cls_weight
        if h.sync_cls_avg_factor:avg=reduce_mean(cls.new_tensor([avg]))
        avg=max(avg,1)
        lc=h.loss_cls(cls.reshape(-1,h.num_classes),labels,lw,avg_factor=avg)
        reg_avg=torch.clamp(reduce_mean(lc.new_tensor([npos])),min=1).item()
        positive=bw[:,:8].sum(1)>0
        normalized=normalize_bbox(target[positive],h.pc_range)
        if not torch.isfinite(normalized).all():raise ValueError('Nonfinite reliable GT')
        lb=h.loss_bbox(box.reshape(-1,10)[positive],normalized,bw[positive]*h.code_weights,avg_factor=reg_avg)
        if not torch.isfinite(lc) or not torch.isfinite(lb):raise ValueError('Nonfinite loss')
        return lc,lb,dict(samples=counts,positive=npos,negative=nneg,ignore=sum(x['ignore'] for x in counts),
            cls_avg_factor=float(avg),reg_avg_factor=reg_avg)
