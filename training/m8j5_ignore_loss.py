"""Opt-in supervision-only wrapper around the frozen official PETR head.

No registry patch, forward override, trainable parameters, or inference metadata.
The caller supplies *already authorized* positives and negative certificates.
Spatial envelopes are detached; scores never decide Ignore/Negative status.
"""
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Supervision:
    # Old PETR center xyz,w,l,h,CW yaw,vx,vy. Only trusted positives.
    boxes: torch.Tensor
    labels: torch.Tensor
    velocity_valid: torch.Tensor
    # HS ego AABBs [xmin,ymin,zmin,xmax,ymax,zmax], same center coordinate axes.
    unknown_aabbs: torch.Tensor
    safe_aabbs: torch.Tensor
    negative_certificate: dict
    complete_annotation: bool = False

    def validate(self, num_classes):
        n = len(self.boxes)
        if self.boxes.shape != (n,9) or self.labels.shape != (n,) or self.velocity_valid.shape != (n,):
            raise ValueError('Positive contract shapes')
        if self.labels.dtype != torch.long or self.velocity_valid.dtype != torch.bool:
            raise ValueError('Positive contract dtype')
        if not torch.isfinite(self.boxes).all() or (self.boxes[:,3:6] <= 0).any():
            raise ValueError('Unreliable boxes cannot enter matching/regression')
        if ((self.labels < 0) | (self.labels >= num_classes)).any():
            raise ValueError('Untrusted/unsupported class')
        for regions in (self.unknown_aabbs,self.safe_aabbs):
            if regions.ndim != 2 or regions.shape[1] != 6 or not torch.isfinite(regions).all():
                raise ValueError('Invalid spatial certificate')
            if (regions[:,3:] <= regions[:,:3]).any():
                raise ValueError('Degenerate spatial certificate')
        if self.complete_annotation or len(self.safe_aabbs):
            cert = self.negative_certificate
            if cert.get('status') != 'Verified' or cert.get('class_ids') != list(range(num_classes)):
                raise ValueError('Safe negatives require all-class reviewed support')
            if not cert.get('evidence_ref') or not cert.get('reviewer'):
                raise ValueError('Missing negative certificate provenance')
            scope = 'CompleteAnnotation' if self.complete_annotation else 'BoundedRegion'
            if cert.get('scope') != scope:
                raise ValueError('Negative certificate scope mismatch')
        if self.complete_annotation and len(self.unknown_aabbs):
            raise ValueError('Complete annotation contradicts Unknown regions')


def query_envelopes(pred, margin=.5):
    """Conservative yaw-independent AABB; full XY circumcircle, full height.

    Uses regression geometry only, detached from gradients. Huge/underflowed
    boxes fail finite/positive checks and cannot become negatives. This is not
    a certificate that model geometry is physically correct.
    """
    p = pred.detach()
    center = p[:,[0,1,4]]
    dims = p[:,[2,3,5]].exp()
    radius = torch.linalg.vector_norm(dims[:,:2], dim=1) * .5
    half = torch.stack([radius,radius,dims[:,2]*.5],dim=1)+margin
    bounds = torch.cat([center-half,center+half],dim=1)
    valid = torch.isfinite(bounds).all(1) & (dims>0).all(1)
    return bounds, valid


def overlaps(bounds, regions):
    if not len(regions): return torch.zeros(len(bounds),dtype=torch.bool,device=bounds.device)
    # Boundary contact is conservatively an intersection, no arbitrary IoU threshold.
    return ((bounds[:,None,:3] <= regions[None,:,3:]).all(-1)
            & (bounds[:,None,3:] >= regions[None,:,:3]).all(-1)).any(1)


def certified_negative_mask(pred, spec, margin=.5):
    bounds, valid = query_envelopes(pred,margin)
    if not len(spec.safe_aabbs): return torch.zeros_like(valid)
    safe=spec.safe_aabbs.to(pred)
    contained=((bounds[:,None,:3] >= safe[None,:,:3]).all(-1)
               & (bounds[:,None,3:] <= safe[None,:,3:]).all(-1)).any(1)
    veto=overlaps(bounds,spec.unknown_aabbs.to(pred))
    # A certified empty region must not override known positive target support.
    if len(spec.boxes):
        from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
        known,_=query_envelopes(normalize_bbox(spec.boxes,None),margin)
        veto |= overlaps(bounds,known)
    return valid & contained & ~veto


class IgnoreAwarePETRLoss:
    """Explicit wrapper: no monkey-patching or persistent model modification."""
    def __init__(self, head, envelope_margin=.5):
        self.head = head
        self.margin = envelope_margin
        if envelope_margin < 0: raise ValueError('Negative uncertainty margin')
        if head.num_classes != 10 or head.code_size != 10 or not head.loss_cls.use_sigmoid:
            raise ValueError('Only audited single-frame ten-class PETR contract supported')

    def layer_loss(self, cls, box, supervision: List[Supervision]):
        from mmdet.core import reduce_mean
        from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
        h=self.head
        if cls.ndim!=3 or box.shape!=(*cls.shape[:2],10) or cls.shape[2]!=h.num_classes or len(supervision)!=len(cls):
            raise ValueError('Prediction batch shape mismatch')
        if not torch.isfinite(cls).all() or not torch.isfinite(box).all():
            raise ValueError('Nonfinite prediction: fail, do not hide with nan_to_num')
        all_labels=[];all_lw=[];targets=[];weights=[];counts=[]
        for c,p,spec in zip(cls,box,supervision):
            spec.validate(h.num_classes)
            # Official Hungarian matcher on trustworthy positives only. Construct
            # targets explicitly: legacy pseudo sampler returns [0,4] with no GT,
            # which cannot fill PETR's [0,9] targets (observed original failure).
            assignment=h.assigner.assign(p.detach(),c.detach(),spec.boxes,spec.labels,None)
            pos=torch.nonzero(assignment.gt_inds>0,as_tuple=False).squeeze(1)
            assigned=assignment.gt_inds[pos]-1
            labels=torch.full((len(p),),h.num_classes,device=p.device,dtype=torch.long)
            labels[pos]=spec.labels[assigned]
            target=p.new_zeros((len(p),9));target[pos]=spec.boxes[assigned]
            bw=torch.zeros_like(p);bw[pos]=1.
            if spec.complete_annotation:
                neg=torch.ones(len(p),device=p.device,dtype=torch.bool)
            else:
                neg=certified_negative_mask(p,spec,self.margin)
            neg[pos]=False
            lw=p.new_zeros(len(p));lw[neg]=1.;lw[pos]=1.
            # Official target weights are nonzero only for matched positives.
            if len(pos):
                bw[pos,8:10] *= spec.velocity_valid[assigned,None].to(bw)
            counts.append(dict(positive=int(pos.numel()),negative=int(neg.sum()),
                               ignore=int(len(p)-pos.numel()-neg.sum()),
                               state=torch.where(lw==0, -1, torch.where(labels<h.num_classes,1,0)).detach().cpu().tolist()))
            all_labels.append(labels);all_lw.append(lw);targets.append(target);weights.append(bw)
        labels=torch.cat(all_labels);lw=torch.cat(all_lw);target=torch.cat(targets);bw=torch.cat(weights)
        np=sum(x['positive'] for x in counts);nn=sum(x['negative'] for x in counts)
        avg=np+nn*h.bg_cls_weight
        if h.sync_cls_avg_factor: avg=reduce_mean(cls.new_tensor([avg]))
        avg=max(avg,1)
        lc=h.loss_cls(cls.reshape(-1,h.num_classes),labels,lw,avg_factor=avg)
        reg_avg=torch.clamp(reduce_mean(lc.new_tensor([np])),min=1).item()
        # Normalize positives only: avoid log(0) of dummy unmatched boxes.
        positive=(bw[:,:8].sum(1)>0)
        normalized=normalize_bbox(target[positive],h.pc_range)
        if not torch.isfinite(normalized).all(): raise ValueError('Nonfinite normalized positive')
        reg_weights=bw[positive]*h.code_weights
        lb=h.loss_bbox(box.reshape(-1,10)[positive],normalized,reg_weights,avg_factor=reg_avg)
        if not torch.isfinite(lc) or not torch.isfinite(lb): raise ValueError('Nonfinite loss')
        return lc,lb,dict(samples=counts,positive=np,negative=nn,ignore=sum(x['ignore'] for x in counts),
                          cls_avg_factor=float(avg),reg_avg_factor=reg_avg)

    def __call__(self,predictions,supervision):
        if predictions.get('enc_cls_scores') is not None or predictions.get('enc_bbox_preds') is not None:
            raise ValueError('Two-stage encoder loss outside audited PETR config')
        cls=predictions['all_cls_scores'].float();box=predictions['all_bbox_preds'].float()
        if cls.ndim!=4 or box.ndim!=4 or len(cls)!=len(box) or len(cls)<1:
            raise ValueError('Decoder layer shape mismatch')
        losses={};audit=[]
        for i,(c,p) in enumerate(zip(cls,box)):
            lc,lb,record=self.layer_loss(c,p,supervision)
            prefix='' if i==len(cls)-1 else 'd%d.'%i
            losses[prefix+'loss_cls']=lc;losses[prefix+'loss_bbox']=lb
            audit.append(dict(layer=i,**record))
        return losses,audit
