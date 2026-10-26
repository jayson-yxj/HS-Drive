"""Explicit opt-in precision boundary and auditable loss decomposition.

The historical J5B module is unchanged. Disabled coverage is permitted only for
an exhaustive certificate, and dispatches to the actual official loss method.
"""
from dataclasses import replace

import torch

from training.m8j5b_coverage_loss import CoverageAwarePETRLoss


def fp32_supervision(spec):
    return replace(spec,boxes=spec.boxes.float(),unknown_aabbs=spec.unknown_aabbs.float(),
        safe_aabbs=spec.safe_aabbs.float(),image_projection=spec.image_projection.float())


class ValidatedCoveragePETRLoss(CoverageAwarePETRLoss):
    def __init__(self,head,coverage_enabled=True,**kwargs):
        super().__init__(head,**kwargs)
        if type(coverage_enabled) is not bool:raise ValueError('coverage_enabled must be bool')
        self.coverage_enabled=coverage_enabled

    def layer_loss(self,cls,box,supervision):
        # Casting inputs alone is insufficient under active CUDA autocast:
        # einsum/projected geometry can otherwise be downcast again.
        with torch.autocast(device_type=cls.device.type,enabled=False):
            c=cls.float();b=box.float();specs=[fp32_supervision(s) for s in supervision]
            lc,lb,audit=super().layer_loss(c,b,specs)
            labels=[];pweights=[];nweights=[];iweights=[]
            for scores,pred,spec,rec in zip(c,b,specs,audit['samples']):
                assignment=self.head.assigner.assign(pred.detach(),scores.detach(),spec.boxes,spec.labels,None)
                pos=assignment.gt_inds>0;assigned=assignment.gt_inds[pos]-1
                y=torch.full((len(pred),),self.head.num_classes,device=pred.device,dtype=torch.long)
                y[pos]=spec.labels[assigned]
                state=torch.tensor(rec['state'],device=pred.device)
                if not torch.equal(state==1,pos):raise RuntimeError('Repeated matching changed')
                labels.append(y);pweights.append((state==1).float());nweights.append((state==0).float());iweights.append((state==-1).float())
                rec.update(matched_gt_indices=assignment.gt_inds.detach().cpu().tolist(),
                    positive_class_ids=spec.labels[assigned].detach().cpu().tolist(),
                    classification_weight_sum=rec['positive']+rec['negative'],
                    effective_fraction=(rec['positive']+rec['negative'])/rec['query_count'])
            # Detached telemetry cannot introduce a second gradient path.
            with torch.no_grad():
                y=torch.cat(labels);flat=c.reshape(-1,self.head.num_classes)
                parts={name:float(self.head.loss_cls(flat,y,torch.cat(w),avg_factor=audit['cls_avg_factor']))
                       for name,w in [('positive',pweights),('negative',nweights)]}
                if not torch.isclose(lc.detach(),lc.new_tensor(sum(parts.values())),atol=1e-6,rtol=1e-5):
                    raise RuntimeError('Loss decomposition mismatch')
            audit.update(classification_weight_sum=audit['positive']+audit['negative'],
                effective_fraction=(audit['positive']+audit['negative'])/sum(r['query_count'] for r in audit['samples']),
                classification_loss_positive=parts['positive'],classification_loss_negative=parts['negative'],
                classification_loss_ignore=0.,classification_loss=float(lc.detach()),regression_loss=float(lb.detach()),
                precision='FP32_ExplicitAutocastDisabled',gradient_scope='DirectQueryLogitsAndBoxOutputs_NotAllSharedParameters')
            return lc,lb,audit

    def __call__(self,predictions,supervision):
        if self.coverage_enabled:return super().__call__(predictions,supervision)
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        specs=[fp32_supervision(s) for s in supervision]
        for s in specs:
            s.validate(self.head.num_classes)
            if not s.complete_annotation:raise ValueError('Cannot disable coverage for partial/unknown labels')
        cls=predictions['all_cls_scores']
        with torch.autocast(device_type=cls.device.type,enabled=False):
            p={k:v.float() if isinstance(v,torch.Tensor) else v for k,v in predictions.items()}
            p.setdefault('enc_cls_scores',None);p.setdefault('enc_bbox_preds',None)
            boxes=[]
            for s in specs:
                bottom=s.boxes.clone();bottom[:,2]-=bottom[:,5]/2
                boxes.append(LiDARInstance3DBoxes(bottom,box_dim=9))
            result=self.head.loss(boxes,[s.labels for s in specs],p)
        return result,[dict(mode='CoverageDisabled_OfficialHeadLoss',supervision='VerifiedCompleteOnly')]
