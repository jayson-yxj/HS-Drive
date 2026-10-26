"""Human-reviewed private supervision -> opt-in PETR Loss, never RGB metadata.

Current real data has no certified negatives. This adapter cannot fabricate
them, and it never enables complete-annotation mode from a partial Sidecar.
"""
import torch

from datasets.m8j5_supervision import hs_to_petr_center, validate_signoff
from training.m8j5_ignore_loss import Supervision


def build_reviewed_supervision(frame, candidates, responses, device='cuda'):
    lookup={c['candidate_id']:c for c in candidates if c['ordinal']==frame['ordinal']}
    approved=set()
    seen=set()
    for response in responses:
        key=response.get('candidate_id')
        if key not in lookup:continue
        if key in seen:raise ValueError('Duplicate review')
        seen.add(key)
        if validate_signoff(lookup[key],response):approved.add(key)
    boxes=[];labels=[];unknown=[]
    for row in frame['objects']:
        obj=row['object'];key=row['candidate_id']
        if key in approved:
            # Review must bind the unchanged source record, not merely the ID.
            if obj!=lookup[key]['object']:raise ValueError('Reviewed object changed')
            boxes.append(hs_to_petr_center(obj))
            labels.append({'car':0,'pedestrian':8}[obj['model_class']])
        elif obj['geometry_status']=='Valid':
            # Plausible observed extent only; geometry-unknown objects instead
            # prevent safe-region certification. No automatic region certification.
            corners=torch.tensor(obj['corners_hs_m'],dtype=torch.float32)
            unknown.append(torch.cat([corners.min(0).values-.5,corners.max(0).values+.5]).tolist())
    return Supervision(torch.tensor(boxes,device=device,dtype=torch.float32).reshape(-1,9),
        torch.tensor(labels,device=device,dtype=torch.long),
        torch.zeros(len(boxes),device=device,dtype=torch.bool),
        torch.tensor(unknown,device=device,dtype=torch.float32).reshape(-1,6),
        torch.empty((0,6),device=device),{},False)
