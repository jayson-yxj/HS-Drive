"""Coverage Sidecar -> explicit loss arguments; never called by model forward."""
from pathlib import Path
import json
import numpy as np
import torch

from training.m8j5b_coverage_loss import CoverageSupervision,integral_mask


def load_coverage_supervision(root,ordinal,device='cuda'):
    root=Path(root)
    record=json.loads((root/'sidecar'/(ordinal+'.json')).read_text())
    if record['authorization']!='LossDiagnosticOnly_NoOptimizer':raise ValueError('Unexpected contract')
    # A file can't certify empty 3D space just because image background is empty.
    if record['frame_negative_enabled'] or record['safe_negative_regions']:
        raise ValueError('Real source has no 3D negative certificate')
    with np.load(root/'masks_v2'/(ordinal+'.npz'),allow_pickle=False) as z:
        if set(z.files)!={'ignore','image_projection'}:raise ValueError('Mask archive schema')
        mask=torch.from_numpy(z['ignore'].copy()).to(device)
        projection=torch.from_numpy(z['image_projection'].copy()).to(device=device,dtype=torch.float32)
    rows=[o for o in record['objects'] if o['positive']['eligible']]
    boxes=torch.tensor([o['petr_box'] for o in rows],device=device,dtype=torch.float32).reshape(-1,9)
    labels=torch.tensor([{'car':0,'pedestrian':8}[o['positive']['model_class']] for o in rows],device=device,dtype=torch.long)
    unknown=torch.tensor(record['unknown_aabbs'],device=device,dtype=torch.float32).reshape(-1,6)
    return CoverageSupervision(boxes,labels,torch.zeros(len(rows),device=device,dtype=torch.bool),
        unknown,torch.empty((0,6),device=device),{},False,projection,integral_mask(mask),False,
        'ActualPayloadCoverage_NoAllClassNegativeCertificate:'+ordinal),record
