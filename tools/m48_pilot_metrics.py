"""Per-channel semantic metrics and collapse diagnostics from saved pilot output."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash

def main():
    dest=Path('reports/m48_pilot_metrics.json')
    if dest.exists():raise FileExistsError(dest)
    root=Path('reports/m48_pilot_artifacts')
    with np.load(root/'labels.npz') as z:labels={k:z[k] for k in z.files}
    records=[]
    for name in ['rgb_bev','position_prior']:
        with np.load(root/(name+'_predictions.npz')) as z:pred=z['semantic']
        for i,kind in enumerate(['ground','obstacle','other']):
            mask=labels['obstacle_valid'] if kind=='obstacle' else labels['surface_semantic_valid'];truth=labels['surface_semantic'][:,i];out=pred[:,i]>.5
            tp=int((out&truth&mask).sum());fp=int((out&~truth&mask).sum());fn=int((~out&truth&mask).sum());tn=int((~out&~truth&mask).sum())
            records.append(dict(model=name,semantic=kind,tp=tp,fp=fp,fn=fn,tn=tn,positive_recall=tp/(tp+fn) if tp+fn else None,iou=tp/(tp+fp+fn) if tp+fp+fn else None,negative_recall=tn/(tn+fp) if tn+fp else None))
    atomic_json(dest,dict(kind='offline_two_frame_training_set_metrics',records=records,source_sha256=file_hash(__file__),label_sha256=file_hash(root/'labels.npz'),scope='Other has no positive support if count=0; do not claim its learnability. No generalization evaluation.'))
if __name__=='__main__':main()
