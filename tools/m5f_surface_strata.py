"""Explicit ground/target/other/ignored semantic strata on saved singleframe predictions."""
import json
from pathlib import Path
import numpy as np
from tools.m5f_common import pixel_label
from diagnostics.m5f_metrics import histogram,metrics
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_surface_strata.json')
    if dest.exists():raise FileExistsError(dest)
    rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels/index.json').read_text())['records'];study=json.loads(Path('reports/m5f_2d_presence_absence.json').read_text())['records'];out=[]
    for model in study:
        hist={};unknown=[]
        for row in rows:
            d=pixel_label(row);raw=Path(row['raw_frame']).parent.parent;frame=json.loads(Path(row['raw_frame']).read_text());semantic=np.array([np.load(raw/frame['cameras'][c]['instance'])['semantic'] for c in row['camera_names']]);rec=next(p for p in model['predictions'] if p['sample_id']==row['sample_id']);score=np.load(rec['path'])['probability_u8']/256
            masks=dict(ground_surface=d['valid']&np.isin(semantic,[1,2,10,24,25]),target_actor_surface=d['valid']&d['target'],other_confirmed_nontarget=d['valid']&~d['target']&~np.isin(semantic,[1,2,10,24,25]))
            for key,mask in masks.items():hist[key]=hist.get(key,np.zeros((2,256),np.int64))+histogram(score,d['target'],mask)
            unknown.append(dict(sample_id=row['sample_id'],pixels=int((~d['valid']).sum()),mean_score=float(score[~d['valid']].mean()),high_fraction=float((score[~d['valid']]>=int(model['threshold']*256)/256).mean())))
        out.append(dict(name=model['name'],seed=model['seed'],threshold=model['threshold'],strata={k:metrics(h,model['threshold']) for k,h in hist.items()},unknown=unknown))
    atomic_json(dest,dict(kind='offline_semantic_strata_from_saved_GPU_probabilities',records=out,source_sha256=file_hash(__file__),schema_correction='In original m5f_evaluate strata the legacy key ground means ALL confirmed non-target pixels, NOT Ground semantic. Use this explicit semantic breakdown for Ground. No numeric/model/threshold changes.',limits='TargetActor includesvehicles/ped/cone; walls/structures remain nontargetforimageclass but physicalobstacle. Unknown notnegative. CoarseBEVmixed positivesandnegatives ignored by reference, notimageclass.'))

if __name__=='__main__':main()
