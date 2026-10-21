"""Confirmed background semantic errors; no unknown-as-negative or mining on test."""
import json
from pathlib import Path
import numpy as np
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5g_background_audit.json')
    if dest.exists():raise FileExistsError(dest)
    rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5g_test_pixels/index.json').read_text())['records'];models=json.loads(Path('reports/m5g_pixel_test.json').read_text())['records'];pre=json.loads(Path('reports/m5g_pretrained_test.json').read_text());metadata=[]
    for r in rows:
        d=pixel_label(r);raw=Path(r['raw_frame']).parent.parent;f=json.loads(Path(r['raw_frame']).read_text());sem=np.array([np.load(raw/f['cameras'][c]['instance'])['semantic']for c in r['camera_names']]);metadata.append((d,sem))
    out=[]
    for model in models+[dict(name='COCO_supported',seed=0,result=pre['results']['supported'],saved=pre['saved'])]:
        stats={};unknown=[];th=model['result']['threshold'];cut=int(th*256)/256
        for r,(d,sem)in zip(rows,metadata):
            with np.load(next(p['path']for p in model['saved']if p['sample_id']==r['sample_id']))as z:score=z['probability_u8']/256 if 'probability_u8'in z else np.maximum(z['vehicle'],z['pedestrian'])/256
            neg=d['valid']&~d['target'];classes={'road_ground':np.isin(sem,[1,10,24,25]),'sidewalk':sem==2,'building_wall':np.isin(sem,[3,4]),'fence_guardrail':np.isin(sem,[5,28]),'vegetation':sem==9,'static_road_furniture':np.isin(sem,[6,7,8]),'other_confirmed':np.isin(sem,[26,27])}
            for name,mask in classes.items():
                mask &= neg;n=int(mask.sum());fp=int(((score>=cut)&mask).sum());s=stats.setdefault(name,dict(n=0,fp=0,score_sum=0.));s['n']+=n;s['fp']+=fp;s['score_sum']+=float(score[mask].sum())
            unknown.append(dict(group=r['group'],configuration=r['configuration'],n=int((~d['valid']).sum()),mean=float(score[~d['valid']].mean()),high_fraction=float((score[~d['valid']]>=cut).mean())))
        out.append(dict(name=model['name'],seed=model['seed'],threshold=th,known_background={k:dict(**v,fpr=v['fp']/v['n']if v['n']else None)for k,v in stats.items()},unknown=unknown))
    atomic_json(dest,dict(kind='offline_saved_frozen_test_predictions_semantic_background_audit',records=out,source_sha256=file_hash(__file__),limits='Allnegative=currentconfirmednontargetsurface. BuildingwallnegativeforActorclass only not safetyFree. Trueobject-shapedhardneg notexhaustivelyannotated; semanticproxy. Roadedge subclassnotexactboundarytruth. NoTestminingfortraining.'))

if __name__=='__main__':main()
