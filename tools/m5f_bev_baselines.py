"""Raw historical head and fixed event-score baseline; no test threshold fitting."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from diagnostics.m5f_metrics import histogram,metrics
from diagnostics.m5c_localization import matching
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_bev_baselines.json')
    if dest.exists():raise FileExistsError(dest)
    rows=json.loads(Path('runs/m5f_bev_test_features/index.json').read_text())['records'];oldstudy=json.loads(Path('reports/m5e_model_ablation.json').read_text())['records'];records=[]
    for seed in [101,102]:
        baseline=next(r for r in oldstudy if r['name']=='A_base' and r['seed']==(91 if seed==101 else 92));th=baseline['selected']['threshold']
        for mode,threshold in [('raw_M5E_A',th),('foreground_commonS16_fixed',.2)]:
            hist=np.zeros((2,256),np.int64);frames=[]
            for row in [r for r in rows if r['seed']==seed]:
                d=np.load(row['path']);score=expit(d['old_logit']) if mode=='raw_M5E_A' else -np.expm1(-d['foreground_mass']);h=histogram(score,d['target'],d['valid']);hist+=h;cut=int(threshold*256)/256;frames.append(dict(group=row['group'],configuration=row['configuration'],metrics=metrics(h,threshold),matching=matching((score>=cut)&d['valid'],d['target']&d['valid']),unknown_high_fraction=float((score[~d['valid']]>=cut).mean())))
            records.append(dict(seed=seed,mode=mode,threshold=threshold,metrics=metrics(hist,threshold),frames=frames))
    atomic_json(dest,dict(kind='offline_saved_actual_GPU_original_head_and_foreground_scores',records=records,source_sha256=file_hash(__file__),limits='RawM5EA originalvalidationthreshold; newforeground fixed.2 uncalibratedeventscore notphysicaloccupancy. CommonS16UV+Depth sameasfusion. Headcomparison notstrictsinglevariabletraining; matchedold-onlyrecalibration vsfusion maincontrolledstudy. Currentfirstsurfacevalid region only; no testreselection.'))

if __name__=='__main__':main()
