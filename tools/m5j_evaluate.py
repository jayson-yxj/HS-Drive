"""Frozen controls, full curves and validation-only matching workpoints."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.m5j_common import write
from tools.m5j_metrics import evaluate,aggregate
from diagnostics.m5f_metrics import histogram,metrics
from simulator.recorder import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='val');a=p.parse_args()
    if a.split=='test':
        from tools.m5j_freeze import verify
        verify()
    idx=json.loads(Path(f'runs/m5j_{a.split}/index.json').read_text())['records'];cfg=json.loads(Path('configs/m5j_protocol.json').read_text());ths=cfg['thresholds'];records=[];hists={};pair_data={};diags=[]
    locked=json.loads(Path('reports/m5j_selection.json').read_text())if a.split!='val'else None
    for row in idx:
        d=np.load(row['cache']);target=d['target'];valid=d['valid'];actors=d['actor_masks'];g=row['group'];histgroup={};rec={}
        for seed in [101,102]:
            for method in ['raw','gate','regularized']:
                score=d[f'{seed}_{method}'];h=histogram(score,target,valid);key=(seed,method,g);hists[key]=hists.get(key,np.zeros((2,256),np.int64))+h
                ts=ths if locked is None else sorted(set([.2]+[locked['workpoints'][f'{seed}_{method}'][k]for k in ['selected','recall_matched','precision_matched']]))
                for t in ts:records.append(dict(seed=seed,method=method,group=g,sample_id=row['sample_id'],configuration=row['configuration'],threshold=t,**evaluate(score,target,valid,actors,t)))
            for mode in ['mobile_learned','oracle_learned','mobile_oracle','oracle_oracle']:
                for method in ['raw','gate','regularized']:
                    t=.2 if method=='raw'or locked is None else locked['workpoints'][f'{seed}_{method}']['selected'];score=d[f'{seed}_diag_{mode}_{method}'];diags.append(dict(seed=seed,mode=mode,method=method,threshold=t,group=g,sample_id=row['sample_id'],**evaluate(score,target,valid,actors,t)))
        print(row['sample_id'],flush=True)
    ag=[]
    for seed in [101,102]:
        for method in ['raw','gate','regularized']:
            for t in sorted({r['threshold']for r in records if r['seed']==seed and r['method']==method}):
                rr=[r for r in records if r['seed']==seed and r['method']==method and r['threshold']==t];ag.append(dict(seed=seed,method=method,threshold=t,**aggregate(rr)))
    curves=[]
    for seed in [101,102]:
        for method in ['raw','gate','regularized']:
            hs=[h for (s,m,g),h in hists.items()if s==seed and m==method];h=sum(hs);curves.append(dict(seed=seed,method=method,histogram=h.tolist(),macro_ap=float(np.mean([metrics(v,.2)['histogram_ap']or 0 for v in hs])),curve=[dict(threshold=t/256,**metrics(h,t/256))for t in range(1,257)]))
    write(f'reports/m5j_aggregation_{a.split}.json',dict(kind='offline_actual_GPU_evidence_conditional_surface_evaluation',records=records,aggregate=ag,curves=curves,oracle_diagnostics=diags,source_sha256=file_hash(__file__),limits='Formal learned path unmasked; fourwayOracle uses matchedGTvalidP8 for allfour. Fullnative surface reference. Distancetransform sameheight many-to-onecoverage notobjectmatching. Testcurves descriptive thresholds frozenVal.'))
    if a.split!='val':return
    select(ag,curves)

def select(ag,curves):
    work={};gain=[]
    for seed in [101,102]:
        raw=next(r for r in ag if r['seed']==seed and r['method']=='raw'and r['threshold']==.2)
        for method in ['raw','gate','regularized']:
            cand=[r for r in ag if r['seed']==seed and r['method']==method];eligible=[r for r in cand if r['recall1']>=raw['recall1']-.02];best=max(eligible,key=lambda r:(r['precision'],r['recall1'],-r['threshold']))if eligible else max(cand,key=lambda r:r['recall1']);rm=min(cand,key=lambda r:(abs(r['recall1']-raw['recall1']),-(r['precision']or 0)));pm=min(cand,key=lambda r:(abs((r['precision']or 0)-raw['precision']),-r['recall1']));work[f'{seed}_{method}']=dict(selected=.2 if method=='raw'else best['threshold'],recall_matched=rm['threshold'],precision_matched=pm['threshold']);gain.append(dict(seed=seed,method=method,precision_gain=best['precision']-raw['precision'],recall1_delta=best['recall1']-raw['recall1'],actor_detection_delta=best['actor_detection']-raw['actor_detection'],passed=best['precision']>=raw['precision']+.05 and best['recall1']>=raw['recall1']-.02 and best['actor_detection']>=raw['actor_detection']-.02))
    eligible=[m for m in ['gate','regularized']if all(r['passed']for r in gain if r['method']==m)];selected=max(eligible,key=lambda m:np.mean([r['precision_gain']for r in gain if r['method']==m]))if eligible else'raw';condition=all(next(r for r in curves if r['seed']==s and r['method']=='regularized')['macro_ap']>next(r for r in curves if r['seed']==s and r['method']=='raw')['macro_ap']for s in [101,102]);write('reports/m5j_selection.json',dict(kind='validation_only_fixed_bounded_selection',workpoints=work,gains=gain,selected=selected,conditional_D=condition,source_sha256=file_hash(__file__),note='Ifnoeligible keepproductionunchanged andraw main diagnostic. NoTestmodelselection.'))
if __name__=='__main__':main()
