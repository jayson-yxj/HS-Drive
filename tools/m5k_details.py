"""Read-only event/Actor detail; fixed threshold and no new model selection."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.m5k_common import rows,write,asset
from tools.m5j_metrics import distance_to,aggregate
from simulator.recorder import file_hash
from tools.m5f_common import inputs,pixel_label
from tools.m5f_factorization import reference

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();ix=json.loads(Path(f'runs/m5k_alignment_{a.split}/raw.json').read_text());byid={r['sample_id']:r['path']for r in ix['index']};rs=rows(a.split);records=[];events=[];actors=[]
 for group in sorted({r['group']for r in rs}):
  rr=sorted([r for r in rs if r['group']==group],key=lambda r:r['configuration']);last=np.load(byid[rr[-1]['sample_id']]);first=np.load(byid[rr[0]['sample_id']]);methods=[k for k in last.files if k not in ['target','valid','actor_masks']and not k.startswith('pixel')]
  for row in rr:
   _,_,meta=reference(pixel_label(row),inputs(row,'cpu'),row,row['camera_names']);z=np.load(byid[row['sample_id']]);truth=z['target']&z['valid'];m=truth&last['valid']&~last['target']
   for method in methods:
    score=z[method];dp=distance_to((score>=.2)&z['valid']);pred=(score>=.2)&z['valid'];dg=distance_to(truth)
    record=dict(group=group,sample_id=row['sample_id'],method=method,removed_confirmed_cells=int(m.sum()),removed_residual=int(((last[method]>=.2)&m).sum()),present_hits=int(((score>=.2)&m).sum()),probability_fall_sum=float((score.astype(np.float32)-last[method].astype(np.float32))[m].sum()),wrong_direction=int(((score<=last[method])&m).sum()))
    records.append(record)
    for index,mask in enumerate(z['actor_masks']):
     values=dp[mask];n=len(values);actors.append(dict(group=group,sample_id=row['sample_id'],method=method,actor_index=index,type_id=meta[index]['type_id'],distance_m=meta[index]['distance_m'],visible_cells=n,matched1=int((values<=1).sum()),directed_surface_distance_mean_m=float(values.mean())if n and np.isfinite(values).all()else None,no_prediction=not pred.any(),note='visible surface-to-nearest predictedsurface in sameheight, NOT Actorcenter localization'))
  for method in methods:
   for event,config in [('empty',rr[0]['configuration']),('removed',rr[-1]['configuration'])]:
    values=[r['metrics']for r in ix['bev']if r['group']==group and f"{r['method']}_{r['seed']}"==method and r['configuration']==config];events.append(dict(group=group,method=method,event=event,**aggregate(values)))
 path=f'runs/m5k_details_{a.split}.json';write(path,dict(local_removed=records,actors=actors,events=events));summary=[]
 for method in sorted({r['method']for r in records}):
  rr=[r for r in records if r['method']==method];n=sum(r['removed_confirmed_cells']for r in rr);summary.append(dict(method=method,n=n,residual_fraction=sum(r['removed_residual']for r in rr)/n if n else None,mean_probability_fall=sum(r['probability_fall_sum']for r in rr)/n if n else None,correct_direction=1-sum(r['wrong_direction']for r in rr)/n if n else None))
 write(f'reports/m5k_details_{a.split}.json',dict(kind='offline_frozen_prediction_readonly_detail',removed=summary,events=events,raw=asset(path),source_sha256=file_hash(__file__),limits='Removed only present reliabletarget AND finalconfirmednontarget. Unknown/Mixed excluded. Empty maycontainmapobjects. Quantizedfloat16storedscores smallthresholdedge effect; primarymetrics computedFP32. Directed surfaces not physicalActorcentererror.'))
if __name__=='__main__':main()
