"""Sealed support disclosed without model/threshold adaptation."""
import numpy as np,torch
from tools.m5l_common import rows,write
from tools.m5k_alignment import sample_gt
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5e_paths import geometry
from training.runtime import prepare_cuda

@torch.no_grad()
def main():
 prepare_cuda();torch.set_num_threads(2);net=encoder_load();records=[]
 for row in rows('test'):
  x=inputs(row);d=pixel_label(row);c=dict(K=x['intrinsics'],T=x['cam_to_ego'],image_hw=(384,640))
  for stride in [8,4]:
   gt=sample_gt(d,stride);g=geometry(net,c,gt['depth'])
   for cls,name in [(1,'vehicle'),(2,'person'),(3,'cone')]:
    for band,lo,hi in [('all',1.5,80),('20_35',20,35)]:
     m=gt['actor']&gt['valid']&(gt['classes']==cls)&(gt['depth']>=lo)&(gt['depth']<hi)&g['inside'];records.append(dict(group=row['group'],stride=stride,type=name,band=band,n=int(m.sum()),supported=int((m&(g['same_layer']&(g['xy']<=1)).any(2)).sum())))
  print(row['sample_id'],flush=True)
 out=[]
 for s in [8,4]:
  for typ in ['vehicle','person','cone']:
   for band in ['all','20_35']:
    r=[r for r in records if r['stride']==s and r['type']==typ and r['band']==band];out.append(dict(stride=s,type=typ,band=band,n=sum(v['n']for v in r),supported=sum(v['supported']for v in r),positions=len({v['group']for v in r if v['n']})))
 write('reports/m5l_sealed_support.json',dict(kind='actualGPU_postfreeze_support_only',records=out,limits='Newtest support notusedtochange trainingdecision. OpticalZband, strictlabelassociation GTinsideBEV. Counts correlated, zero classes unevaluated.'))
if __name__=='__main__':main()
