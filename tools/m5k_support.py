"""Actual renderer support audit, with unchanged strict Actor-depth masks."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from tools.m5k_common import rows,write,asset
from tools.m5k_alignment import sample_gt
from tools.m5f_common import inputs,pixel_label
from models.m5e_paths import geometry
from tools.m5a_learning import encoder_load
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();prepare_cuda();torch.set_num_threads(2);net=encoder_load();rr=[];views=[]
 for row in rows(a.split):
  x=inputs(row);d=pixel_label(row);c=dict(K=x['intrinsics'],T=x['cam_to_ego'],image_hw=(384,640))
  for ci,cam in enumerate(row['camera_names']):
   for actor in row['actors'][ci]['actors']:views.append(dict(group=row['group'],camera=cam,sample_id=row['sample_id'],**actor))
   for klass,name in [(1,'vehicle'),(2,'person'),(3,'cone')]:
    for lo,hi in [(1.5,80),(1.5,10),(10,20),(20,35),(35,80)]:
     m=d['target'][ci]&(d['classes'][ci]==klass)&(d['depth'][ci]>=lo)&(d['depth'][ci]<hi);rr.append(dict(group=row['group'],sample_id=row['sample_id'],camera=cam,type=name,band=f'{lo}_{hi}',stride=1,native_visible=int(m.sum()),associated=int((m&(d['actor'][ci]>0)).sum())))
  for stride in [16,8,4]:
   gt=sample_gt(d,stride);g=geometry(net,c,gt['depth']);sp=(g['same_layer']&(g['xy']<=1)).any(2)
   for ci,cam in enumerate(row['camera_names']):
    for klass,name in [(1,'vehicle'),(2,'person'),(3,'cone')]:
     for lo,hi in [(1.5,80),(1.5,10),(10,20),(20,35),(35,80)]:
      m=gt['target'][0,ci]&(gt['classes'][0,ci]==klass)&(gt['depth'][0,ci]>=lo)&(gt['depth'][0,ci]<hi);actor=m&gt['actor'][0,ci];strict=actor&gt['valid'][0,ci];inside=strict&g['inside'][0,ci]
      rr.append(dict(group=row['group'],sample_id=row['sample_id'],camera=cam,type=name,band=f'{lo}_{hi}',stride=stride,native_visible=int(m.sum()),associated=int(actor.sum()),strict=int(strict.sum()),inside=int(inside.sum()),supported=int((inside&sp[0,ci]).sum())))
  print(row['sample_id'],flush=True)
 path=f'runs/m5k_support_{a.split}.json';write(path,dict(records=rr,views=views));aggregate=[]
 for stride in [1,4,8,16]:
  for typ in ['vehicle','person','cone']:
   for band in ['1.5_80','1.5_10','10_20','20_35','35_80']:
    selected=[r for r in rr if r['stride']==stride and r['type']==typ and r['band']==band];keys=['native_visible','associated']+(['strict','inside','supported']if stride>1 else[])
    aggregate.append(dict(stride=stride,type=typ,band=band,**{k:sum(r[k]for r in selected)for k in keys},positions=len({r['group']for r in selected if r.get('inside',r['associated'])>0})))
 far=[r for r in rr if r['stride']==8 and r['band']=='20_35'];n=sum(r['inside']for r in far);groups={r['group']for r in far if r['inside']};types={t:sum(r['inside']for r in far if r['type']==t)for t in ['vehicle','person','cone']};passed=n>=100 and len(groups)>=4 and sum(v>=10 for v in types.values())>=2
 bins={}
 for lo,hi in [(1,32),(33,128),(129,512),(513,10000000)]:bins[f'{lo}_{hi}']=sum(lo<=v.get('pixels',v.get('visible_pixels',-1))<=hi for v in views)
 write(f'reports/m5k_support_{a.split}.json',dict(kind='actual_CARLA_recorded_offline_GT_support',split=a.split,raw=asset(path),aggregate=aggregate,eligibility=dict(passed=passed,strict_actor_inBEV_20_35=n,positions=len(groups),by_class=types,minimum=dict(rays=100,positions=4,classes_at_least10=2)),actor_view_size=bins,source_sha256=file_hash(__file__),limits='Strict mask unchanged; ray counts correlated; nominal placement not support. S4 hypothetical sampling of renderer, no network. Native visible means reliable labeled interior pixels, not full unfiltered instance silhouette. Reference same renderer not independentMesh.'))
if __name__=='__main__':main()
