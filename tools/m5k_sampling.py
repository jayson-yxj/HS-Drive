"""Matched-budget adaptive ray diagnosis, no S4 Depth network and no GT ROI."""
import argparse,json,time
from pathlib import Path
import torch,numpy as np
from tools.m5k_common import rows,write,asset,summary
from tools.m5k_alignment import sample_gt
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb
from models.m5k_geometry import resample_probability,sampling_masks
from models.m5f_lift import lift
from tools.m5f_factorization import reference
from tools.m5j_metrics import evaluate,aggregate
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();prepare_cuda();torch.set_num_threads(2);net=encoder_load();dest=Path('runs/m5k_sampling_'+a.split);dest.mkdir(exist_ok=False);ix=json.loads(Path(f'runs/m5k_alignment_{a.split}/raw.json').read_text());cache={r['sample_id']:r['path']for r in ix['index']};records=[]
 for row in rows(a.split):
  x=inputs(row);d=pixel_label(row);c=cache_p8(net,x);s16=cache_rgb(net,x);p8s={'original':c['probs'],'aligned':resample_probability(s16['probs'],(48,80))};gt8=sample_gt(d,8);gt4=sample_gt(d,4);truth,valid,actors=reference(d,x,row,row['camera_names']);am=[v['mask']for v in actors];saved={}
  with np.load(cache[row['sample_id']])as z:qs={s:torch.from_numpy(z[f'pixel_{s}'][None].astype(np.float32)).cuda()for s in [101,102]}
  for seed,q in qs.items():
   q8=q[:,:,4::8,4::8];q4=q[:,:,2::4,2::4];roi,uniform=sampling_masks(q4);basecount=q8.numel();extra=int(roi.sum())
   for depmethod in ['oracle','oracle2d','original','aligned']:
    pr8=p8s.get(depmethod,c['probs']);pr4=resample_probability(pr8,(96,160));oracle=depmethod in ['oracle','oracle2d']
    def project(q,p,g,selection=None):
     # Selection is RGB-only. Oracle strictvalid used only in diagnostic arm.
     q=g['target'].float() if depmethod=='oracle2d' else q
     return lift(net.splat,q if selection is None else q*selection,p,c['K'],c['T'],c['image_hw'],depth=g['depth']if oracle else None,valid=g['valid']if oracle else None)
    for sampling,sel in [('p8',None),('full_s4',None),('adaptive',roi),('uniform',uniform)]:
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
     if sampling=='p8':out=project(q8,pr8,gt8);mass=out['foreground_mass']
     elif sampling=='full_s4':out=project(q4,pr4,gt4);mass=out['foreground_mass']
     else:
      base=project(q8,pr8,gt8);out=project(q4,pr4,gt4,sel);mass=base['foreground_mass']+out['foreground_mass']
     score=-torch.expm1(-mass);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000;peak=torch.cuda.max_memory_allocated()/2**20;score=score[0].cpu().numpy();m=evaluate(score,truth,valid,am,.2);extra_target=int((sel&gt4['target']&gt4['valid']).sum())if sel is not None else None
     records.append(dict(group=row['group'],sample_id=row['sample_id'],configuration=row['configuration'],seed=seed,depth=depmethod,sampling=sampling,base_rays=0 if sampling=='full_s4'else basecount,extra_rays=q4.numel()if sampling=='full_s4'else extra if sel is not None else 0,selected_s4_target_rays=extra_target,all_s4_target_rays=int((gt4['target']&gt4['valid']).sum()),projection_ms=ms,peak_mib=peak,metrics=m,background_mass=float(mass[0][torch.from_numpy(valid&~truth).cuda()].sum()),unknown_mass=float(mass[0][torch.from_numpy(~valid).cuda()].sum()),multi_height_columns=int(((score>=.2).sum(0)>1).sum())))
     if depmethod in ['oracle','original']:saved[f'{seed}_{depmethod}_{sampling}']=score.astype(np.float16)
  path=dest/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,**saved);print(row['sample_id'],flush=True)
 raw=dest/'raw.json';write(raw,dict(records=records));agg=[]
 for seed in [101,102]:
  for dep in ['oracle','oracle2d','original','aligned']:
   for sampling in ['p8','full_s4','adaptive','uniform']:
    rr=[r for r in records if r['seed']==seed and r['depth']==dep and r['sampling']==sampling];agg.append(dict(seed=seed,depth=dep,sampling=sampling,**aggregate([r['metrics']for r in rr]),base_rays=summary([r['base_rays']for r in rr]),extra_rays=summary([r['extra_rays']for r in rr]),projection_ms=summary([r['projection_ms']for r in rr]),peak_mib=max(r['peak_mib']for r in rr),background_mass=sum(r['background_mass']for r in rr),unknown_mass=sum(r['unknown_mass']for r in rr),multi_height_columns=sum(r['multi_height_columns']for r in rr)))
 write(f'reports/m5k_sampling_{a.split}.json',dict(kind='actual_RTX4090_offline_projection',records=agg,raw=asset(raw),source_sha256=file_hash(__file__),limits='FullS4 and extras use interpolated P8 probability in learned arms, NOT trainedS4 network. Oracle retains Mobileforeground. Alltargetdenominator includesROImisses. Adaptive zeroes nonselected rays but dense kernel still computes geometry: measuredcurrentimplementationcost NOT sparse speed claim. Heightmix counts columns with>=2positive layers, not all are errors. ROI/uniform exactcountperframe/seed. Fixed.2 no retune.'))
if __name__=='__main__':main()
