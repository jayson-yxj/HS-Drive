"""Fixed ROI diagnostics: new endpoint coverage vs repeated mass, matched FPR/Recall."""
import argparse,json
from pathlib import Path
import numpy as np,torch,cv2
from tools.m5l_common import rows,cache_index,write,asset
from tools.m5k_alignment import sample_gt
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5k_geometry import resample_probability,sampling_masks
from models.m5f_lift import lift
from tools.m5f_factorization import reference
from tools.m5j_metrics import evaluate,aggregate
from training.runtime import prepare_cuda
from simulator.recorder import file_hash

def region_uniform(roi):
 out=torch.zeros_like(roi)
 for b in range(len(roi)):
  region=np.zeros(roi[b].shape,bool)
  for c in range(roi.shape[1]):
   im=roi[b,c].cpu().numpy().astype(np.uint8);n,_,stats,_=cv2.connectedComponentsWithStats(im,8)
   for x,y,w,h,_ in stats[1:]:region[c,y:y+h,x:x+w]=True
  candidates=np.flatnonzero(region);budget=int(roi[b].sum());idx=np.floor((np.arange(budget)+.5)*len(candidates)/max(budget,1)).astype(int);out[b].view(-1)[torch.tensor(candidates[idx],device=roi.device)]=True
 assert int(out.sum())==int(roi.sum())
 return out

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();prepare_cuda();torch.set_num_threads(2);net=encoder_load();cache=cache_index(a.split);records=[];attribution=[];thresholds=json.loads(Path('configs/m5l_protocol.json').read_text())['adaptive']['thresholds']
 for row in rows(a.split):
  x=inputs(row);d=pixel_label(row);c=cache_p8(net,x);pr4=resample_probability(c['probs'],(96,160));g4=sample_gt(d,4);g8=sample_gt(d,8);truth,valid,actors=reference(d,x,row,row['camera_names']);am=[r['mask']for r in actors];z=np.load(cache[row['sample_id']]);K=c['K'];T=c['T'];hw=c['image_hw']
  oraclebase=lift(net.splat,g8['target'].float(),c['probs'],K,T,hw,depth=g8['depth'],valid=g8['valid'])['foreground_mass'][0].cpu().numpy()>0
  for seed in [101,102]:
   q=torch.tensor(z[f'pixel_{seed}'][None].astype(np.float32),device='cuda');q8=q[:,:,4::8,4::8];q4=q[:,:,2::4,2::4];roi,uniform=sampling_masks(q4);region=region_uniform(roi);base=lift(net.splat,q8,c['probs'],K,T,hw)['foreground_mass'][0].cpu().numpy();bs=-np.expm1(-base)
   for method,sel in [('adaptive',roi),('uniform',uniform),('region_uniform',region)]:
    extra=lift(net.splat,q4*sel,pr4,K,T,hw)['foreground_mass'][0].cpu().numpy();score=-np.expm1(-(base+extra));oracleextra=lift(net.splat,g4['target'].float()*sel,pr4,K,T,hw,depth=g4['depth'],valid=g4['valid'])['foreground_mass'][0].cpu().numpy()>0;new=oracleextra&~oraclebase&truth&valid;existing=(base>0)&(bs<.2)&(score>=.2)&valid;target=truth&valid
    attribution.append(dict(group=row['group'],sample_id=row['sample_id'],seed=seed,method=method,extra_rays=int(sel.sum()),oracle_base_target_cells=int((oraclebase&target).sum()),new_gt_endpoint_cells=int(new.sum()),repeat_endpoint_cells=int((oracleextra&oraclebase&target).sum()),learned_new_support_cells=int(((base==0)&(extra>0)&target).sum()),existing_mass_crossing_target=int((existing&target).sum()),existing_mass_crossing_background=int((existing&~truth).sum()),new_endpoint_and_crossing=int((new&existing).sum()),new_endpoint_not_crossing=int((new&~existing).sum()),total_target_cells=int(target.sum()),nonexclusive=True))
    for t in thresholds:records.append(dict(seed=seed,method=method,threshold=t,group=row['group'],sample_id=row['sample_id'],metrics=evaluate(score,truth,valid,am,t)))
  print(row['sample_id'],flush=True)
 raw=f'runs/m5l_adaptive_{a.split}.json';write(raw,dict(records=records,attribution=attribution));agg=[dict(seed=s,method=m,threshold=t,**aggregate([r['metrics']for r in records if r['seed']==s and r['method']==m and r['threshold']==t]))for s in [101,102]for m in ['adaptive','uniform','region_uniform']for t in thresholds]
 if a.split=='val':
  locked={}
  for seed in [101,102]:
   base=next(r for r in agg if r['seed']==seed and r['method']=='uniform'and r['threshold']==.2)
   for method in ['adaptive','region_uniform','uniform']:
    rr=[r for r in agg if r['seed']==seed and r['method']==method];fpr=[r for r in rr if r['fpr']<=base['fpr']];rec=[r for r in rr if r['recall1']>=base['recall1']];locked[f'{seed}_{method}']=dict(same_fpr=max(fpr,key=lambda r:(r['recall1'],r['precision'],r['threshold']))['threshold']if fpr else None,same_recall=max(rec,key=lambda r:(r['precision'],r['recall1'],r['threshold']))['threshold']if rec else None)
  write('reports/m5l_adaptive_lock.json',dict(kind='validation_only_pretest',workpoints=locked))
 lock=json.loads(Path('reports/m5l_adaptive_lock.json').read_text());matched=[]
 for key,v in lock['workpoints'].items():
  seed,method=key.split('_',1)
  for scope,t in v.items():
   r=next((r for r in agg if r['seed']==int(seed)and r['method']==method and r['threshold']==t),None);matched.append(dict(key=key,scope=scope,metrics=r))
 at=[dict(seed=s,method=m,**{k:sum(r[k]for r in attribution if r['seed']==s and r['method']==m)for k in ['extra_rays','oracle_base_target_cells','new_gt_endpoint_cells','repeat_endpoint_cells','learned_new_support_cells','existing_mass_crossing_target','existing_mass_crossing_background','new_endpoint_and_crossing','new_endpoint_not_crossing','total_target_cells']})for s in [101,102]for m in ['adaptive','uniform','region_uniform']]
 write(f'reports/m5l_adaptive_{a.split}.json',dict(kind='actualGPU_fixedP8interpolatedS4',attribution=at,matched=matched,curve=agg,raw=asset(raw),source_sha256=file_hash(__file__),limits='AnewOracleendpointcoverage andBlearnedthresholdcrossing overlap, notexclusivefractions. Softmaxdense positive tails mean almostallnewmass lands onalready-supported cells. Sameforegroundregion bboxuniform differsGT-free fineq mask; exactsamebudget. No newmassnormalization. FixedVal thresholds transported unchanged; TestFPR neednot equal.'))
if __name__=='__main__':main()
