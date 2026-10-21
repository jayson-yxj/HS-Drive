"""First experiment: fixed-ray depth distribution and real Mobile BEV comparisons."""
import argparse,json,copy
from pathlib import Path
import numpy as np,torch
from tools.m5l_common import rows,write,asset,group_depth
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb
from models.m5k_geometry import resample_probability
from models.m5e_paths import geometry
from tools.m5g_depth_data import conditioned
from tools.m5f_factorization import reference
from tools.m5j_metrics import evaluate,aggregate
from models.m5f_lift import lift
from datasets.labels import depth_bin_edges
from tools.m5i_data import load
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

RANGES=[('all',1.5,80),('near',1.5,10),('10_20',10,20),('20_35',20,35),('35_80',35,80)]
def sample_gt(d,stride):
    phase=stride//2;sl=(slice(None),slice(phase,None,stride),slice(phase,None,stride));dep=d['depth'][sl];v=d['valid'][sl].copy()
    patch=np.stack([d['depth'][:,phase+dy:384:stride,phase+dx:640:stride]for dy in [-1,0,1]for dx in [-1,0,1]])
    v &= np.isfinite(patch).all(0)&((patch.max(0)-patch.min(0))<=.5)
    return {k:torch.from_numpy(np.array(vv)[None]).cuda()for k,vv in dict(depth=dep,valid=v,actor=(d['actor'][sl]>0)&d['target'][sl],target=d['target'][sl],classes=d['classes'][sl]).items()}

def depth_records(net,c,gt,pr,method,row):
    dep=gt['depth'];g=geometry(net,c,dep);centers=net.splat.depth_centers[None,None,:,None,None];edges=torch.tensor(depth_bin_edges(),device='cuda');bid=torch.bucketize(dep.contiguous(),edges,right=True)-1;ids=torch.arange(48,device='cuda')[None,None,:,None,None];exp=(pr*centers).sum(2);mode=net.splat.depth_centers[pr.argmax(2)]
    fields=dict(expected_mae=(exp-dep).abs(),mode_mae=(mode-dep).abs(),bias=exp-dep,gt_bin=(pr*(ids==bid[:,:,None])).sum(2),gt_bin_pm1=(pr*((ids-bid[:,:,None]).abs()<=1)).sum(2),inside_mass=(pr*g['bin_inside']).sum(2),outside_mass=(pr*~g['bin_inside']).sum(2))
    for radius in [.5,1,2]:fields['space'+str(radius)]=(pr*g['same_layer']*(g['xy']<=radius)).sum(2)
    records=[]
    for scope in ['all','actor','vehicle','person','cone']:
      scope_mask=torch.ones_like(dep,dtype=torch.bool)if scope=='all'else gt['actor']
      if scope in ['vehicle','person','cone']:scope_mask=scope_mask&(gt['classes']=={'vehicle':1,'person':2,'cone':3}[scope])
      for band,lo,hi in RANGES:
       # One identical strict GT-in-BEV denominator for all reported depth fields.
       m=gt['valid']&g['inside']&scope_mask&(dep>=lo)&(dep<hi)
       for ci,cam in enumerate(row['camera_names']):
        mask=m[0,ci];n=int(mask.sum())
        if n:records.append(dict(method=method,group=row['group'],sample_id=row['sample_id'],scope=scope,band=band,camera=cam,n=n,sums={k:float(value[0,ci][mask].double().sum())for k,value in fields.items()}))
    return records

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',default='historical_val');a=p.parse_args();dest=Path('runs/m5l_baseline_'+a.split);dest.mkdir(exist_ok=False)
 if a.split=='test':
    freeze=json.loads(Path('reports/m5l_pretest_freeze.json').read_text())
    for path,sha in freeze['sources'].items():assert file_hash(path)==sha,path
 prepare_cuda();torch.set_num_threads(2);net=encoder_load();original=copy.deepcopy(net.depth_head.state_dict());cfg=json.loads(Path('configs/m5j_models.json').read_text());pixel={r['seed']:load(r)[0]for r in cfg['pixel']if r['name']=='mobile_gt'}
 replay={r['seed']:torch.load(r['selected']['checkpoint'],weights_only=True)['depth']for r in json.loads(Path('reports/m5h_depth_preservation.json').read_text())['records']if r['name']=='replay'}
 records=[];bev=[];index=[]
 for row in rows(a.split):
  net.depth_head.load_state_dict(original);x=inputs(row);d=pixel_label(row);c=cache_p8(net,x);c16=cache_rgb(net,x);gt=sample_gt(d,8);gt16=sample_gt(d,16);probs=dict(original_p8=c['probs'],native16_to_p8=resample_probability(c16['probs'],(48,80)));f=conditioned(net,c)
  for seed,weight in replay.items():
   net.depth_head.load_state_dict(weight)
   with torch.autocast('cuda',dtype=torch.bfloat16):probs[f'replay{seed}_p8']=net.depth_head(f.flatten(0,1)).reshape(1,6,48,48,80).float().softmax(2)
  net.depth_head.load_state_dict(original);truth,known,actors=reference(d,x,row,row['camera_names']);am=[v['mask']for v in actors];qs={}
  for seed,model in pixel.items():
   with torch.autocast('cuda',dtype=torch.bfloat16):q=model(**x).float().sigmoid()
   qs[seed]=torch.clamp((q*256).floor(),max=255)/256
  saved=dict(target=truth,valid=known,actor_masks=np.stack(am)if am else np.zeros((0,4,160,120),bool))
  for method,pr in [*probs.items(),('native16',c16['probs']),('oracle_p8',c['probs'])]:
   native=method=='native16';cc=c16 if native else c;gg=gt16 if native else gt;stride=16 if native else 8;phase=stride//2
   if method!='oracle_p8':records+=depth_records(net,cc,gg,pr,method,row)
   for seed,q in [(0,gg['target'].float()),*[(s,v[:,:,phase::stride,phase::stride])for s,v in qs.items()]]:
    # GT valid is confined to Oracle diagnostic; RGB formal path unmasked.
    out=lift(net.splat,q,pr,cc['K'],cc['T'],cc['image_hw'],depth=gg['depth']if method=='oracle_p8'else None,valid=gg['valid']if seed==0 or method=='oracle_p8'else None)
    score=out['event_score'][0].cpu().numpy();record=dict(method=method,seed=seed,group=row['group'],sample_id=row['sample_id'],configuration=row['configuration'],metrics=evaluate(score,truth,known,am,.2));bev.append(record);saved[f'{method}_{seed}']=score.astype(np.float16)
  for seed,q in qs.items():saved[f'pixel_{seed}']=q[0].cpu().numpy().astype(np.float16)
  path=dest/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,**saved);index.append(dict(sample_id=row['sample_id'],**asset(path)))
  print(row['sample_id'],flush=True)
 write(dest/'raw.json',dict(depth=records,bev=bev,index=index));by=[dict(method=m,seed=s,**aggregate([r['metrics']for r in bev if r['method']==m and r['seed']==s]))for m,s in sorted({(r['method'],r['seed'])for r in bev})];positions=[dict(method=m,seed=s,group=g,**aggregate([r['metrics']for r in bev if r['method']==m and r['seed']==s and r['group']==g]))for m,s,g in sorted({(r['method'],r['seed'],r['group'])for r in bev})]
 write(f'reports/m5l_baseline_{a.split}.json',dict(kind='actual_RTX4090_readonly_offline',split=a.split,frames=len(index),raw=asset(dest/'raw.json'),depth=group_depth(records,['method','scope','band']),by_camera=group_depth([r for r in records if r['scope']=='actor'and r['band']=='all'],['method','camera']),bev=by,positions=positions,source_sha256=file_hash(__file__),mask='depth: all methods strict identical ray mask AND GTinBEV. Mobile formal learned path NO GTmask. Oracle interventions explicit strict mask. native16 different density separately.',geometry='OpticalZ; rounded UV8+16i vs4+8i; bilinear align_cornersFalse thenD normalization; no logit interpolation; near borders replicated',limits='conditional visible firstsurface reference not physicalMesh; unknown never negative; oracle_p8 missing probability statistics intentionally continuous not quantized'))
if __name__=='__main__':main()
