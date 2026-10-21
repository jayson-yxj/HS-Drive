"""Selected residual checkpoints: same rays, Mobile and real BEV as original."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from tools.m5k_common import rows,write,asset,group_depth
from tools.m5k_alignment import sample_gt,depth_records
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from tools.m5g_depth_data import conditioned
from models.m5k_depth_adapter import DepthResidual
from models.m5f_lift import lift
from tools.m5f_factorization import reference
from tools.m5j_metrics import evaluate,aggregate
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();prepare_cuda();torch.set_num_threads(2);net=encoder_load();models=[]
 for r in json.loads(Path('reports/m5k_depth_training.json').read_text())['records']:
  assert file_hash(r['selected']['path'])==r['selected']['sha256'];ck=torch.load(r['selected']['path'],weights_only=True);model=DepthResidual(mobile_condition=r['mode']=='mobile').cuda().eval();model.load_state_dict(ck['adapter']);models.append((r,model))
 ix=json.loads(Path(f'runs/m5k_alignment_{a.split}/raw.json').read_text());cache={r['sample_id']:r['path']for r in ix['index']};bev=[];records=[];dest=Path('runs/m5k_residual_eval_'+a.split);dest.mkdir(exist_ok=False)
 for row in rows(a.split):
  x=inputs(row);d=pixel_label(row);c=cache_p8(net,x);f=conditioned(net,c);gt=sample_gt(d,8);truth,valid,actors=reference(d,x,row,row['camera_names']);am=[r['mask']for r in actors]
  with torch.autocast('cuda',dtype=torch.bfloat16):log=net.depth_head(f.flatten(0,1)).float()
  z=np.load(cache[row['sample_id']]);saved={}
  for r,model in models:
   seed=r['seed'];method=r['mode']+str(seed);q=torch.from_numpy(z[f'pixel_{seed}'][None,:,4::8,4::8].astype(np.float32)).cuda();pr=model(f[0],log,q[0,:,None]).softmax(1)[None];records+=depth_records(net,c,gt,pr,method,row);out=lift(net.splat,q,pr,c['K'],c['T'],c['image_hw']);score=out['event_score'][0].cpu().numpy();bev.append(dict(method=method,seed=seed,group=row['group'],sample_id=row['sample_id'],configuration=row['configuration'],metrics=evaluate(score,truth,valid,am,.2)));saved[method]=score.astype(np.float16)
  np.savez_compressed(dest/(row['sample_id'].replace('/','_')+'.npz'),**saved);print(row['sample_id'],flush=True)
 write(dest/'raw.json',dict(depth=records,bev=bev));write(f'reports/m5k_residual_{a.split}.json',dict(kind='actual_RTX4090_selected_frozen_residual',depth=group_depth(records,['method','scope','band']),by_position_depth=group_depth([r for r in records if r['band']=='near'],['method','scope','band','group']),bev=[dict(method=m,seed=s,**aggregate([r['metrics']for r in bev if r['method']==m]))for m,s in sorted({(r['method'],r['seed'])for r in bev})],positions=[dict(method=m,seed=s,group=g,**aggregate([r['metrics']for r in bev if r['method']==m and r['group']==g]))for m,s,g in sorted({(r['method'],r['seed'],r['group'])for r in bev})],raw=asset(dest/'raw.json'),source_sha256=file_hash(__file__)))
 # Original M4 validation replay never used to optimize residual.
 if a.split=='val':
  replay=[];edges=net.splat.depth_centers[None,:,None,None]
  for r in json.loads(Path('runs/m5h_replay/index.json').read_text())['records']:
   if r['split']!='val':continue
   d={k:v.cuda()for k,v in torch.load(r['path'],weights_only=True).items()}
   with torch.autocast('cuda',dtype=torch.bfloat16):old=net.depth_head(d['rgb_conditioned']).float()
   for method,model in [('original',None)]+[(m['mode']+str(m['seed']),v)for m,v in models]:
    pred=old if model is None else model(d['rgb_conditioned'],old,torch.zeros_like(d['depth'][:,None]));delta=(pred.softmax(1)*edges).sum(1)-d['depth'];mask=d['valid'];replay.append(dict(method=method,stride=r['stride'],sample_id=r['sample_id'],n=int(mask.sum()),abs_sum=float(delta.abs()[mask].double().sum()),bias_sum=float(delta[mask].double().sum())))
  write('reports/m5k_m4_replay_validation.json',dict(kind='actual_GPU_M4_original_validation_replay',records=replay,limits='Mobile q=0 on originalcachedReplay; actualMobile nearActor performance measuredseparately onnewVal. Zeroq replay cannot standinforfullMobile path protection.'))
if __name__=='__main__':main()
