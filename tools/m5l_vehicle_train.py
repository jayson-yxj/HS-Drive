"""Conditional 600-step two-seed same-capacity residual experiment."""
import json,time,copy
from pathlib import Path
import torch,numpy as np
from tools.m5l_common import rows,write,asset
from tools.m5k_alignment import sample_gt
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from tools.m5g_depth_data import conditioned
from models.m5e_paths import geometry
from models.m5k_depth_adapter import DepthResidual
from training.m5k_losses import losses
from datasets.labels import depth_bin_edges
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

def main():
 protocol=json.loads(Path('configs/m5l_protocol.json').read_text());assert protocol['vehicle']['steps']==600
 for split in ['train','val']:
  gate=json.loads(Path(f'reports/m5k_support_{split}.json').read_text())['eligibility'];assert gate['by_class']['vehicle']>=100 and gate['positions']>=4
 # New M5-L Vehicle-only authorization, M5-K gate unchanged.
 prepare_cuda();torch.set_num_threads(2);out=Path('runs/m5l_residual_v2');out.mkdir(exist_ok=False);net=encoder_load();data=[];rs=rows('train')+rows('val');train=[i for i,r in enumerate(rs)if r['split']=='train'];val=[i for i,r in enumerate(rs)if r['split']=='val'];edges=torch.tensor(depth_bin_edges(),device='cuda');cache={}
 for split in ['train','val']:
  ix=json.loads(Path(f'runs/m5k_alignment_{split}/raw.json').read_text());cache.update({r['sample_id']:r['path']for r in ix['index']})
 with torch.no_grad():
  for r in rs:
   x=inputs(r);d=pixel_label(r);c=cache_p8(net,x);gt=sample_gt(d,8);g=geometry(net,c,gt['depth']);f=conditioned(net,c)
   with torch.autocast('cuda',dtype=torch.bfloat16):log=net.depth_head(f.flatten(0,1)).float()
   z=np.load(cache[r['sample_id']]);data.append(dict(features=f[0].cpu(),original=log.cpu(),depth=gt['depth'][0].cpu(),valid=(gt['valid']&g['inside']&(~gt['target']|(gt['classes']==1)))[0].cpu(),actor=(gt['actor']&(gt['classes']==1))[0].cpu(),spatial=(g['same_layer']&(g['xy']<=1))[0].cpu(),q={s:torch.from_numpy(z[f'pixel_{s}'][:,4::8,4::8].astype(np.float32))[:,None]for s in [101,102]}));print('cache',r['sample_id'],flush=True)
 repidx=json.loads(Path('runs/m5h_replay/index.json').read_text())['records'];rep=[torch.load(r['path'],weights_only=True)for r in repidx if r['split']=='train'];results=[]
 def batch(i,c,seed):
  d=data[i];return {k:v[c:c+1].cuda()for k,v in d.items()if k!='q'}|dict(q=d['q'][seed][c:c+1].cuda())
 def forward(model,d):return model(d['features'],d['original'],d['q'])
 @torch.no_grad()
 def evaluate(model,seed):
  vv=[]
  for i in val:
   d={k:v.cuda()for k,v in data[i].items()if k!='q'};d['q']=data[i]['q'][seed].cuda();log=forward(model,d);ce,sp,kl=losses(log,d['original'],d['depth'],d['valid'],d['actor'],d['spatial'],edges);vv.append(float(ce+.25*sp+.1*kl))
  return float(np.mean(vv))
 for seed in [101,102]:
  rng=np.random.default_rng(seed);eligible=[(i,c)for i in train for c in range(6)if (data[i]['valid'][c]&data[i]['actor'][c]).any()];order=[]
  for step in range(600):
   i,c=eligible[int(rng.integers(len(eligible)))]if step%2==0 else(int(rng.choice(train)),int(rng.integers(6)));order.append((i,c,int(rng.integers(len(rep))),int(rng.integers(6))))
  write(out/f'order_{seed}.json',order)
  for mode in ['no_mobile','mobile']:
   torch.manual_seed(seed);model=DepthResidual(mobile_condition=mode=='mobile').cuda();opt=torch.optim.Adam(model.parameters(),lr=.001);history=[];best=None;torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
   for step in range(601):
    if step in [0,200,400,600]:
     ev=evaluate(model,seed);path=out/f'm5l_{mode}_{seed}_step{step}.pt';torch.save(dict(adapter=model.state_dict(),mode=mode,seed=seed,step=step),path);candidate=dict(step=step,validation_loss=ev,**asset(path));history.append(candidate)
     if best is None or ev<best['validation_loss']:best=candidate
     print(mode,seed,step,ev,flush=True)
    if step==600:break
    i,ci,ri,rc=order[step];d=batch(i,ci,seed);rd={k:v[rc:rc+1].cuda()for k,v in rep[ri].items()};rf=rd['rgb_conditioned']
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):rl=net.depth_head(rf).float()
    opt.zero_grad(set_to_none=True);log=forward(model,d);ce,sp,kl=losses(log,d['original'],d['depth'],d['valid'],d['actor'],d['spatial'],edges)
    rlog=model(rf,rl,torch.zeros_like(rd['depth'][:,None]));rce=torch.nn.functional.cross_entropy(rlog,(torch.bucketize(rd['depth'].contiguous(),edges,right=True)-1).clamp(0,47),reduction='none');rm=rd['valid']&(rd['depth']<10);rce=rce[rm].mean()if rm.any()else rlog.sum()*0
    oldrp=rl.softmax(1);rkl=(oldrp*(oldrp.clamp_min(1e-8).log()-rlog.log_softmax(1))).sum(1);rkl=rkl[rm].mean()if rm.any()else rlog.sum()*0
    loss=ce+.25*sp+.5*rce+.1*(kl+rkl);assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    if step%50==0:history.append(dict(step=step+1,loss=float(loss.detach()),ce=float(ce.detach()),spatial=float(sp.detach()),replay=float(rce.detach()),kl=float(kl.detach()),gradient_norm=float(grad)))
   results.append(dict(mode=mode,seed=seed,parameters=sum(p.numel()for p in model.parameters()),selected=best,history=history,elapsed_s=time.perf_counter()-t,peak_mib=torch.cuda.max_memory_allocated()/2**20,order=asset(out/f'order_{seed}.json')))
 write('reports/m5l_vehicle_training.json',dict(kind='actual_RTX4090_conditional_residual_training',records=results,source_sha256=file_hash(__file__),budget=600,limits='Vehicle-only newM5L authorization; otherforeground excludedtraining. 50%vehiclecameras50%uniform; sameB/C capacity/init/order/loss; Mobilebranch frozen. Replay S16/P8 frozenconditioned features, zeroMobile on replay (onlydistributionprotection). Val loss selection, no Test tuning; no spatialloss ablation claimed. BF16 frozenfeatures FP32 residual/geometry.'))
if __name__=='__main__':main()
