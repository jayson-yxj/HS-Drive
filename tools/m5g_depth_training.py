"""Matched frozen-visual P8 DepthHead CE versus CE+spatial probability."""
import json,time
from pathlib import Path
import torch,numpy as np
from tools.m5a_learning import encoder_load
from tools.m5g_common import rows
from datasets.labels import depth_bin_edges
from training.m5g_losses import stratified_depth_loss
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('runs/m5g_depth');out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);rr=rows();idx=json.loads(Path('runs/m5g_depth_cache/index.json').read_text())['records'];data=[{k:v.cuda()for k,v in torch.load(r['path'],weights_only=True).items()}for r in idx];edges=torch.tensor(depth_bin_edges(),device='cuda',dtype=torch.float32);results=[];net=encoder_load();centers=net.splat.depth_centers
    def evaluate(head):
        scenes={};near_error=near_count=0;losses=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,row in enumerate(rr):
                if row['split']!='val':continue
                d=data[i];log=head(d['rgb_conditioned']);prob=log.float().softmax(1);exp=(prob*centers[None,:,None,None]).sum(1);m=d['valid']&(d['depth']<10);near_error+=float((exp-d['depth']).abs()[m].sum());near_count+=int(m.sum());losses.append(float(stratified_depth_loss(log,d['depth'],d['valid'],d['target'],edges)));sp=(prob*d['spatial']).sum(1);mask=d['valid']&d['target']&(d['depth']>=10)&(d['depth']<35)&d['spatial'].any(1);n=int(mask.sum());s=scenes.setdefault(row['group'],[0.,0]);s[0]+=float(sp[mask].sum());s[1]+=n
        return dict(near_mae=near_error/max(near_count,1),near_n=near_count,ce_loss=float(np.mean(losses)),target_spatial_macro=float(np.mean([s[0]/s[1]for s in scenes.values()if s[1]])),by_scene={k:dict(sum=v[0],n=v[1],mass=v[0]/v[1]if v[1]else None)for k,v in scenes.items()})
    original=evaluate(net.depth_head)
    for seed in [101,102]:
        rng=np.random.default_rng(seed);groups=sorted({r['group']for r in rr if r['split']=='train'});schedule=[]
        while len(schedule)<1200:
            for g in rng.permutation(groups):
                i=int(rng.choice([i for i,r in enumerate(rr)if r['group']==g]));cams=[c for c in range(6)if data[i]['target'][c].any()];c=int(rng.choice(cams))if cams and rng.random()<.5 else int(rng.integers(6));schedule.append((i,c))
        schedule=schedule[:1200]
        for mode in ['ce','ce_spatial']:
            torch.manual_seed(seed);net=encoder_load();head=net.depth_head
            for p in head.parameters():p.requires_grad_(True)
            opt=torch.optim.Adam(head.parameters(),lr=.0002);history=[];best=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats();maxgrad=0
            for step in range(1201):
                if step%200==0:
                    ev=evaluate(head);ok=ev['near_mae']<=1.05*original['near_mae'];key=(not ok,-ev['target_spatial_macro'],ev['near_mae'],step);history.append(dict(step=step,validation=ev,near_gate=ok));path=out/f'm5g_{mode}_{seed}_step{step}.pt';torch.save(dict(depth=head.state_dict(),step=step,seed=seed,mode=mode),path)
                    if best is None or key<best:best=key;selected=dict(checkpoint=str(path),sha256=file_hash(path),step=step,validation=ev,near_gate=ok)
                    print(mode,seed,step,ev,flush=True)
                if step==1200:break
                i,c=schedule[step];d=data[i];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):log=head(d['rgb_conditioned'][c:c+1]);loss=stratified_depth_loss(log,d['depth'][c:c+1],d['valid'][c:c+1],d['target'][c:c+1],edges,d['spatial'][c:c+1]if mode=='ce_spatial'else None)
                loss.backward();gr=[p.grad for p in head.parameters()if p.grad is not None]
                if not torch.isfinite(loss)or any(not torch.isfinite(g).all()for g in gr):raise FloatingPointError('depthnonfinite')
                maxgrad=max(maxgrad,max(float(g.abs().max())for g in gr));opt.step()
                if step%100==0:history.append(dict(step=step,train_loss=float(loss.detach())))
            results.append(dict(name=mode,seed=seed,selected=selected,history=history,steps=1200,schedule=schedule,elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,parameters=sum(p.numel()for p in head.parameters()),finite=True,max_gradient=maxgrad));atomic_json(out/'progress.json',dict(records=results))
    atomic_json('reports/m5g_depth_training.json',dict(kind='actual_RTX4090_fixed_RGB_P8_DepthHead_training',original_validation=original,records=results,source_sha256=file_hash(__file__),protocol_sha256=file_hash('configs/m5g_protocol.json'),limits='OnlyDepthHeadtrains; originalP8inputfrozen. CEandspatial sameorder/budget. GTmaskonlyloss. Neargate5percent; steps0mayselectedifnewmodelsdegrade. Head notdefault.'))

if __name__=='__main__':main()
