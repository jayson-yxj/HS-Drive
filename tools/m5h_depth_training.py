"""Original-head matched near replay and KL controls; no distance reweighting."""
import json,time,copy
from pathlib import Path
import torch,numpy as np
from tools.m5a_learning import encoder_load
from tools.m5g_common import rows
from datasets.labels import depth_bin_edges
from training.m5h_losses import depth_ce,near_kl
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('runs/m5h_depth');out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);cfg=json.loads(Path('configs/m5h_protocol.json').read_text())['depth'];rr=rows();idx=json.loads(Path('runs/m5g_depth_cache/index.json').read_text())['records'];data=[{k:v.cuda()for k,v in torch.load(r['path'],weights_only=True).items()}for r in idx];repidx=json.loads(Path('runs/m5h_replay/index.json').read_text())['records'];replay=[torch.load(r['path'],weights_only=True)for r in repidx];edges=torch.tensor(depth_bin_edges(),device='cuda',dtype=torch.float32);net=encoder_load();old=copy.deepcopy(net.depth_head).eval();centers=net.splat.depth_centers;results=[];trainrep=[i for i,r in enumerate(repidx)if r['split']=='train'];valrep=[i for i,r in enumerate(repidx)if r['split']=='val'and r['stride']==16]
    def forward(head,f):
        with torch.autocast('cuda',dtype=torch.bfloat16):return head(f)
    def evaluate(head):
        scenes={};near_error=near_count=total_error=total_count=bias=0
        with torch.no_grad():
            for i,row in enumerate(rr):
                if row['split']!='val':continue
                d=data[i];prob=forward(head,d['rgb_conditioned']).float().softmax(1);exp=(prob*centers[None,:,None,None]).sum(1);delta=exp-d['depth'];m=d['valid']&(d['depth']<10);near_error+=float(delta.abs()[m].sum());near_count+=int(m.sum());total_error+=float(delta.abs()[d['valid']].sum());bias+=float(delta[d['valid']].sum());total_count+=int(d['valid'].sum());sp=(prob*d['spatial']).sum(1);mask=d['valid']&d['target']&(d['depth']>=10)&(d['depth']<35)&d['spatial'].any(1);s=scenes.setdefault(row['group'],[0.,0]);s[0]+=float(sp[mask].sum());s[1]+=int(mask.sum())
            origerr=orign=0
            for i in valrep:
                d={k:v.cuda()for k,v in replay[i].items()};prob=forward(head,d['rgb_conditioned']).float().softmax(1);exp=(prob*centers[None,:,None,None]).sum(1);m=d['valid'];origerr+=float((exp-d['depth']).abs()[m].sum());orign+=int(m.sum())
        return dict(near_mae=near_error/max(near_count,1),near_n=near_count,global_mae=total_error/max(total_count,1),global_bias=bias/max(total_count,1),original_S16_val_near_mae=origerr/max(orign,1),original_S16_val_near_n=orign,target_spatial_macro=float(np.mean([s[0]/s[1]for s in scenes.values()if s[1]])))
    original=evaluate(old);print('original',original,flush=True)
    # Gradient conflict at the untouched original head on 12 training-only fixed pairs.
    probe=copy.deepcopy(old);[p.requires_grad_(True)for p in probe.parameters()];gradient=[];train=[i for i,r in enumerate(rr)if r['split']=='train']
    def vector(loss):
        g=torch.autograd.grad(loss,list(probe.parameters()),retain_graph=True,allow_unused=True);return torch.cat([x.flatten()for x in g if x is not None])
    for j,i in enumerate(train[:12]):
        d=data[i];rd={k:v.cuda()for k,v in replay[trainrep[j]].items()};log=forward(probe,d['rgb_conditioned']);rl=forward(probe,rd['rgb_conditioned']);near=depth_ce(rl,rd['depth'],rd['valid'],edges);far=depth_ce(log,d['depth'],d['valid']&(d['depth']>=10),edges);g1=vector(near);g2=vector(far);gradient.append(dict(sample_id=rr[i]['sample_id'],replay=repidx[trainrep[j]]['sample_id'],near_far_cosine=float(torch.nn.functional.cosine_similarity(g1,g2,dim=0)),near_norm=float(g1.norm()),far_norm=float(g2.norm()),old_kl_gradient_at_init=0.0))
    del probe
    for seed in [101,102]:
        rng=np.random.default_rng(seed);order=[(int(rng.choice(train)),int(rng.integers(6)),int(rng.choice(trainrep)),int(rng.integers(6)))for _ in range(cfg['steps'])]
        for mode in ['replay','replay_kl']:
            torch.manual_seed(seed);net=encoder_load();head=net.depth_head;[p.requires_grad_(True)for p in head.parameters()];opt=torch.optim.Adam(head.parameters(),lr=cfg['lr']);history=[];best=None;selected=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats();maxgrad=0
            for step in range(cfg['steps']+1):
                if step%cfg['evaluate_every']==0:
                    ev=evaluate(head);ok=ev['near_mae']<=1.05*original['near_mae'] and ev['original_S16_val_near_mae']<=1.05*original['original_S16_val_near_mae'];key=(not ok,-ev['target_spatial_macro'],ev['near_mae'],step);history.append(dict(step=step,validation=ev,near_gate=ok));path=out/f'm5h_{mode}_{seed}_step{step}.pt';torch.save(dict(depth=head.state_dict(),step=step,seed=seed,mode=mode,optimizer=opt.state_dict()),path)
                    if best is None or key<best:best=key;selected=dict(checkpoint=str(path),sha256=file_hash(path),step=step,validation=ev,near_gate=ok)
                    print(mode,seed,step,ev,ok,flush=True)
                if step==cfg['steps']:break
                i,c,ri,rc=order[step];d=data[i];rd={k:v[rc:rc+1].cuda()for k,v in replay[ri].items()};opt.zero_grad(set_to_none=True);log=forward(head,d['rgb_conditioned'][c:c+1]);rl=forward(head,rd['rgb_conditioned']);mask=d['valid'][c:c+1];dep=d['depth'][c:c+1];ce=depth_ce(log,dep,mask,edges);rep=depth_ce(rl,rd['depth'],rd['valid'],edges)
                with torch.no_grad():oldlog=forward(old,d['rgb_conditioned'][c:c+1]);oldrl=forward(old,rd['rgb_conditioned'])
                kl=.5*(near_kl(log,oldlog,mask&(dep<10))+near_kl(rl,oldrl,rd['valid']));loss=ce+cfg['replay_weight']*rep+(cfg['kl_weight']*kl if mode=='replay_kl'else 0)
                if step in [199,599,1199] and mode=='replay_kl':
                    vectors=[]
                    for term in [rep,kl,depth_ce(log,dep,mask&(dep>=10),edges)]:
                        gs=torch.autograd.grad(term,list(head.parameters()),retain_graph=True);vectors.append(torch.cat([g.flatten()for g in gs]))
                    history.append(dict(step=step+1,gradient_norms=[float(g.norm())for g in vectors],gradient_cosines={f'{a}_{b}':float(torch.nn.functional.cosine_similarity(vectors[a],vectors[b],dim=0))for a,b in [(0,1),(0,2),(1,2)]},gradient_terms=['near_GT_replay','old_near_KL','new_far_GT']))
                loss.backward();gr=[p.grad for p in head.parameters()if p.grad is not None]
                if not torch.isfinite(loss)or any(not torch.isfinite(g).all()for g in gr):raise FloatingPointError('nonfinite')
                maxgrad=max(maxgrad,max(float(g.abs().max())for g in gr));opt.step()
                if step%100==0:history.append(dict(step=step+1,new_ce=float(ce.detach()),replay_ce=float(rep.detach()),kl=float(kl.detach()),loss=float(loss.detach())))
            results.append(dict(name=mode,seed=seed,selected=selected,history=history,steps=cfg['steps'],schedule=order,elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,parameters=sum(p.numel()for p in head.parameters()),all_finite=True,max_gradient=maxgrad));atomic_json(out/'progress.json',dict(records=results))
    eligible=all(r['selected']['step']>0 and r['selected']['near_gate']and r['selected']['validation']['global_mae']<=1.05*original['global_mae']for r in results)
    atomic_json('reports/m5h_depth_preservation.json',dict(kind='actual_RTX4090_original_replay_depth_training',original_validation=original,records=results,initial_gradient_conflict=gradient,spatial_D_condition=eligible,spatial_D_status='Pending'if eligible else'NotRun_condition_failed',source_sha256=file_hash(__file__),protocol_sha256=file_hash('configs/m5h_protocol.json'),limits='P8/S16 replay matchedsupport. Original replay train ids never val. NewuniformCE differsM5G bandbalancedobjective; no singlefactorclaim vsM5G. Noheadreplacesdefault.'))
if __name__=='__main__':main()
