"""Bounded A/B/C/D study with joint scene acceptance, sealed test excluded."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from tools.m5a_learning import encoder_load,stack
from tools.m5e_common import ROOT,sidecars,pairs
from datasets.m5a_training import VisualEvidenceDataset
from models.m5c_obstacle_head import VisibleObstacleHead
from models.m5b_cached_encoder import cache_rgb,cached_forward
from models.m5e_paths import depth_logits,geometry
from training.m48_evidence_loss import evidence_loss
from training.m5b_change_loss import change_loss
from training.m5c_obstacle_loss import visible_obstacle_loss
from training.m5d_existence_loss import existence_loss
from training.m5e_losses import presence_only,spatial_loss
from diagnostics.m5e_metrics import selection
from training.runtime import prepare_cuda
from tools.m47_common import freeze_sources
from simulator.recorder import atomic_json,file_hash


def predict(net,head,caches,geoms,sups):
    probabilities=[];near=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for cache,g,s in zip(caches,geoms,sups):
            logits=depth_logits(net,cache);p=logits.float().softmax(2)
            probabilities.append(head(cached_forward(net,cache,probabilities=p))['visible_obstacle'][0].float().sigmoid().cpu().numpy())
            m=s['valid16']&g['inside']&(g['rad']<10)
            if m.any():near.append(float((p*g['xy']).sum(2)[m].mean()))
    return np.array(probabilities),float(np.mean(near))


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--index',default=ROOT);pa.add_argument('--run',default='runs/m5e_learning');a=pa.parse_args();root=Path(a.run);root.mkdir(exist_ok=False)
    cfg=json.loads(Path('configs/m5e_protocol.json').read_text())['training'];quality=json.loads(Path(a.index,'m5e_supervision.json').read_text())['summary']['train']
    gates={band:quality[band]['presence_cells']>=12 and len(quality[band]['groups'])>=3 for band in ['15_20','20_25','25_35']}
    atomic_json('reports/m5e_training_release.json',dict(kind='pretraining_actual_label_support_gate',presence_gates=gates,candidate_negatives=False,scope='Presence and spatial research only; fullabsence may remain blocked',source_sha256=file_hash(__file__)))
    if not all(gates.values()):raise RuntimeError('Insufficient reliable positive support: no research training')
    prepare_cuda();torch.set_num_threads(2);net=encoder_load();start=time.perf_counter();ds={s:VisualEvidenceDataset(a.index,s) for s in ['train','val']};items={s:[d[i] for i in range(len(d))] for s,d in ds.items()};lookup={s:{r['sample_id']:i for i,r in enumerate(d.rows)} for s,d in ds.items()};pair={s:pairs(a.index,s) for s in ds};cache={s:[cache_rgb(net,stack([i],'inputs')) for i in it] for s,it in items.items()};labels={s:[stack([i],'targets') for i in it] for s,it in items.items()};refs=sidecars(a.index,ds['val'].rows,'m5e_surface_index.json')
    sup={s:[{k:torch.from_numpy(v[None]).cuda() for k,v in d.items() if k in ['depth16','valid16','actor16','presence']} for d in sidecars(a.index,d.rows,'m5e_supervision.json')] for s,d in ds.items()}
    geo={s:[geometry(net,c,d['depth16']) for c,d in zip(cache[s],sup[s])] for s in ds}
    source=freeze_sources(root/'source',[str(p) for folder in ['tools','models','training','diagnostics','datasets','configs'] for p in Path(folder).glob('*m5e*') if p.is_file()]);records=[];precompute=time.perf_counter()-start
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);groups=sorted({r['group'] for r,p in pair['train']});schedule=[]
        while len(schedule)<cfg['steps']:
            for g in rng.permutation(groups):schedule.append(int(rng.choice([i for i,(r,p) in enumerate(pair['train']) if r['group']==g])))
        schedule=schedule[:cfg['steps']];np.save(root/f'm5e_schedule_{seed}.npy',schedule)
        for name in cfg['models']:
            net=encoder_load();torch.manual_seed(seed);head=VisibleObstacleHead().cuda();params=[dict(params=head.parameters(),lr=.001),dict(params=net.context_head.parameters(),lr=.0003)]
            for v in net.context_head.parameters():v.requires_grad_(True)
            if name=='D_depth':
                for v in net.depth_head.parameters():v.requires_grad_(True)
                params.append(dict(params=net.depth_head.parameters(),lr=1e-5))
            opt=torch.optim.Adam(params);bestkey=None;history=[];all_candidates=[];finite=True;frozen=True;maxgrad=0.;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for step in range(cfg['steps']+1):
                if step%cfg['validate_every']==0:
                    prob,near=predict(net,head,cache['val'],geo['val'],sup['val'])
                    if step==0:baseline_near=near
                    selected,candidates=selection(prob,ds['val'].rows,refs,pair['val'],lookup['val'],cfg['thresholds'],near/baseline_near)
                    selected=dict(selected,step=step);key=(selected['worst_violation'],-selected['pareto_vector'][0],step)
                    all_candidates.extend(dict(c,step=step) for c in candidates);history.append(dict(step=step,validation=selected,near_spatial_error=near))
                    if bestkey is None or key<bestkey:
                        bestkey=key;best=selected;torch.save(dict(head=head.state_dict(),context=net.context_head.state_dict(),depth=net.depth_head.state_dict(),seed=seed,model=name,stride=16,selection=best),root/f'm5e_{name}_{seed}.pt')
                    print(name,seed,step,selected['worst_violation'],selected['pareto_vector'],flush=True)
                if step==cfg['steps']:break
                r,p=pair['train'][schedule[step]];ids=[lookup['train'][r[k]] for k in ['a','b']];opt.zero_grad(set_to_none=True);outputs=[];spl=[]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    for i in ids:
                        c=cache['train'][i];g=geo['train'][i];s=sup['train'][i];l=depth_logits(net,c);outputs.append(head(cached_forward(net,c,probabilities=l.float().softmax(2))));spl.append(spatial_loss(l,g['xy'],g['same_layer'],s['valid16']&g['inside'],g['rad']))
                out={k:torch.cat([o[k] for o in outputs]) for k in outputs[0]};lab={k:torch.cat([labels['train'][i][k] for i in ids]) for k in labels['train'][ids[0]]};base,parts=evidence_loss(out,lab);old,_=change_loss(out,{k:torch.from_numpy(p['legacy_'+k]).cuda() for k in ['appeared','vanished','stable']});direct,_=visible_obstacle_loss(out,lab);ex,_=existence_loss(out,{k:torch.from_numpy(p[k]).cuda() for k in ['appeared','vanished','stable']},stable=True);pos=presence_only(out['visible_obstacle'],torch.cat([sup['train'][i]['presence'] for i in ids]));sp=torch.stack(spl).mean();loss=base+.2*old+direct/3+ex
                if name!='A_base':loss=loss+.5*pos
                if name in ['C_spatial_frozen','D_depth']:loss=loss+.1*sp
                loss.backward();gr=[v.grad for v in list(net.parameters())+list(head.parameters()) if v.grad is not None];finite &= bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gr);maxgrad=max(maxgrad,max(float(g.abs().max()) for g in gr));frozen &= all(v.grad is None for n,v in net.named_parameters() if not n.startswith('context_head.') and not(name=='D_depth' and n.startswith('depth_head.')))
                if not finite:raise FloatingPointError('Stop nonfinite training')
                opt.step()
                if step%25==0:history.append(dict(step=step,total=float(loss.detach()),evidence=float(base.detach()),direct=float(direct.detach()),existence=float(ex.detach()),presence=float(pos.detach()),spatial=float(sp.detach())))
            torch.cuda.synchronize();ck=root/f'm5e_{name}_{seed}.pt';records.append(dict(name=name,seed=seed,stride=16,steps=cfg['steps'],selected=best,history=history,pareto_candidates=all_candidates,finite_gradients=finite,frozen_gradients_absent=frozen,max_gradient=maxgrad,elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,trainable_parameters=sum(v.numel() for v in list(net.parameters())+list(head.parameters()) if v.requires_grad),checkpoint=str(ck),sha256=file_hash(ck),schedule_sha256=file_hash(root/f'm5e_schedule_{seed}.npy')));atomic_json(root/'m5e_progress.json',dict(records=records))
    atomic_json('reports/m5e_model_ablation.json',dict(kind='actual_RTX4090_200step_research',records=records,precompute_s=precompute,sources=source,protocol_sha256=file_hash('configs/m5e_protocol.json'),gpu=torch.cuda.get_device_name(),test_opened=False,limits='Cspatial frozen is negativecontrol: adds constant loss noDepthgradient; D changes trainableDepth with sameobjective. Cache features only, originalimageBackbone frozen; spatial targets noCandidateabsence. Validation candidates may all be infeasible; never declarepassed from selectedbest.'))


if __name__=='__main__':main()
