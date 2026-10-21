"""Matched two-seed short PixelHead studies, frozen old Depth and Splat."""
import json,time
from pathlib import Path
import numpy as np
import torch
from tools.m5a_learning import encoder_load
from models.m5f_pixel import PixelHead,pyramid
from training.m5f_loss import pixel_loss
from diagnostics.m5f_metrics import histogram,select
from training.runtime import prepare_cuda
from tools.m5f_dataset import ROOT
from simulator.recorder import atomic_json,file_hash
from tools.m47_common import freeze_sources


def main():
    out=Path('runs/m5f_learning');out.mkdir(exist_ok=False);cfg=json.loads(Path('configs/m5f_protocol.json').read_text());prepare_cuda();torch.set_num_threads(2);rows=json.loads(Path(ROOT,'index.json').read_text())['records'];refs={r['sample_id']:r for r in json.loads(Path('runs/m5f_cache/index.json').read_text())['records']};data=[]
    for row in rows:
        c=torch.load(refs[row['sample_id']]['path'],weights_only=True);c={k:[v.cuda() for v in x] if k=='levels' else {s:v.cuda() for s,v in x.items()} for k,x in c.items()}
        with np.load(row['pixel_label']) as z:t=torch.from_numpy(z['target']).cuda();v=torch.from_numpy(z['valid']).cuda()
        data.append(dict(cache=c,target=t,valid=v))
    source=freeze_sources(out/'source',[str(p) for base in ['datasets','models','training','diagnostics','tools','configs'] for p in Path(base).glob('*m5f*') if p.is_file()]);results=[]
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);groups=sorted({r['group'] for r in rows if r['split']=='train'});schedule=[]
        while len(schedule)<cfg['steps']:
            for g in rng.permutation(groups):
                ix=int(rng.choice([i for i,r in enumerate(rows) if r['group']==g]));positive=[j for j in range(6) if data[ix]['target'][j].any()];cam=int(rng.choice(positive)) if positive and rng.random()<.5 else int(rng.integers(6));schedule.append((ix,cam))
        schedule=schedule[:cfg['steps']];atomic_json(out/f'm5f_schedule_{seed}.json',dict(schedule=schedule))
        for name in cfg['models']:
            torch.manual_seed(seed);net=encoder_load();head=PixelHead().cuda();stride=16 if name.startswith('S16') else 8;fpn=name=='P8_fpn';params=[dict(params=head.parameters(),lr=cfg['head_lr'])]
            if fpn:
                for mod in [net.encoder.lateral,net.encoder.output]:
                    for p in mod.parameters():p.requires_grad_(True)
                params.append(dict(params=list(net.encoder.lateral.parameters())+list(net.encoder.output.parameters()),lr=cfg['fpn_lr']))
            opt=torch.optim.Adam(params);history=[];bestkey=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats();finite=True;maxgrad=0.
            def feature(i,cam=None):
                c=data[i]['cache'];sl=slice(None) if cam is None else slice(cam,cam+1)
                return pyramid(net.encoder,[x[sl] for x in c['levels']],stride) if fpn else c['features'][str(stride)][sl]
            for step in range(cfg['steps']+1):
                if step%cfg['validate_every']==0:
                    groupshist={};emptyhist={}
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        for i,row in enumerate(rows):
                            if row['split']!='val':continue
                            prob=head(feature(i)).float().sigmoid().cpu().numpy();h=histogram(prob,data[i]['target'].cpu().numpy(),data[i]['valid'].cpu().numpy());g=row['group'];groupshist[g]=groupshist.get(g,np.zeros((2,256),np.int64))+h
                            emptyhist.setdefault(g,np.zeros((2,256),np.int64))
                            if row['configuration'] in [0,15]:emptyhist[g]+=h
                    best,curve=select(groupshist,emptyhist,[i/10 for i in range(1,10)]);best=dict(best,step=step);key=(best['violation'],-best['macro_iou'],step);history.append(dict(step=step,validation=best,threshold_curve=curve))
                    if bestkey is None or key<bestkey:
                        bestkey=key;selected=best;torch.save(dict(head=head.state_dict(),lateral=net.encoder.lateral.state_dict(),output=net.encoder.output.state_dict(),stride=stride,name=name,seed=seed,selection=best),out/f'm5f_{name}_{seed}.pt')
                    print(name,seed,step,best['violation'],best['macro_iou'],flush=True)
                if step==cfg['steps']:break
                i,cam=schedule[step];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=head(feature(i,cam));loss=pixel_loss(logits,data[i]['target'][cam:cam+1],data[i]['valid'][cam:cam+1])
                loss.backward();gr=[p.grad for group in params for p in group['params'] if p.grad is not None] if False else [p.grad for p in list(net.parameters())+list(head.parameters()) if p.grad is not None];finite &= bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gr);maxgrad=max(maxgrad,max(float(g.abs().max()) for g in gr))
                if not finite:raise FloatingPointError('stop nonfinite')
                opt.step()
                if step%25==0:history.append(dict(step=step,loss=float(loss.detach())))
            ck=out/f'm5f_{name}_{seed}.pt';results.append(dict(name=name,seed=seed,stride=stride,checkpoint=str(ck),sha256=file_hash(ck),selected=selected,history=history,steps=cfg['steps'],elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,parameters=sum(p.numel() for p in list(net.parameters())+list(head.parameters()) if p.requires_grad),finite_gradients=finite,max_gradient=maxgrad,depth_frozen=all(p.grad is None for p in net.depth_head.parameters()),schedule_sha256=file_hash(out/f'm5f_schedule_{seed}.json')));atomic_json(out/'progress.json',dict(records=results))
    atomic_json('reports/m5f_image_foreground_training.json',dict(kind='actual_RTX4090_cached_RGB_backbone_pixel_training',records=results,sources=source,protocol_sha256=file_hash('configs/m5f_protocol.json'),limits='FPNcopy only affects foregroundbranch; oldDepth computed fromoriginalfrozenencoder. Allmodels same400steps butselectedstepscanvary. No pairloss/noGTforward. GPUmemoryincludesresidentfeaturecache.'))

if __name__=='__main__':main()
