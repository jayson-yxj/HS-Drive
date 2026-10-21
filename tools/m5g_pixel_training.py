"""Budget-matched pixel ablations; no depth changes and no new-test access."""
import json,time
from pathlib import Path
import torch,numpy as np
from tools.m5g_common import rows,schedule
from tools.m5f_common import pixel_label
from tools.m5a_learning import encoder_load
from models.m5f_pixel import PixelHead,pyramid
from training.m5g_losses import foreground_loss
from diagnostics.m5f_metrics import histogram,metrics,select
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    dest=Path('runs/m5g_pixels');dest.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);cfg=json.loads(Path('configs/m5g_protocol.json').read_text());rr=rows();labels=[pixel_label(r)for r in rr];cacheidx={r['sample_id']:r for r in json.loads(Path('runs/m5f_cache/index.json').read_text())['records']};data=[]
    for i,r in enumerate(rr):
        c=torch.load(cacheidx[r['sample_id']]['path'],weights_only=True);data.append(dict(levels=[x.cuda()for x in c['levels']],feature=c['features']['8'].cuda(),target=torch.from_numpy(labels[i]['target']).cuda(),valid=torch.from_numpy(labels[i]['valid']).cuda()))
    for g in sorted({r['group']for r in rr}):
        ids=[i for i,r in enumerate(rr)if r['group']==g];union=np.any([labels[i]['target']for i in ids],axis=0)
        for i in ids:
            hard=union&labels[i]['valid']&~labels[i]['target'] if rr[i]['configuration']in[0,15]else np.zeros_like(union)
            data[i]['hard']=torch.from_numpy(hard).cuda()
    results=[]
    for mode in cfg['pixel']['models']:
        for seed in cfg['seeds']:
            torch.manual_seed(seed);net=encoder_load();head=PixelHead().cuda();params=[dict(params=list(head.parameters()),lr=cfg['pixel']['head_lr'])]
            if mode!='head_bce':
                ps=list(net.encoder.lateral.parameters())+list(net.encoder.output.parameters())
                for p in ps:p.requires_grad_(True)
                params.append(dict(params=ps,lr=cfg['pixel']['fpn_lr']))
            if mode=='high_bce':
                ps=list(net.encoder.backbone.layer3.parameters())+list(net.encoder.backbone.layer4.parameters())
                for p in ps:p.requires_grad_(True)
                params.append(dict(params=ps,lr=cfg['pixel']['high_lr']))
            opt=torch.optim.Adam(params);order=schedule(rr,labels,seed,2400);lossmode=mode.split('_')[1];history=[];best=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats();maxgrad=0.;budget=1200;step=0
            def feature(i,c=None):
                sl=slice(None)if c is None else slice(c,c+1);d=data[i]
                if mode=='head_bce':return d['feature'][sl]
                lev=[x[sl]for x in d['levels']]
                if mode=='high_bce':
                    p2=lev[0];p3=net.encoder.backbone.layer3(p2);p4=net.encoder.backbone.layer4(p3);lev=[p2,p3,p4]
                return pyramid(net.encoder,lev,8)
            def evaluate(split):
                hs={};empty={};losses=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for i,r in enumerate(rr):
                        if r['split']!=split:continue
                        log=head(feature(i));pr=log.float().sigmoid().cpu().numpy();h=histogram(pr,labels[i]['target'],labels[i]['valid']);g=r['group'];hs[g]=hs.get(g,np.zeros((2,256),np.int64))+h;empty.setdefault(g,np.zeros((2,256),np.int64))
                        if r['configuration']in[0,15]:empty[g]+=h
                        losses.append(float(foreground_loss(log,data[i]['target'],data[i]['valid'],'bce')))
                chosen,curve=select(hs,empty,[x/10 for x in range(1,10)]);pooled=sum(hs.values());low=[dict(threshold=t,**metrics(pooled,t))for t in [x/100 for x in range(1,100)]];eligible=[x for x in low if(x['fpr']or 0)<=.01];return dict(loss=float(np.mean(losses)),selection=chosen,curve=curve,metrics=metrics(pooled,chosen['threshold']),macro_ap=float(np.mean([metrics(h,.5)['histogram_ap']or 0 for h in hs.values()])),low_fpr=max(eligible,key=lambda x:x['recall']or 0)if eligible else None)
            while True:
                if step%200==0:
                    ev=evaluate('val');tr=evaluate('train');history.append(dict(step=step,validation=ev,train=tr));choice=ev['selection'];key=(choice['violation'],-choice['macro_iou'],step)
                    ck=dict(head=head.state_dict(),lateral=net.encoder.lateral.state_dict(),output=net.encoder.output.state_dict(),layer3=net.encoder.backbone.layer3.state_dict(),layer4=net.encoder.backbone.layer4.state_dict(),stride=8,name=mode,seed=seed,selection=dict(choice,step=step),optimizer=opt.state_dict())
                    path=dest/f'm5g_{mode}_{seed}_step{step}.pt';torch.save(ck,path)
                    if best is None or key<best:best=key;selected=dict(name=mode,seed=seed,stride=8,checkpoint=str(path),sha256=file_hash(path),selected=ck['selection'])
                    print(mode,seed,step,ev['macro_ap'],ev['metrics']['precision'],ev['metrics']['recall'],flush=True)
                if step==budget:break
                i,c=order[step];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):log=head(feature(i,c));loss=foreground_loss(log,data[i]['target'][c:c+1],data[i]['valid'][c:c+1],lossmode,data[i]['hard'][c:c+1])
                loss.backward();gr=[p.grad for group in params for p in group['params']if p.grad is not None]
                if not torch.isfinite(loss)or any(not torch.isfinite(g).all()for g in gr):raise FloatingPointError('nonfinite stop')
                maxgrad=max(maxgrad,max(float(g.abs().max())for g in gr));opt.step();step+=1
            results.append(dict(**selected,history=history,steps=budget,schedule=order[:budget],schedule_sha256=file_hash(f'runs/m5f_learning/m5f_schedule_{seed}.json'),parameters=sum(p.numel()for group in params for p in group['params']),finite=True,max_gradient=maxgrad,elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,depth_frozen=all(p.grad is None for p in net.depth_head.parameters())))
            atomic_json(dest/'progress.json',dict(records=results))
    atomic_json('reports/m5g_backbone_ablation.json',dict(kind='actual_RTX4090_matched_pixel_training',records=results,source_sha256=file_hash(__file__),protocol_sha256=file_hash('configs/m5g_protocol.json'),limits='Cachedfrozenlayer2; highlayer3/4 gradient genuine, BNstatsfixed. No originalDepthmodelchanges. Steps400 reconstructedfromsameinitialization/schedule, nothistoricalcheckpointresume. All2seeds1200 comparisons. Extensiondecisionseparate.'))

if __name__=='__main__':main()
