"""Paired same-order GT versus positive-agreement teacher distillation."""
import json,time
from pathlib import Path
from collections import Counter
import torch,numpy as np
from tools.m5h_data import rows
from tools.m5h_sampling import metadata,schedule
from tools.m5f_common import pixel_label
from tools.m5a_learning import encoder_load
from models.m5f_pixel import PixelHead,pyramid
from training.m5g_losses import foreground_loss
from training.m5h_losses import teacher_mask,distillation_loss
from diagnostics.m5f_metrics import histogram,metrics,select
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    dest=Path('runs/m5h_pixels');dest.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);cfg=json.loads(Path('configs/m5h_protocol.json').read_text());rr=rows();labels=[pixel_label(r)for r in rr];cacheidx={r['sample_id']:r for f in ['runs/m5f_cache/index.json','runs/m5h_added_cache/index.json']for r in json.loads(Path(f).read_text())['records']};ti={r['sample_id']:r for r in json.loads(Path('runs/m5h_teacher_trainval/index.json').read_text())['records']};data=[]
    for i,r in enumerate(rr):
        c=torch.load(cacheidx[r['sample_id']]['path'],weights_only=True);t=ti[r['sample_id']];assert file_hash(t['path'])==t['sha256'];q=np.load(t['path'])['trusted'].astype(np.float32)/256;d=labels[i];score,mask=teacher_mask(torch.from_numpy(d['target']),torch.from_numpy(d['valid']),torch.from_numpy(d['classes']),torch.from_numpy(q));data.append(dict(levels=[x.cuda()for x in c['levels']],target=torch.from_numpy(d['target']).cuda(),valid=torch.from_numpy(d['valid']).cuda(),teacher=score.cuda(),distill_mask=mask.cuda()))
    views=metadata(rr,labels);coverage=[];orders={}
    for seed in cfg['seeds']:
        order,exposure=schedule(views,seed,cfg['pixel']['steps']);orders[seed]=order;coverage.append(dict(seed=seed,available_views=len(views),unique_views=len(set(order)),unseen_views=len(views)-len(set(order)),exposure=exposure,by_stratum=dict(Counter(t for v in exposure for t in v['tags']for _ in range(v['updates']))),schedule=order))
    positive_by_camera={name:sum(v['positive']>0 and v['camera']==name for v in views)for name in rr[0]['camera_names']}
    if not all(positive_by_camera.values()):raise RuntimeError('Missing camera positive coverage '+str(positive_by_camera))
    atomic_json('reports/m5h_training_coverage.json',dict(kind='offline_actual_train_labels_and_executed_fixed_schedule',records=coverage,positive_views_by_camera=positive_by_camera,train_groups=len({r['group']for r in rr if r['split']=='train'}),val_groups=len({r['group']for r in rr if r['split']=='val'}),teacher_mask_pixels_by_frame=[dict(sample_id=r['sample_id'],n=int(d['distill_mask'].sum()))for r,d in zip(rr,data)],source_sha256=file_hash('tools/m5h_sampling.py'),limits='Complete view coverage is not exhaustive appearance coverage. Addedtrainpositions and newsampler jointly changed vsM5G, cannot separatetheircausal effects.'))
    results=[]
    for mode in ['gt','distill']:
        for seed in cfg['seeds']:
            torch.manual_seed(seed);net=encoder_load();head=PixelHead().cuda();fp=list(net.encoder.lateral.parameters())+list(net.encoder.output.parameters());[p.requires_grad_(True)for p in fp];params=[dict(params=list(head.parameters()),lr=cfg['pixel']['head_lr']),dict(params=fp,lr=cfg['pixel']['fpn_lr'])];opt=torch.optim.Adam(params);order=orders[seed];history=[];best=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats();maxgrad=0.;losslog=[]
            def feature(i,c=None):return pyramid(net.encoder,[x if c is None else x[c:c+1]for x in data[i]['levels']],8)
            def evaluate(split):
                hs={};empty={};losses=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for i,r in enumerate(rr):
                        if r['split']!=split:continue
                        log=head(feature(i));pr=log.float().sigmoid().cpu().numpy();h=histogram(pr,labels[i]['target'],labels[i]['valid']);g=r['group'];hs[g]=hs.get(g,np.zeros((2,256),np.int64))+h;empty.setdefault(g,np.zeros((2,256),np.int64));configs=[a['configuration']for a in rr if a['group']==g]
                        if r['configuration']in[min(configs),max(configs)]:empty[g]+=h
                        losses.append(float(foreground_loss(log,data[i]['target'],data[i]['valid'])))
                chosen,curve=select(hs,empty,[x/10 for x in range(1,10)]);return dict(loss=float(np.mean(losses)),selection=chosen,curve=curve,metrics=metrics(sum(hs.values()),chosen['threshold']),macro_ap=float(np.mean([metrics(h,.5)['histogram_ap']or 0 for h in hs.values()])))
            for step in range(cfg['pixel']['steps']+1):
                if step%cfg['pixel']['evaluate_every']==0:
                    ev=evaluate('val');tr=evaluate('train');history.append(dict(step=step,validation=ev,train=tr));choice=ev['selection'];key=(choice['violation'],-choice['macro_iou'],step);ck=dict(head=head.state_dict(),lateral=net.encoder.lateral.state_dict(),output=net.encoder.output.state_dict(),stride=8,name=mode,seed=seed,selection=dict(choice,step=step),optimizer=opt.state_dict());path=dest/f'm5h_{mode}_{seed}_step{step}.pt';torch.save(ck,path)
                    if best is None or key<best:best=key;selected=dict(name=mode,seed=seed,checkpoint=str(path),sha256=file_hash(path),selected=ck['selection'],validation=ev)
                    print(mode,seed,step,ev['macro_ap'],ev['metrics'],flush=True)
                if step==cfg['pixel']['steps']:break
                i,c=order[step];d=data[i];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    log=head(feature(i,c));gt=foreground_loss(log,d['target'][c:c+1],d['valid'][c:c+1]);kd=distillation_loss(log,d['teacher'][c:c+1],d['distill_mask'][c:c+1]);loss=gt+(cfg['pixel']['teacher_lambda']*kd if mode=='distill'else 0)
                loss.backward();gr=[p.grad for group in params for p in group['params']if p.grad is not None]
                if not torch.isfinite(loss)or any(not torch.isfinite(g).all()for g in gr):raise FloatingPointError('nonfinite')
                maxgrad=max(maxgrad,max(float(g.abs().max())for g in gr));opt.step()
                if step%100==0:losslog.append(dict(step=step+1,gt=float(gt.detach()),teacher=float(kd.detach()),total=float(loss.detach()),teacher_n=int(d['distill_mask'][c].sum())))
            results.append(dict(**selected,history=history,losses=losslog,steps=cfg['pixel']['steps'],parameters=sum(p.numel()for group in params for p in group['params']),elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,max_gradient=maxgrad,all_finite=True,depth_frozen=all(p.grad is None for p in net.depth_head.parameters())));atomic_json(dest/'progress.json',dict(records=results))
    paired=[]
    for seed in cfg['seeds']:
        a,b=[next(r for r in results if r['seed']==seed and r['name']==mode)for mode in ['gt','distill']];paired.append(dict(seed=seed,ap_gain=b['validation']['macro_ap']-a['validation']['macro_ap'],violation_not_worse=b['selected']['violation']<=a['selected']['violation']))
    eligible=all(r['ap_gain']>=.01 and r['violation_not_worse']for r in paired)
    atomic_json('reports/m5h_distillation_ablation.json',dict(kind='actual_RTX4090_paired_GT_vs_teacher',records=results,paired_validation=paired,conditional_hard_eligible=eligible,conditional_hard_status='Pending'if eligible else'NotRun_condition_failed',backbone_upgrade='NotRun_no_unconfounded_capacity_evidence',protocol_sha256=file_hash('configs/m5h_protocol.json'),source_sha256=file_hash(__file__),limits='Sameinit/order2400budget, selectedstep maydiffer. CachedfrozenRGBbackbone, noTeacherduringsteps. Teacherpositiveagreement only; outcome doesnotruleout feature/negative distillation.'))
if __name__=='__main__':main()
