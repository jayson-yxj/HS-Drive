"""Matched bounded A/B, independent Mobile architecture and conditional feature route."""
import argparse,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,torch
from tools.m5i_data import rows
from tools.m5h_sampling import metadata,schedule
from tools.m5f_common import pixel_label
from tools.m5i_teacher_audit import teacher_index
from tools.m5a_learning import encoder_load
from models.m5f_pixel import PixelHead,pyramid
from models.m5i_visual import MobilePixel
from training.m5g_losses import foreground_loss
from training.m5i_transfer import masks,output_kd,feature_regions,feature_kd
from diagnostics.m5f_metrics import histogram,metrics,select
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    p=argparse.ArgumentParser();p.add_argument('--feature',action='store_true');a=p.parse_args();suffix='_feature'if a.feature else'';dest=Path('runs/m5i_learning'+suffix);dest.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);cfg=json.loads(Path('configs/m5i_protocol.json').read_text());rr=rows();labels=[pixel_label(r)for r in rr];ci={r['sample_id']:r for f in ['runs/m5f_cache/index.json','runs/m5h_added_cache/index.json','runs/m5i_added_cache/index.json']for r in json.loads(Path(f).read_text())['records']};mi={r['sample_id']:r for r in json.loads(Path('runs/m5i_mobile_cache/index.json').read_text())['records']};ti=teacher_index();fi={r['sample_id']:r for r in json.loads(Path('runs/m5i_teacher_features/index.json').read_text())['records']}if a.feature else{};data=[]
    if a.feature:
        study=json.loads(Path('reports/m5i_distillation_comparison.json').read_text());assert study['feature_condition_met']
    for r,d in zip(rr,labels):
        c=torch.load(ci[r['sample_id']]['path'],weights_only=True);mobile=torch.load(mi[r['sample_id']]['path'],weights_only=True);t=ti[r['sample_id']];assert file_hash(t['path'])==t['sha256'];q=np.load(t['path']);y=torch.from_numpy(d['target']);v=torch.from_numpy(d['valid']);cl=torch.from_numpy(d['classes']);soft,pos,neg=masks(y,v,cl,torch.from_numpy(q['score'].astype(np.float32)/256),torch.from_numpy(q['trusted'].astype(np.float32)/256));z=dict(levels=[x.cuda()for x in c['levels']],mobile=[x.cuda()for x in mobile['levels']],target=y.cuda(),valid=v.cuda(),soft=soft.cuda(),positive=pos.cuda(),negative=neg.cuda(),regions=[x.cuda()for x in feature_regions(y,v,cl)])
        if a.feature:z['teacherfeature']=torch.load(fi[r['sample_id']]['path'],weights_only=True)['feature'] # CPU cache move selectedcamera only
        data.append(z)
    views=metadata(rr,labels);orders={};coverage=[]
    for seed in cfg['seeds']:
        order,exposure=schedule(views,seed,cfg['pixel']['steps']);orders[seed]=order;coverage.append(dict(seed=seed,views=len(views),unique=len(set(order)),schedule=order,exposure=exposure))
    if not a.feature:atomic_json('reports/m5i_training_exposure.json',dict(records=coverage,source_sha256=file_hash('tools/m5h_sampling.py'),all_arms_same_order=True))
    results=[]
    for mode in (['feature_kd']if a.feature else['gt','output_kd','mobile_gt']):
        for seed in cfg['seeds']:
            torch.manual_seed(seed);net=encoder_load();head=PixelHead().cuda();mobile=None;adapter=None
            if mode=='mobile_gt':
                mobile=MobilePixel(json.loads(Path('reports/m5i_pretrained_provenance.json').read_text())['candidate']['path']).cuda();head=mobile.head;fp=list(mobile.lateral.parameters())+list(mobile.output.parameters())
            else:fp=list(net.encoder.lateral.parameters())+list(net.encoder.output.parameters())
            for x in fp:x.requires_grad_(True)
            params=[dict(params=list(head.parameters()),lr=cfg['pixel']['head_lr']),dict(params=fp,lr=cfg['pixel']['fpn_lr'])]
            if mode=='feature_kd':adapter=torch.nn.Conv2d(64,256,1).cuda();params.append(dict(params=list(adapter.parameters()),lr=cfg['pixel']['feature_adapter_lr']))
            initial_hash=hashlib.sha256()
            for group in params[:2]:
                for parameter in group['params']:initial_hash.update(parameter.detach().float().cpu().numpy().tobytes())
            initialization_sha256=initial_hash.hexdigest()
            opt=torch.optim.Adam(params);history=[];losslog=[];best=None;maxgrad=0.;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            def feature(i,c=None):
                key='mobile'if mode=='mobile_gt'else'levels';lev=[x if c is None else x[c:c+1]for x in data[i][key]];return mobile.fuse(lev)if mobile is not None else pyramid(net.encoder,lev,8)
            def evaluate(split):
                hs={};empty={};losses=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for i,r in enumerate(rr):
                        if r['split']!=split:continue
                        log=head(feature(i));pr=log.float().sigmoid().cpu().numpy();h=histogram(pr,labels[i]['target'],labels[i]['valid']);g=r['group'];hs[g]=hs.get(g,np.zeros((2,256),np.int64))+h;empty.setdefault(g,np.zeros((2,256),np.int64));configs=[x['configuration']for x in rr if x['group']==g]
                        if r['configuration']in[min(configs),max(configs)]:empty[g]+=h
                        losses.append(float(foreground_loss(log,data[i]['target'],data[i]['valid'])))
                chosen,curve=select(hs,empty,[x/10 for x in range(1,10)]);return dict(loss=float(np.mean(losses)),selection=chosen,metrics=metrics(sum(hs.values()),chosen['threshold']),macro_ap=float(np.mean([metrics(h,.5)['histogram_ap']or 0 for h in hs.values()])),scenes={g:metrics(h,chosen['threshold'])for g,h in hs.items()})
            for step in range(cfg['pixel']['steps']+1):
                if step%cfg['pixel']['eval_every']==0:
                    ev=evaluate('val');tr=evaluate('train');history.append(dict(step=step,validation=ev,train=tr));choice=ev['selection'];key=(choice['violation'],-choice['macro_iou'],step);ck=dict(name=mode,seed=seed,stride=8,selection=dict(choice,step=step),optimizer=opt.state_dict())
                    if mobile is not None:ck['model']=mobile.state_dict()
                    else:ck.update(head=head.state_dict(),lateral=net.encoder.lateral.state_dict(),output=net.encoder.output.state_dict())
                    if adapter is not None:ck['adapter']=adapter.state_dict()
                    path=dest/f'm5i_{mode}_{seed}_step{step}.pt';torch.save(ck,path)
                    if best is None or key<best:best=key;selected=dict(name=mode,seed=seed,checkpoint=str(path),sha256=file_hash(path),selected=ck['selection'],validation=ev,unique_views_at_selection=len(set(orders[seed][:step])))
                    print(mode,seed,step,ev['macro_ap'],ev['selection'],flush=True)
                if step==cfg['pixel']['steps']:break
                i,c=orders[seed][step];d=data[i];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    f=feature(i,c);log=head(f);gt=foreground_loss(log,d['target'][c:c+1],d['valid'][c:c+1]);kd=log.sum()*0
                    if mode=='output_kd':kd=output_kd(log,d['soft'][c:c+1],d['positive'][c:c+1],d['negative'][c:c+1])*cfg['pixel']['output_lambda']
                    if mode=='feature_kd':kd=feature_kd(adapter(f),d['teacherfeature'][c:c+1].cuda(),[x[c:c+1]for x in d['regions']])*cfg['pixel']['feature_lambda']
                    loss=gt+kd
                loss.backward();gr=[x.grad for g in params for x in g['params']if x.grad is not None]
                if not torch.isfinite(loss)or any(not torch.isfinite(x).all()for x in gr):raise FloatingPointError('nonfinite')
                maxgrad=max(maxgrad,max(float(g.abs().max())for g in gr));opt.step()
                if step%100==0:losslog.append(dict(step=step+1,gt_loss=float(gt.detach()),weighted_kd=float(kd.detach()),loss=float(loss.detach())))
            results.append(dict(**selected,initialization_sha256=initialization_sha256,schedule_sha256=hashlib.sha256(json.dumps(orders[seed]).encode()).hexdigest(),history=history,losses=losslog,steps=cfg['pixel']['steps'],elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,trainable_parameters=sum(x.numel()for g in params for x in g['params']),max_gradient=maxgrad,finite=True));atomic_json(dest/'progress.json',dict(records=results))
    gains=[]
    if not a.feature:
        for seed in cfg['seeds']:
            aa,bb=[next(r for r in results if r['seed']==seed and r['name']==mode)for mode in ['gt','output_kd']];gains.append(dict(seed=seed,ap_gain=bb['validation']['macro_ap']-aa['validation']['macro_ap'],joint_not_worse=bb['selected']['violation']<=aa['selected']['violation']))
    atomic_json('reports/m5i_distillation_comparison'+suffix+'.json',dict(kind='actual_RTX4090_controlled_training',records=results,validation_gains=gains,feature_condition_met=not all(g['ap_gain']>=.01 and g['joint_not_worse']for g in gains)if gains else None,source_sha256=file_hash(__file__),protocol_sha256=file_hash('configs/m5i_protocol.json'),limits='A/B sameinit/order/steps. Mobile architecturepretraining differs and frozenfeatures only, notfullyfinetunedcapacityupperbound. FeatureC independentGT+feature nooutputKD. Selectedstepmaydiffer; fullbudgetresults retained.'))
if __name__=='__main__':main()
