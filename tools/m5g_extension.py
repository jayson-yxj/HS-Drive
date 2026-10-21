"""Validation-authorized 1200-to2400 continuation; no structure/loss changes."""
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
    dest=Path('reports/m5g_budget_extension.json')
    if dest.exists():raise FileExistsError(dest)
    study=json.loads(Path('reports/m5g_backbone_ablation.json').read_text());base=[r for r in study['records']if r['name']=='fpn_bce'];delta=[next(h for h in r['history']if h['step']==1200)['validation']['macro_ap']-next(h for h in r['history']if h['step']==800)['validation']['macro_ap']for r in base];decision=dict(deltas=delta,allowed=bool(min(delta)>=0 and np.mean(delta)>=.01),rule='bothnondecreasing andmeanAPgain>=.01 val800to1200')
    atomic_json('reports/m5g_extension_decision.json',decision)
    if not decision['allowed']:atomic_json(dest,dict(decision=decision,records=[]));return
    prepare_cuda();torch.set_num_threads(2);rr=rows();labels=[pixel_label(r)for r in rr];idx={r['sample_id']:r for r in json.loads(Path('runs/m5f_cache/index.json').read_text())['records']};data=[];out=Path('runs/m5g_extension');out.mkdir(exist_ok=False)
    for i,r in enumerate(rr):
        c=torch.load(idx[r['sample_id']]['path'],weights_only=True);data.append(dict(levels=[x.cuda()for x in c['levels']],target=torch.from_numpy(labels[i]['target']).cuda(),valid=torch.from_numpy(labels[i]['valid']).cuda()))
    results=[]
    for original in base:
        seed=original['seed'];net=encoder_load();head=PixelHead().cuda();ck=torch.load(f'runs/m5g_pixels/m5g_fpn_bce_{seed}_step1200.pt',weights_only=True);head.load_state_dict(ck['head']);net.encoder.lateral.load_state_dict(ck['lateral']);net.encoder.output.load_state_dict(ck['output']);ps=list(net.encoder.lateral.parameters())+list(net.encoder.output.parameters())
        for p in ps:p.requires_grad_(True)
        opt=torch.optim.Adam([dict(params=head.parameters(),lr=.001),dict(params=ps,lr=.0001)]);opt.load_state_dict(ck['optimizer']);order=schedule(rr,labels,seed,2400);history=[];best=None;torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for step in range(1200,2401):
            if step%200==0:
                hs={};empty={};losses=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for i,r in enumerate(rr):
                        if r['split']!='val':continue
                        d=data[i];log=head(pyramid(net.encoder,d['levels'],8));prob=log.float().sigmoid().cpu().numpy();h=histogram(prob,labels[i]['target'],labels[i]['valid']);g=r['group'];hs[g]=hs.get(g,np.zeros((2,256),np.int64))+h;empty.setdefault(g,np.zeros((2,256),np.int64))
                        if r['configuration']in[0,15]:empty[g]+=h
                        losses.append(float(foreground_loss(log,d['target'],d['valid'])))
                selection,curve=select(hs,empty,[x/10 for x in range(1,10)]);ev=dict(selection=selection,metrics=metrics(sum(hs.values()),selection['threshold']),macro_ap=float(np.mean([metrics(h,.5)['histogram_ap']or 0 for h in hs.values()])),loss=float(np.mean(losses)));history.append(dict(step=step,validation=ev));path=out/f'm5g_fpn_bce_extended_{seed}_step{step}.pt';ck=dict(head=head.state_dict(),lateral=net.encoder.lateral.state_dict(),output=net.encoder.output.state_dict(),selection=dict(selection,step=step),stride=8);torch.save(ck,path);key=(selection['violation'],-selection['macro_iou'],step)
                if best is None or key<best:best=key;selected=dict(name='fpn_bce_extended',seed=seed,checkpoint=str(path),sha256=file_hash(path),selected=ck['selection'])
                print(seed,step,ev,flush=True)
            if step==2400:break
            i,c=order[step];d=data[i];opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):log=head(pyramid(net.encoder,[x[c:c+1]for x in d['levels']],8));loss=foreground_loss(log,d['target'][c:c+1],d['valid'][c:c+1])
            loss.backward()
            if not torch.isfinite(loss)or any(not torch.isfinite(p.grad).all()for p in list(head.parameters())+ps if p.grad is not None):raise FloatingPointError('extensionnonfinite')
            opt.step()
        results.append(dict(**selected,history=history,steps=2400,extra_steps=1200,elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    atomic_json(dest,dict(kind='actual_GPU_predeclared_conditional_budget_extension',decision=decision,records=results,source_sha256=file_hash(__file__),limits='Continuationfromactual1200optimizer, extra1200. Notmatchedbudgetbackbonecomparison; reportseparately. NoTestaccess.'))

if __name__=='__main__':main()
