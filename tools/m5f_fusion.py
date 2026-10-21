"""Matched 100-step scalar fusion versus old-only recalibration, no new networks."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from training.m5f_loss import pixel_loss
from diagnostics.m5f_metrics import histogram,metrics
from simulator.recorder import atomic_json,file_hash


def features(d,mode):
    x=d['old_logit'].clip(-12,12)[None]
    if mode=='fusion':x=np.concatenate([x,np.log1p(d['foreground_mass'])[None],np.log1p(d['total_mass'])[None]],0)
    return torch.from_numpy(x)


def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');a=p.parse_args();root=Path('runs/m5f_bev_test_features' if a.test else 'runs/m5f_bev_features');rows=json.loads((root/'index.json').read_text())['records'];dest=Path('reports/m5f_bev_foreground_evidence.json' if a.test else 'reports/m5f_fusion_training.json')
    if dest.exists():raise FileExistsError(dest)
    data=[]
    for r in rows:
        with np.load(r['path']) as z:data.append({k:z[k] for k in z.files})
    out=Path('runs/m5f_fusion');out.mkdir(exist_ok=True);results=[];torch.set_num_threads(2)
    for seed in [101,102]:
        rng=np.random.default_rng(seed);train=[i for i,r in enumerate(rows) if r['seed']==seed and r['split']=='train'];val=[i for i,r in enumerate(rows) if r['seed']==seed and r['split']==('test' if a.test else 'val')];schedule=rng.choice(train,100).tolist() if train else []
        for mode in ['old_only','fusion']:
            torch.manual_seed(seed);w=torch.nn.Parameter(torch.zeros(1 if mode=='old_only' else 3));bias=torch.nn.Parameter(torch.tensor(0.));w.data[0]=1;ckpath=out/f'm5f_{mode}_{seed}.pt';history=[]
            if a.test:
                ck=torch.load(ckpath,weights_only=True);w.data.copy_(ck['w']);bias.data.copy_(ck['bias']);th=ck['threshold']
            else:
                opt=torch.optim.Adam([w,bias],lr=.01)
                for step,i in enumerate(schedule):
                    d=data[i];opt.zero_grad();log=(features(d,mode)*w[:,None,None,None]).sum(0)+bias;loss=pixel_loss(log,torch.from_numpy(d['target']),torch.from_numpy(d['valid']));loss.backward();opt.step()
                    if step%10==0:history.append(dict(step=step,loss=float(loss.detach())))
            hist=np.zeros((2,256),np.int64);records=[]
            for i in val:
                d=data[i];score=torch.sigmoid((features(d,mode)*w[:,None,None,None]).sum(0)+bias).detach().numpy();h=histogram(score,d['target'],d['valid']);hist+=h;records.append(dict(group=rows[i]['group'],configuration=rows[i]['configuration'],histogram=h.tolist(),unknown_high_fraction=float((score[~d['valid']]>.5).mean())))
            if not a.test:
                curve=[dict(threshold=t,**metrics(hist,t)) for t in [.1,.2,.4,.6,.8,.9]];th=max(curve,key=lambda v:(v['iou'] or 0,v['threshold']))['threshold'];torch.save(dict(w=w.detach(),bias=bias.detach(),threshold=th),ckpath)
            results.append(dict(seed=seed,mode=mode,threshold=th,metrics=metrics(hist,th),records=records,history=history,weights=w.detach().tolist(),bias=float(bias.detach()),checkpoint=str(ckpath),sha256=file_hash(ckpath)))
    atomic_json(dest,dict(kind='offline_CPU_scalar_supervision_fusion_test' if a.test else 'offline_CPU_matched_scalar_fusion_training',records=results,source_sha256=file_hash(__file__),limits='100steps2seeds oldonly vsfusion sameframeorder; no backbonetrain. Fusion trainedknownvisiblefirstsurfaces, maynotgeneralizeunknown. Modelselected P8fpn a priori researchcandidate nottestdriven. Oldrawthreshold availableM5E; recalibration comparator distinct.'))

if __name__=='__main__':main()
