"""Actual fixed-validation ray distributions and learning curves; no test tuning."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5e_common import ROOT,sidecars,load_checkpoint
from tools.m5a_learning import encoder_load,stack
from datasets.m5a_training import VisualEvidenceDataset
from models.m5b_cached_encoder import cache_rgb
from models.m5e_paths import geometry,depth_logits
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    out=Path('reports/m5e_diagnostic_figures');out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2)
    ds=VisualEvidenceDataset(ROOT,'val',supervision=False);sup=sidecars(ROOT,ds.rows,'m5e_supervision.json');net=encoder_load();chosen=[]
    for lo,hi in [(10,15),(15,20),(20,25),(25,35)]:
        for i,s in enumerate(sup):
            c=cache_rgb(net,stack([ds[i]],'inputs'));g=geometry(net,c,torch.from_numpy(s['depth16'][None]).cuda());m=torch.from_numpy((s['valid16']&s['actor16'])[None]).cuda()&g['inside']&(g['rad']>=lo)&(g['rad']<hi)
            if m.any():chosen.append((lo,hi,i,c,g,tuple(m.nonzero()[0].tolist())));break
    fig,axes=plt.subplots(len(chosen),2,figsize=(13,3.4*len(chosen)));records=[]
    for j,(lo,hi,i,c,g,idx) in enumerate(chosen):
        b,n,y,x=idx;gt=float(sup[i]['depth16'][n,y,x]);curves=[]
        for name,seed in [('original_depth',0),('D_depth',91),('D_depth',92)]:
            net=encoder_load()
            if seed:net.depth_head.load_state_dict(load_checkpoint(f'runs/m5e_learning/m5e_D_depth_{seed}.pt')['depth'])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=depth_logits(net,c).float().softmax(2)[b,n,:,y,x].cpu().numpy()
            bins=net.splat.depth_centers.cpu().numpy();curves.append(dict(name=name,seed=seed,probability=p.tolist()));axes[j,0].plot(bins,p,label=f'{name}{seed}');xy=g['points'][b,n,:,y,x,:2].cpu().numpy();axes[j,1].scatter(xy[:,1],xy[:,0],s=5+800*p,alpha=.4,label=f'{name}{seed}')
        gtpt=g['gt'][idx].cpu().numpy();axes[j,0].axvline(gt,color='k',ls='--');axes[j,0].set(xlabel='Optical depth m',ylabel='Probability',title=f'{ds.rows[i]["group"]} cfg{ds.rows[i]["configuration"]} camera{n} featureUV({x},{y})');axes[j,0].legend(fontsize=8);axes[j,1].scatter(gtpt[1],gtpt[0],c='r',marker='x',s=100);axes[j,1].set(xlabel='HS Y left m',ylabel='HS X forward m',title=f'GT radial {np.linalg.norm(gtpt[:2]):.1f}m; marker area ~ bin mass');records.append(dict(group=ds.rows[i]['group'],configuration=ds.rows[i]['configuration'],feature_index=list(idx),gt_depth_m=gt,gt_xyz=gtpt.tolist(),bin_depths=bins.tolist(),bin_xy=xy.tolist(),curves=curves))
    fig.tight_layout();fig.savefig(out/'m5e_depth_distributions.png',dpi=150);plt.close(fig)
    train=json.loads(Path('reports/m5e_model_ablation.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,8))
    for r in train['records']:
        h=[x for x in r['history'] if 'total' in x]
        for ax,key in zip(axes.ravel(),['total','existence','presence','spatial']):ax.plot([x['step'] for x in h],[x[key] for x in h],label=f'{r["name"]} {r["seed"]}');ax.set(xlabel='Matched optimizer step',ylabel=key)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5e_training_losses.png',dpi=150);plt.close(fig)
    atomic_json('reports/m5e_distribution_examples.json',dict(kind='actual_GPU_fixed_validation_first_eligible_ray_each_band',records=records,source_sha256=file_hash(__file__),limits='Chosen first eligible validationray by deterministicorder, notrepresentative statistic; plotted all48bins with noGT inferenceinput. Spatialplot includes invalidBEV points to expose leakage.'))

if __name__=='__main__':main()
