"""Standalone diagnostic plots from actual frozen reports and per-ray records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(name):return json.loads(Path('reports',name+'.json').read_text())


def main():
    out=Path('reports/m5e_figures');out.mkdir(exist_ok=False)
    audit=read('m5e_witness_attrition');fig,axes=plt.subplots(2,3,figsize=(13,7));chosen=[]
    for r in audit['records']:
        with np.load(r['path']) as z:
            for kind in ['candidate','edge']:
                m=z['candidate'] if kind=='candidate' else ((z['rejection_bits']&(1<<5))>0)
                ids=np.flatnonzero(m&(z['distance']>=20))
                if len(ids) and kind not in [x[0] for x in chosen]:chosen.append((kind,r,{k:z[k][ids[0]] for k in ['uv','background_patch','plane_residual_m','depth_patch_range_m','points_hs']}))
        if len(chosen)==2:break
    for i,(kind,r,d) in enumerate(chosen):
        axes[i,0].imshow(d['background_patch'],cmap='viridis');axes[i,0].set_title(f"{kind}: {r['group']} UV{d['uv'].tolist()}")
        axes[i,1].plot(d['background_patch'][2],marker='o');axes[i,1].set(title=f"range {d['depth_patch_range_m']:.2f}m, residual{d['plane_residual_m']:.3f}m",ylabel='depth m');axes[i,2].text(.05,.6,f"{r['camera']} cfg{r['configuration']}\nHS point {np.round(d['points_hs'],2)}\nCandidate is NOT negative supervision",fontsize=10);axes[i,2].axis('off')
    fig.tight_layout();fig.savefig(out/'m5e_slope_vs_boundary.png',dpi=150);plt.close(fig)
    d=read('m5e_evaluation_digest');fig,axes=plt.subplots(1,2,figsize=(12,5))
    for r in d['records']:
        w=r['witness_micro'];axes[0].scatter(w['absence_fpr'],r['actor_exact_recall'],label=f"{r['name']} {r['seed']}");axes[1].scatter(w['absence_fpr'],w['presence_recall'],label=f"{r['name']} {r['seed']}")
    axes[0].set(xlabel='Gold absence FPR',ylabel='Actor exact surface recall');axes[1].set(xlabel='Gold absence FPR',ylabel='Gold presence recall')
    for a in axes:a.set_xlim(0,1);a.set_ylim(0,1);a.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5e_actor_absence_pareto.png',dpi=150);plt.close(fig)
    geo=read('m5e_depth_spatial_probability')['summary'];fig,axes=plt.subplots(1,2,figsize=(12,5));bands=['0_10','10_15','15_20','20_25','25_35']
    for name,seed in [('original_depth',0),('D_depth',91),('D_depth',92)]:
        rr=[next((r for r in geo if r['name']==name and r['seed']==seed and r['target']=='actor' and r['mode']=='learned_probability' and r['distance']==b),None) for b in bands]
        axes[0].plot(bands,[r['expected_abs_xy_m'] if r else np.nan for r in rr],marker='o',label=f'{name}{seed}');axes[1].plot(bands,[r['xy_1.0m_mass'] if r else np.nan for r in rr],marker='o',label=f'{name}{seed}')
    axes[0].set(ylabel='E[XY absolute spatial error] m');axes[1].set(ylabel='Correct layer and XY <=1m probability mass')
    for a in axes:a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'m5e_depth_spatial_probability.png',dpi=150);plt.close(fig)
    predictions=read('m5e_presence_absence');index=Path('/root/gpufree-data/hs-drive-data/processed/m5e_test_index');rows=json.loads((index/'index.json').read_text())['samples'];group=sorted({r['group'] for r in rows})[0];configs=[0,2,4,6,15];names=['Historical_A_matched','A_base','B_presence','D_depth'];fig,axes=plt.subplots(4,5,figsize=(13,11))
    for i,name in enumerate(names):
        r=next(r for r in predictions['records'] if r['name']==name and r['seed']==(81 if name.startswith('Historical') else 91));s=np.load(r['predictions'])['score']
        for j,c in enumerate(configs):
            ix=next(k for k,r in enumerate(rows) if r['group']==group and r['configuration']==c);axes[i,j].imshow(s[ix,2],origin='lower',vmin=0,vmax=1,cmap='magma',extent=(-30,30,-20,60));axes[i,j].set(xlim=(-8,8),ylim=(0,40),title=f'{name}\ncfg{c}',xlabel='Y left m',ylabel='X forward m')
    fig.suptitle('Height[1,2.5)m; no GTmask at inference; empty / obstacles / removed');fig.tight_layout();fig.savefig(out/'m5e_singleframe_maps.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
