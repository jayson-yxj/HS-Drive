"""Export diagnostic figures from immutable study results; no parameter selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tools.m5h_data import rows
from tools.m5h_shadow import TEST
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash

def read(p):return json.loads(Path(p).read_text())
def main():
    out=Path('reports/m5h_figures');out.mkdir(exist_ok=False);pixel=read('reports/m5h_distillation_ablation.json');depth=read('reports/m5h_depth_preservation.json');pr=read('reports/m5h_pixel_test.json');teacher=read('reports/m5h_teacher_test.json');shadow=read('reports/m5h_shadow_comparison.json');rr=rows(TEST);frozen=read('reports/m5h_pretest_freeze.json');fig,ax=plt.subplots(1,3,figsize=(15,4))
    for r in pixel['records']:
        h=[v for v in r['history']if 'validation'in v];name=f'{r["name"]} seed{r["seed"]}';x=[v['step']for v in h];ax[0].plot(x,[v['validation']['macro_ap']for v in h],label=name);ax[1].plot(x,[v['validation']['metrics']['fpr']for v in h],label=name);ax[2].plot(x,[v['validation']['loss']for v in h],label=name)
    for a,title in zip(ax,['Validation macro AP','Validation background FPR','Validation GT BCE']):a.set_title(title);a.set_xlabel('Actual optimizer updates');a.grid(alpha=.2)
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5h_pixel_learning.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for r in depth['records']:
        h=[v for v in r['history']if 'validation'in v];x=[v['step']for v in h];name=f'{r["name"]} seed{r["seed"]}'
        for a,key in zip(ax,['near_mae','original_S16_val_near_mae','target_spatial_macro']):a.plot(x,[v['validation'][key]for v in h],label=name);a.set_title(key);a.set_xlabel('Actual updates');a.grid(alpha=.2)
    ax[0].axhline(depth['original_validation']['near_mae']*1.05,c='r',ls='--',label='near 1.05 gate');ax[1].axhline(depth['original_validation']['original_S16_val_near_mae']*1.05,c='r',ls='--');ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5h_depth_preservation.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for r in pr['records']+teacher['records']:
        h=np.array(r['supported']['histogram']);pos,neg=h;tp=np.cumsum(pos[::-1])[::-1];fp=np.cumsum(neg[::-1])[::-1];precision=tp/np.maximum(tp+fp,1);rec=tp/max(pos.sum(),1);fpr=fp/max(neg.sum(),1);name=f'{r["name"]} {r["seed"]}';ax[0].plot(rec,precision,label=name);ax[1].plot(fpr,rec,label=name)
    ax[0].set(xlabel='Recall',ylabel='Precision',title='Same supported vehicle/person domain');ax[1].set(xlabel='Confirmed background FPR',ylabel='Recall',xscale='log',xlim=(1e-5,1),title='Test curves descriptive; thresholds frozen');ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5h_low_fpr_curves.png',dpi=150);plt.close(fig)
    # Fixed configuration and camera, one per test group: no cherry-picking by performance.
    for group in sorted({r['group']for r in rr}):
        row=next(r for r in rr if r['group']==group and r['configuration']==1);inp=read(row['input']);d=pixel_label(row);c=0;img=cv2.cvtColor(cv2.imread(inp['images'][c]['path']),cv2.COLOR_BGR2RGB);fig,ax=plt.subplots(2,3,figsize=(15,6));ax[0,0].imshow(img);ax[0,0].set_title(group+' front10m');ax[0,1].imshow(np.where(d['valid'][c],d['target'][c].astype(float),np.nan),vmin=0,vmax=1);ax[0,1].set_title('Reliable GT (Unknown masked)')
        selected=[next(r for r in pr['records']if r['name']==n and r['seed']==101)for n in ['M5G_fpn_bce','gt','distill']]+[teacher['records'][0]]
        for a,r in zip([ax[0,2],ax[1,0],ax[1,1],ax[1,2]],selected):
            path=next(v['path']for v in r['saved']if v['sample_id']==row['sample_id']);p=np.load(path)['probability_u8'][c]/256;a.imshow(p,vmin=0,vmax=1,cmap='magma');a.set_title(r['name'])
        for a in ax.flat:a.axis('off')
        fig.tight_layout();fig.savefig(out/(group+'_m5h_pixel.png'),dpi=140);plt.close(fig)
        fig,ax=plt.subplots(2,3,figsize=(12,9));modes=['oldpixel_olddepth','newpixel_olddepth','oldpixel_newdepth','newpixel_newdepth','oracle2d_oracledepth','teacher2d_oracledepth']
        for a,mode in zip(ax.flat,modes):
            r=next(r for r in shadow['records']if r['sample_id']==row['sample_id']and r['seed']==101 and r['mode']==mode);z=np.load(r['path']);a.imshow(z['score'].max(0).T,origin='lower',extent=[-20,60,-30,30],vmin=0,vmax=1,cmap='magma');a.contour(np.any(z['target']&z['valid'],axis=0).T,levels=[.5],colors='cyan',linewidths=.5,origin='lower',extent=[-20,60,-30,30]);a.set_title(mode,fontsize=9);a.set_xlabel('ego x (m)');a.set_ylabel('ego y (m)')
        fig.suptitle(group+' 10m: max over height, cyan visible GT; full4layermetrics inJSON');fig.tight_layout();fig.savefig(out/(group+'_m5h_bev.png'),dpi=140);plt.close(fig)
    atomic_json('reports/m5h_visualizations.json',dict(kind='offline_figures_of_actual_results',files={str(p):file_hash(p)for p in out.iterdir()},source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
