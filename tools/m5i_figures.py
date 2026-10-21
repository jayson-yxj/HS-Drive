"""Figures from frozen evaluation data, fixed displayed configuration not modelselection."""
import json
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5i_data import rows,TEST
from tools.m5f_common import pixel_label,input_path
from simulator.recorder import atomic_json,file_hash

def read(n):return json.loads(Path('reports',n+'.json').read_text())
def main():
    out=Path('reports/m5i_figures');out.mkdir(exist_ok=False);pix=read('m5i_pixel_test');teach=read('m5i_teacher_test');train=read('m5i_distillation_comparison')['records'];f=Path('reports/m5i_distillation_comparison_feature.json')
    if f.exists():train+=json.loads(f.read_text())['records']
    fig,ax=plt.subplots(1,3,figsize=(16,4))
    for r in train:
        h=r['history'];steps=[x['step']for x in h];name=f'{r["name"]}_{r["seed"]}';ax[0].plot(steps,[x['validation']['macro_ap']for x in h],label=name);ax[1].plot(steps,[x['validation']['selection']['violation']for x in h]);ax[2].plot(steps,[x['train']['loss']for x in h])
    for a,title in zip(ax,['Val macroAP','Val jointviolation (lower better)','Train GT BCE']):a.set_title(title);a.set_xlabel('updates');a.grid(alpha=.2)
    ax[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'m5i_learning.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for r in pix['records']+teach['records']:
        pos,neg=np.array(r['supported']['histogram']);tp=pos[::-1].cumsum()[::-1];fp=neg[::-1].cumsum()[::-1];rec=tp/max(pos.sum(),1);ax[0].plot(rec,tp/np.maximum(tp+fp,1),label=f'{r["name"]}{r["seed"]}');ax[1].plot(fp/max(neg.sum(),1),rec)
    ax[0].set(xlabel='Recall',ylabel='Precision',title='Same vehicle/person reliable mask');ax[1].set(xlabel='Background FPR',ylabel='Recall',xscale='log',xlim=(1e-5,1));ax[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'m5i_precision_recall.png',dpi=150);plt.close(fig)
    bias=read('m5i_depth_bias')['splits']['test'];fig,ax=plt.subplots(1,2,figsize=(12,4));dist=['1.5_10','10_20','20_35','35_60','60_80']
    for name,seed in [('original',0),('replay',101),('replay',102)]:
        rr=[next(r for r in bias if r['name']==name and r['seed']==seed and r['category']=='all'and r['scope']=='all'and r['distance']==d)for d in dist]
        ax[0].plot(dist,[r['mae']for r in rr],marker='o',label=f'{name}{seed}');ax[1].plot(dist,[r['bias']for r in rr],marker='o',label=f'{name}{seed}')
    ax[0].set_title('Expected depth MAE');ax[1].set_title('Signed bias by optical-Z range');ax[0].legend();ax[1].axhline(0,c='grey');fig.tight_layout();fig.savefig(out/'m5i_depth_bias.png',dpi=150);plt.close(fig)
    rr=rows(TEST)
    for group in sorted({r['group']for r in rr}):
        row=next(r for r in rr if r['group']==group and r['configuration']==4);d=pixel_label(row);inp=json.loads(input_path(row).read_text());c=row['camera_names'].index('rear_right');fig,ax=plt.subplots(2,3,figsize=(15,6));ax[0,0].imshow(cv2.cvtColor(cv2.imread(inp['images'][c]['path']),cv2.COLOR_BGR2RGB));ax[0,0].set_title(group+' rear_right config4');ax[0,1].imshow(np.where(d['valid'][c],d['target'][c].astype(float),np.nan));ax[0,1].set_title('GT firstsurface / Unknownwhite');choices=[next(r for r in pix['records']if r['name']==n and r['seed']==101)for n in ['gt','output_kd','mobile_gt']]+[teach['records'][0]]
        for a,r in zip([ax[0,2],ax[1,0],ax[1,1],ax[1,2]],choices):
            p=next(x['path']for x in r['saved']if x['sample_id']==row['sample_id']);a.imshow(np.load(p)['probability_u8'][c]/256,cmap='magma',vmin=0,vmax=1);a.set_title(r['name'])
        for a in ax.flat:a.axis('off')
        fig.tight_layout();fig.savefig(out/(group+'_m5i_rear.png'),dpi=140);plt.close(fig)
    shadow=read('m5i_shadow_comparison')['records']
    for group in sorted({r['group']for r in rr}):
        fig,axs=plt.subplots(2,3,figsize=(12,9))
        for ax,mode in zip(axs.flat,['old_old','new_old','old_replay','new_replay','teacher_oracle','oracle_oracle']):
            rec=next(r for r in shadow if r['group']==group and r['seed']==101 and r['configuration']==11 and r['mode']==mode)
            data=np.load(rec['path']);score=data['score'].max(axis=0);valid=data['valid'].any(axis=0);truth=data['target'].any(axis=0)
            ax.imshow(score,origin='lower',extent=(-30,30,-20,60),vmin=0,vmax=1,cmap='magma',aspect='equal')
            yy,xx=np.where(truth);ax.scatter(-30+(xx+.5)*.5,-20+(yy+.5)*.5,s=2,c='cyan',label='visible GT surface')
            ax.contour(np.arange(120)*.5-29.75,np.arange(160)*.5-19.75,valid,levels=[.5],colors=['grey'],linewidths=.3)
            ax.set(title=mode,xlabel='ego y left (m)',ylabel='ego x forward (m)')
        fig.suptitle(group+' fixed config11: max-height display; metrics keep layers; unknown not negative')
        fig.tight_layout();fig.savefig(out/(group+'_m5i_bev.png'),dpi=140);plt.close(fig)
    atomic_json('reports/m5i_visualizations.json',dict(kind='offline_frozen_result_figures',files={str(p):file_hash(p)for p in out.iterdir()},source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
