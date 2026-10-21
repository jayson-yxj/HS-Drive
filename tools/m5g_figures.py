"""Auditable plots, positive histogram first. Figures from actual stored results."""
import json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('reports/m5g_figures');out.mkdir(exist_ok=False);study=json.loads(Path('reports/m5g_backbone_ablation.json').read_text())['records'];fig,ax=plt.subplots(2,5,figsize=(22,8))
    for r in study:
        h=r['history'];x=[v['step']for v in h];label=f"{r['name']}/{r['seed']}"
        for row,scope in enumerate(['train','validation']):
            for col,key in enumerate(['loss','histogram_ap','precision','recall','empty_fpr']):
                if key=='loss':y=[v[scope]['loss']for v in h]
                elif key=='empty_fpr':y=[np.mean([s['empty_fpr']or 0 for s in v[scope]['selection']['scenes'].values()])for v in h]
                else:y=[v[scope]['metrics'][key]for v in h]
                ax[row,col].plot(x,y,label=label);ax[row,col].set_title(scope+' '+key);ax[row,col].set_xlabel('updates')
    ax[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'m5g_learning_curves.png',dpi=150);plt.close(fig)
    exp=json.loads(Path('reports/m5g_training_exposure.json').read_text());fig,ax=plt.subplots(1,2,figsize=(11,4))
    for r in exp['records']:
        ax[0].bar(str(r['seed']),r['unique_views'],label=f"{r['coverage']:.1%} covered");ax[1].plot(list(r['by_camera']),[v['positive_updates']for v in r['by_camera'].values()],marker='o',label=str(r['seed']))
    ax[0].axhline(1152,color='k',linestyle='--');ax[0].set_title('Historical distinct views /1152');ax[0].legend();ax[1].set_title('Historical positive image updates');ax[1].tick_params(axis='x',rotation=45);ax[1].legend();fig.tight_layout();fig.savefig(out/'m5g_exposure.png',dpi=150);plt.close(fig)
    sampling=json.loads(Path('reports/m5g_oracle_sampling_ceiling.json').read_text())['aggregate'];fig,ax=plt.subplots(figsize=(8,4))
    for phase in ['center','zero','end']:
        rr=[r for r in sampling if r['phase']==(r['stride']//2 if phase=='center'and r['stride']>1 else r['stride']-1 if phase=='end'else 0)];ax.plot([r['stride']for r in rr],[r['recall_1m']for r in rr],marker='o',label=phase)
    ax.set(xlabel='Oracle ray sampling stride',ylabel='Same-layer 1m recall',title='Dense renderer self-consistency vs sampled-ray ceiling');ax.legend();fig.tight_layout();fig.savefig(out/'m5g_sampling.png',dpi=160);plt.close(fig)
    pix=json.loads(Path('reports/m5g_pixel_test.json').read_text())['records'];pre=json.loads(Path('reports/m5g_pretrained_test.json').read_text());fig,ax=plt.subplots(1,2,figsize=(12,4))
    for r in pix:
        h=np.array(r['supported']['histogram']);tp=h[0,::-1].cumsum();fp=h[1,::-1].cumsum();ax[0].plot(tp/max(tp[-1],1),tp/np.maximum(tp+fp,1),label=f"{r['name']}/{r['seed']}")
    h=np.array(pre['results']['supported']['histogram']);tp=h[0,::-1].cumsum();fp=h[1,::-1].cumsum();ax[0].plot(tp/max(tp[-1],1),tp/np.maximum(tp+fp,1),'k--',label='COCO MaskRCNN');ax[0].set(xlabel='Recall',ylabel='Precision',title='Supported vehicle/person classes (cone ignored)');ax[0].legend(fontsize=5)
    for r in pix:
        ps=r['result']['pairs'];n=sum(p['n']for p in ps);nr=sum(p['removed_n']for p in ps);x=sum(p['present_hits']for p in ps)/max(n,1);y=sum(p['removed_fp']for p in ps)/max(nr,1);ax[1].scatter(x,y);ax[1].annotate(f"{r['name']}/{r['seed']}",(x,y),fontsize=5)
    ax[1].set(xlabel='Paired presence recall',ylabel='Removed local FPR',title='Locked validation thresholds');fig.tight_layout();fig.savefig(out/'m5g_pr_pairs.png',dpi=160);plt.close(fig)
    rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5g_test_pixels/index.json').read_text())['records'];choice=json.loads(Path('reports/m5g_pretest_freeze.json').read_text())['choices']['101'];model=next(r for r in pix if r['name']==choice['pixel']and r['seed']==101)
    for group in sorted({r['group']for r in rows}):
        rr=sorted([r for r in rows if r['group']==group],key=lambda r:r['configuration']);selected=[rr[i]for i in [0,2,5,7,8,11]];fig,ax=plt.subplots(len(selected),4,figsize=(16,13))
        for j,row in enumerate(selected):
            raw=Path(row['raw_frame']).parent.parent;frame=json.loads(Path(row['raw_frame']).read_text());ci=3 if row['configuration']==5 else 0;cam=row['camera_names'][ci];im=np.array(Image.open(raw/frame['cameras'][cam]['rgb']));d=pixel_label(row);pred=np.load(next(x['path']for x in model['saved']if x['sample_id']==row['sample_id']))['probability_u8'][ci]/256;z=np.load(next(x['path']for x in pre['saved']if x['sample_id']==row['sample_id']));pp=np.maximum(z['vehicle'][ci],z['pedestrian'][ci])/256
            ax[j,0].imshow(im);ax[j,0].set_title(f'{group}/{row["configuration"]} {cam}');ax[j,1].imshow(d['target'][ci],vmin=0,vmax=1);ax[j,1].set_title('Current target surface GT');ax[j,2].imshow(pred,cmap='magma',vmin=0,vmax=1);ax[j,2].set_title('HS RGB foreground');ax[j,3].imshow(pp,cmap='magma',vmin=0,vmax=1);ax[j,3].set_title('COCO supported foreground')
            for a in ax[j]:a.axis('off')
        fig.tight_layout();fig.savefig(out/f'm5g_pixels_{group}.png',dpi=130);plt.close(fig)
    atomic_json('reports/m5g_visualizations.json',dict(kind='offline_figures_actual_experiments',source_sha256=file_hash(__file__),files={str(p):file_hash(p)for p in out.glob('*.png')},selection='Alltestgroups fixedconfiglist firstseed, notbesttest.'))

if __name__=='__main__':main()
