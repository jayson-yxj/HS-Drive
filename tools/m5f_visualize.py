"""Static figures from frozen recorded results; no selection or training."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('reports/m5f_figures');out.mkdir(exist_ok=False)
    pix=json.loads(Path('reports/m5f_2d_presence_absence.json').read_text())['records'];train=json.loads(Path('reports/m5f_image_foreground_training.json').read_text())['records'];geo=json.loads(Path('reports/m5f_2d_3d_factorization.json').read_text())['records'];rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels/index.json').read_text())['records']
    fig,ax=plt.subplots(1,3,figsize=(16,4))
    for r in pix:
        h=np.array(r['histograms']['pooled/all']);tp=h[1,::-1].cumsum();fp=h[0,::-1].cumsum();ax[0].plot(tp/max(tp[-1],1),tp/np.maximum(tp+fp,1),label=f"{r['name']}/{r['seed']}")
        pairs=r['pairs'];n=sum(x['n'] for x in pairs);prob=[sum((x[k] or 0)*x['n'] for x in pairs)/max(n,1) for k in ['mean_empty','mean_present']];nr=sum(x['removed_n'] for x in pairs);prob+=[sum((x['mean_removed'] or 0)*x['removed_n'] for x in pairs)/max(nr,1)];ax[1].plot(['Empty','Present','Removed'],prob,marker='o',label=f"{r['name']}/{r['seed']}")
    for r in train:
        h=[v for v in r['history'] if 'loss'in v];ax[2].plot([v['step'] for v in h],[v['loss'] for v in h],label=f"{r['name']}/{r['seed']}")
    ax[0].set(xlabel='Recall',ylabel='Precision',title='New-position pixel PR (256-bin)');ax[1].set(ylabel='Mean probability',title='Same-pixel target change (known surfaces)');ax[2].set(xlabel='Step',ylabel='Balanced BCE',title='Actual short GPU training');ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5f_pixel_curves.png',dpi=160);plt.close(fig)
    model=next(r for r in pix if r['name']=='P8_fpn' and r['seed']==101) # predeclared first seed, not best test
    for group in sorted({r['group'] for r in rows}):
        rr=sorted([r for r in rows if r['group']==group],key=lambda r:r['configuration']);selected=[rr[0],rr[3],rr[7],rr[8],rr[-1]];fig,ax=plt.subplots(len(selected),3,figsize=(13,12))
        for j,row in enumerate(selected):
            raw=Path(row['raw_frame']).parent.parent;f=json.loads(Path(row['raw_frame']).read_text());cam=next((c for c in row['camera_names'] if c.lower() in ['front','cam_front']),row['camera_names'][0]);ci=row['camera_names'].index(cam);im=np.array(Image.open(raw/f['cameras'][cam]['rgb']));d=pixel_label(row);rec=next(x for x in model['predictions'] if x['sample_id']==row['sample_id']);p=np.load(rec['path'])['probability_u8'][ci]/256
            ax[j,0].imshow(im);ax[j,0].set_title(f'{group} config{row["configuration"]} {cam}');overlay=np.zeros((384,640,4));overlay[d['valid'][ci]]=[0,.4,1,.2];overlay[d['target'][ci]]=[1,0,0,.8];ax[j,1].imshow(im);ax[j,1].imshow(overlay);ax[j,1].set_title('GT: red target / blue known nontarget');ax[j,2].imshow(p,cmap='magma',vmin=0,vmax=1);ax[j,2].set_title('P8 FPN seed101 RGB-only probability')
            for a in ax[j]:a.axis('off')
        fig.tight_layout();fig.savefig(out/f'm5f_pixels_{group}.png',dpi=130);plt.close(fig)
        row=rr[3];chosen=[r for r in geo if r['sample_id']==row['sample_id'] and r['name']=='P8_fpn' and r['seed']==101];fig,ax=plt.subplots(4,5,figsize=(14,10))
        for k,r in enumerate(chosen):
            d=np.load(r['path'])
            for z in range(4):
                ax[z,k].imshow(d['score'][z],origin='lower',extent=(-30,30,-20,60),vmin=0,vmax=1,cmap='magma');ax[z,k].set_title(f'{r["mode"][0]} layer{z}')
            if k==0:
                for z in range(4):ax[z,4].imshow(d['target'][z]&d['valid'][z],origin='lower',extent=(-30,30,-20,60),vmin=0,vmax=1);ax[z,4].set_title(f'Rendered reference layer{z}')
        fig.suptitle(f'{group} configuration3: shared UV/mask four-way event scores, not occupancy');fig.tight_layout();fig.savefig(out/f'm5f_fourway_{group}.png',dpi=140);plt.close(fig)
    digest=json.loads(Path('reports/m5f_results_digest.json').read_text());fig,ax=plt.subplots(1,2,figsize=(13,4))
    for r in digest['pixels']:
        x=['5_10','10_15','15_20','20_35'];ys=[r['actor_strata']['distance_'+k]['detection_rate'] for k in x];ax[0].plot(x,[np.nan if y is None else y for y in ys],marker='o',label=f"{r['name']}/{r['seed']}")
    for name in ['S16_frozen','P8_frozen','P8_fpn']:
        for seed in [101,102]:
            rr=[r for r in digest['geometry'] if r['name']==name and r['seed']==seed];ax[1].plot([r['mode'][0] for r in rr],[r['actor_tolerant_recall']['1.0'] for r in rr],marker='o',label=f'{name}/{seed}')
    ax[0].set(title='2D actor-camera detection by distance',ylabel='Detected / eligible',xlabel='Actor center distance m');ax[1].set(title='Actor visible surface BEV recall (1m)',ylabel='Recall',xlabel='A/B/C/D (validation thresholds)');ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'m5f_actor_geometry.png',dpi=160);plt.close(fig)
    atomic_json('reports/m5f_visualization_index.json',dict(kind='offline_visualization_actual_predictions',files={str(p):file_hash(p) for p in out.glob('*.png')},source_sha256=file_hash(__file__),selection='All3newgroups; fixedconfig3/7/8 andfirstseed101, notbesttest. Figures distinguish label and RGBinference; no syntheticphoto.'))

if __name__=='__main__':main()
