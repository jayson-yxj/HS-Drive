"""Audit RGB, reliable actor support and unmasked changes on frozen test predictions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tools.m5e_common import sidecars


def main():
    dest=Path('reports/m5e_figures/m5e_rgb_support_change.png')
    if dest.exists():raise FileExistsError(dest)
    root='/root/gpufree-data/hs-drive-data/processed/m5e_test_index';rows=json.loads(Path(root,'index.json').read_text())['samples'];ref=sidecars(root,rows,'m5e_surface_index.json');source=json.loads(Path('reports/m5e_presence_absence.json').read_text());model=next(r for r in source['records'] if r['name']=='B_presence' and r['seed']==91);score=np.load(model['predictions'])['score'];group=sorted({r['group'] for r in rows})[0];base=next(i for i,r in enumerate(rows) if r['group']==group and r['configuration']==0);cfg=[0,2,4,6,15];fig,axes=plt.subplots(4,len(cfg),figsize=(16,12));extent=(-30,30,-20,60)
    for j,c in enumerate(cfg):
        i=next(i for i,r in enumerate(rows) if r['group']==group and r['configuration']==c);row=rows[i];f=json.loads(Path(row['raw_frame']).read_text());raw=Path(row['raw_frame']).parent.parent;img=Image.open(raw/f['cameras']['front']['rgb'])
        axes[0,j].imshow(img);axes[0,j].axis('off');axes[0,j].set_title(f'{group} cfg{c}')
        for a in axes[1:,j]:a.set(xlim=(-8,8),ylim=(0,40),xlabel='Y left m',ylabel='X forward m')
        m=ref[i]['actor_masks'].any(0)[2];axes[1,j].imshow(np.ma.masked_where(~m,m),origin='lower',extent=extent,vmin=0,vmax=1,cmap='Greens');axes[1,j].set_title('Reliable Actor surface only')
        im=axes[2,j].imshow(score[i,2],origin='lower',extent=extent,cmap='magma',vmin=0,vmax=1);axes[2,j].set_title('B_presence91 raw probability')
        de=axes[3,j].imshow(score[i,2]-score[base,2],origin='lower',extent=extent,cmap='RdBu_r',vmin=-.3,vmax=.3);axes[3,j].set_title('Probability minus Empty')
    fig.colorbar(im,ax=axes[2].tolist(),shrink=.65,label='Probability');fig.colorbar(de,ax=axes[3].tolist(),shrink=.65,label='Delta, clipped display ±0.3');fig.suptitle('Frozen new test; height [1,2.5)m; Unknown is not negative; GT only audit overlay');fig.subplots_adjust(top=.94,hspace=.45,wspace=.4);fig.savefig(dest,dpi=150)

if __name__=='__main__':main()
