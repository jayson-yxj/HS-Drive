"""Supplementary descriptive C-spread and per-camera examples; no model selection."""
import json
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5j_common import rows,write
from tools.m5f_common import input_path,pixel_label
from datasets.labels import depth_bin_edges
from simulator.recorder import file_hash

def main():
    dest=Path('reports/m5j_extra_figures');dest.mkdir(exist_ok=False);idx=json.loads(Path('runs/m5j_test/index.json').read_text())['records'];rr={r['sample_id']:r for r in idx};ex=json.loads(Path('reports/m5j_evidence_attribution_test.json').read_text())['examples'];r=max([r for r in ex if r['seed']==101 and r['domain']=='background'],key=lambda r:r['C_diffuse_ray_count']);fig,ax=plt.subplots(1,2,figsize=(13,4));edges=depth_bin_edges();centers=(edges[:-1]+edges[1:])/2
    for ray in r['top_rays'][:5]:ax[0].plot(centers,ray['probabilities'],label=f"{ray['camera']} {ray['pixel_uv']} cells={ray['depth_cell_count']}")
    ax[0].legend(fontsize=6);ax[0].set(xlabel='Optical depth m',ylabel='Probability',title='C: multiple locations from each unchanged ray');d=np.load(rr[r['sample_id']]['cache']);ax[1].imshow(d['101_raw'].max(0),origin='lower',cmap='magma',vmin=0,vmax=1);ax[1].scatter([r['cell'][2]],[r['cell'][1]],color='cyan');ax[1].set_title(r['group']+' source voxel; maxheight display');fig.tight_layout();fig.savefig(dest/'m5j_C_spread.png',dpi=140);plt.close(fig)
    for group in sorted({r['group']for r in idx}):
        row=next(r for r in idx if r['group']==group and r['configuration']==8);d=np.load(row['cache']);labels=pixel_label(row);inp=json.loads(input_path(row).read_text());fig,ax=plt.subplots(3,6,figsize=(18,7))
        for c,cam in enumerate(row['camera_names']):
            rgb=cv2.cvtColor(cv2.imread(inp['images'][c]['path']),cv2.COLOR_BGR2RGB);ax[0,c].imshow(rgb);ax[0,c].set_title(cam);ax[1,c].imshow(d['101_pixel'][c],cmap='magma',vmin=0,vmax=1);ax[2,c].imshow(np.where(labels['valid'][c],labels['target'][c].astype(float),np.nan),vmin=0,vmax=1)
        for a in ax.flat:a.axis('off')
        fig.suptitle(group+' fixed lateral pedestrian configuration8: RGB / Mobile101 / reliableGT');fig.tight_layout();fig.savefig(dest/(group+'_m5j_cameras.png'),dpi=140);plt.close(fig)
    write('reports/m5j_extra_visualizations.json',dict(kind='postfreeze_descriptive_fixed_examples',files={str(p):file_hash(p)for p in dest.iterdir()},spread_example=r,source_sha256=file_hash(__file__),no_reselection=True))
if __name__=='__main__':main()
