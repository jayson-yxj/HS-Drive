"""Inspect full-ray interventions and camera-local failures from saved arrays."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.schema import CAMERA_IDS
from simulator.recorder import file_hash,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='reports/m4_bev_full_ray_interventions.json');p.add_argument('--output',default='reports/m4_bev_full_ray_figures');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);report=json.loads(Path(a.input).read_text());root=Path(report['artifacts'])
    for path in root.glob('*.npz'):
        with np.load(path,allow_pickle=False) as d:
            fig,axes=plt.subplots(6,4,figsize=(15,15))
            for i,camera in enumerate(CAMERA_IDS):
                axes[i,0].imshow(d['rgb'][i].transpose(1,2,0));axes[i,0].set_ylabel(camera)
                for col,key in enumerate(['depth_gt','depth_expected','entropy'],1):
                    value=np.ma.masked_where(~d['depth_valid'][i],d[key][i]);axes[i,col].imshow(value,vmin=0,vmax=1 if key=='entropy' else 80,cmap='turbo')
                for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
            for j,t in enumerate(['RGB','GT optical depth (0-80m)','Expected depth (0-80m)','Normalized entropy (0-1)']):axes[0,j].set_title(t)
            fig.suptitle(path.stem+' | first scene frame, GT-independent selection',fontsize=10);fig.tight_layout();fig.savefig(out/(path.stem+'_cameras.png'));plt.close(fig)
            keys=['original','depth_probability_shuffle','context_spatial_shuffle','oracle_gt_bin_valid_only','pixel_distribution_prior_valid_only']
            fig,axes=plt.subplots(1,6,figsize=(15,5))
            for ax,key in zip(axes,['GT',*keys]):
                value=d['road_gt'] if key=='GT' else d[key+'_road'];color=np.full((*value.shape,3),.25)
                color[d['road_valid']&(value>=.5)]=[.2,.75,.2];color[d['road_valid']&(value<.5)]=[.85,.1,.1]
                ax.imshow(color[::-1,::-1]);ax.set_title(key.replace('_','\n'),fontsize=8);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(path.stem+' | common v1 road mask; privileged/OOD diagnostics only',fontsize=10);fig.tight_layout();fig.savefig(out/(path.stem+'_road.png'));plt.close(fig)
    atomic_json(out/'manifest.json',{'input_sha256':file_hash(a.input),'figures':{p.name:file_hash(p) for p in out.glob('*.png')}})

if __name__=='__main__':main()
