"""Representative train rays: actual distributions, bins and geometric support."""
import json
from pathlib import Path
import numpy as np,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5k_common import rows,write,asset
from tools.m5k_alignment import sample_gt
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb
from models.m5k_geometry import resample_probability
from models.m5e_paths import geometry
from training.runtime import prepare_cuda

@torch.no_grad()
def main():
 prepare_cuda();torch.set_num_threads(2);net=encoder_load();selected=[]
 # Deterministic first strict associated far ray per new Train position. No selection by error.
 for group in sorted({r['group']for r in rows('train')}):
  for row in [r for r in rows('train')if r['group']==group]:
   d=pixel_label(row);gt=sample_gt(d,8);x=inputs(row);c=cache_p8(net,x);g=geometry(net,c,gt['depth']);m=gt['actor']&gt['valid']&g['inside']&(gt['depth']>=20)&(gt['depth']<35)
   if not m.any():continue
   _,cam,v,u=torch.nonzero(m)[0].tolist();c16=cache_rgb(net,x);p=c['probs'][0,cam,:,v,u].cpu().numpy();b=resample_probability(c16['probs'],(48,80))[0,cam,:,v,u].cpu().numpy();depth=float(gt['depth'][0,cam,v,u]);z=net.splat.depth_centers.cpu().numpy();sp=(g['same_layer']&(g['xy']<=1))[0,cam,:,v,u].cpu().numpy();q=g['points'][0,cam,:,v,u].cpu().numpy();point=g['gt'][0,cam,v,u].cpu().numpy();fig,axes=plt.subplots(1,3,figsize=(14,4));im=x['images'][0,cam].permute(1,2,0).cpu().numpy();axes[0].imshow(im);axes[0].plot(u*8+4,v*8+4,'r+');axes[0].set_title(f'{group} {row["camera_names"][cam]}');axes[1].plot(z,p,'o-',ms=2,label='Head P8');axes[1].plot(z,b,'o-',ms=2,label='S16->P8');axes[1].axvline(depth,color='red',label='GT optical Z');axes[1].scatter(z[sp],p[sp],s=50,facecolors='none',edgecolors='green',label='same-layer1m bin');axes[1].legend(fontsize=7);axes[1].set_title('Matched ray depth probability');axes[2].scatter(q[:,0],q[:,1],s=p*1000+1,c=z,cmap='viridis');axes[2].plot(point[0],point[1],'r*',ms=13);axes[2].set_aspect('equal');axes[2].set_title('XY bin centers; area ~ P8 mass');axes[2].set_xlabel('HS x forward m');axes[2].set_ylabel('HS y left m');fig.tight_layout();path=Path('reports/m5k_figures')/(group+'_m5k_ray.png');fig.savefig(path,dpi=150);plt.close(fig);selected.append(dict(group=group,sample_id=row['sample_id'],camera=cam,uv=[u*8+4,v*8+4],gt_z=depth,supported_bins=int(sp.sum()),**asset(path)));break
 write('reports/m5k_distribution_visualizations.json',dict(kind='actualGPU_readonly_firsteligibleTrainray_eachposition',records=selected,limits='No best-case or error-based ray selection; GT supervisionevaluationonly. Coarse endpoint wrongcell not physicalempty.'))
if __name__=='__main__':main()
