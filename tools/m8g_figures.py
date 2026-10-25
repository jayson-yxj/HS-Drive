"""Plot frozen evidence, not generated illustrative substitutes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tools.m8g_common import ROOT,asset,write,provenance
OUT=Path('reports/m8g_figures');OUT.mkdir(exist_ok=True)
def load(n):return json.loads(Path('reports/m8g_'+n+'.json').read_text())
def draw_box(ax,box,color,alpha=.15):
 p=np.asarray(box['corners_world_m']);faces=[[0,1,3,2],[4,5,7,6],[0,4,5,1],[2,6,7,3],[0,2,6,4],[1,3,7,5]]
 ax.add_collection3d(Poly3DCollection([p[f] for f in faces],facecolors=color,edgecolors=color,alpha=alpha))
def main():
 artifacts=[]
 # 3D locations of ALL rejected rays for one phase per site, plus hypothetical box.
 none=load('none_hit_analysis')['records'];trees=load('overhead_clearance')['records']
 fig=plt.figure(figsize=(15,5))
 for i,(records,title) in enumerate([(none,'NONE: trigger-volume hypothesis, not mesh'),(trees,'Canopy: high hits do not prove full clearance')]):
  ax=fig.add_subplot(1,2,i+1,projection='3d');rs=[r for r in records if r['group'].endswith('00') and r['phase']=='present'];b=rs[0]['hypothetical_box'];draw_box(ax,b,'royalblue',.25)
  pts=np.array([r['hit']['xyz'] for r in rs]);ax.scatter(*pts.T,c='red',s=25,label='overlap-channel hits')
  gr=np.array([r['all_ray_hits']['decision']['selected']['xyz'] for r in rs]);ax.scatter(*gr.T,c='green',s=15,label='selected ground')
  center=np.array(b['corners_world_m']).mean(0);ax.set_xlim(center[0]-3,center[0]+3);ax.set_ylim(center[1]-3,center[1]+3);ax.set_zlim(0,9 if i else 3);ax.set_xlabel('CARLA world X [m]');ax.set_ylabel('CARLA world Y [m]');ax.set_zlabel('Z [m]');ax.set_title(title,fontsize=10);ax.legend(fontsize=8)
 fig.tight_layout();p=OUT/'m8g_ray_body_3d.png';fig.savefig(p,dpi=160);plt.close(fig);artifacts.append(asset(p))
 # Lane conflict visualized in authored road coordinates, retaining nearest wrong-lane query.
 lanes=load('lane_boundary_differential')['records'];rs=[r for r in lanes if r['group'].endswith('00') and r['key']=='target_2' and r['phase']=='present']
 fig,ax=plt.subplots(figsize=(8,4));x=np.array([r['independent']['s'] for r in rs]);y=np.array([r['independent']['t'] for r in rs]);ax.axhspan(.1,3.9,color='lightgreen',label='declared Driving lane, 0.10m inset');ax.axhspan(4,4.3,color='gray',alpha=.4,label='neighbor narrow lane');ax.scatter(x,y,c='red',label='15 Any=False rejections');ax.axhline(4.15,color='black',ls='--',label='Any=True selects lane 2 center');ax.set(xlabel='OpenDRIVE s [m]',ylabel='OpenDRIVE lateral t [m]',title='Road 12 cone: authored lane contains points; Any query disagrees');ax.legend(fontsize=8);fig.tight_layout();p=OUT/'m8g_lane_boundary.png';fig.savefig(p,dpi=160);plt.close(fig);artifacts.append(asset(p))
 # Every candidate; no rejected samples hidden.
 rows=json.loads((ROOT/'scan_all_candidates.json').read_text())['records'];fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True)
 cats=['vehicle','person','cone','barrier'];ds=[10,15,20,25,30,-10,-15,-20,-25,-30]
 for i,ax in enumerate(axes):
  group='m8g_development_%02d'%i;mat=np.array([[next(r['geometry_pass'] for r in rows if r['group']==group and r['category']==c and r['distance_m']==d) for d in ds] for c in cats],int)
  ax.imshow(mat,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto');ax.set_xticks(range(10),ds);ax.set_yticks(range(4),cats);ax.set_xlabel('Signed distance along declared lane [m]');ax.set_title(group+' | red reject, green geometry proxy')
  for y in range(4):
   for x in range(10):ax.text(x,y,'PASS' if mat[y,x] else 'REJ',ha='center',va='center',fontsize=7)
 fig.suptitle('80 fixed candidates, both Present and Shifted required; no RGB Smoke');fig.tight_layout();p=OUT/'m8g_scan_matrix.png';fig.savefig(p,dpi=160);plt.close(fig);artifacts.append(asset(p))
 # Historic image crop around NONE point, explicit predicted pixel and ray depth mismatch.
 from PIL import Image
 r=none[0];ev=r['render_evidence'][0];old='/root/gpufree-data/hs-drive-experiments/m8f-grounded-quality/capture/'+r['group']+'/rgb/empty_'+ev['camera']+'.png';img=Image.open(old)
 fig,ax=plt.subplots(figsize=(10,5));ax.imshow(img);u,v=ev['pixel'];ax.scatter([u],[v],marker='x',s=100,color='red');ax.set_title('Historical RGB only: NONE at projected pixel; depth residual %.2fm\nNo new RGB and no physical-free-space conclusion'%ev['depth_residual_m']);ax.axis('off');fig.tight_layout();p=OUT/'m8g_none_historical_rgb.png';fig.savefig(p,dpi=160);plt.close(fig);artifacts.append(asset(p))
 # True footprint distances recovered from raw history.
 reg=load('historical_preflight_index')['regression'];fig,ax=plt.subplots(figsize=(8,4));labels=[r['group'][-2:]+'/'+r['target'] for r in reg];dist=[r['actual']['actor']['distance_m'] for r in reg];ax.bar(labels,dist);ax.axhline(22,c='red',ls='--',label='20m + 2m uncertainty');ax.set(ylabel='Exact XY polygon distance [m]',title='Raw-frame target footprint regression (not Ego-center distance)');ax.legend();fig.tight_layout();p=OUT/'m8g_history_footprints.png';fig.savefig(p,dpi=160);plt.close(fig);artifacts.append(asset(p))
 write('reports/m8g_visualizations.json',dict(**provenance('OfflinePlotsOfActualEvidence'),files=artifacts,synthetic_images=False,new_RGB=0))
if __name__=='__main__':main()
