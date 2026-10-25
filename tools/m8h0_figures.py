"""Geometry-only maps: no simulated images and no generated visual substitutes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from tools.m8h0_common import *
from datasets.m8h0_geometry import footprint
OUT=Path('reports/m8h0_figures')
def main():
 vehicles=json.loads((ROOT/'vehicle_inventory_full.json').read_text())['records'];people=json.loads((ROOT/'navigation_inventory_full.json').read_text())['records'];templates=json.loads(Path('reports/m8h0_curated_template_candidates.json').read_text());env=json.loads((ROOT/'environment_inventory.json').read_text())['objects'];files=[]
 fig,ax=plt.subplots(figsize=(9,9))
 for rows,marker,label in [(vehicles,'s','Native vehicle'),(people,'o','Native navigation')]:
  for passed in [False,True]:
   rs=[r for r in rows if (r['geometry']['status']=='GeometryCandidateProxySupported')==passed and r.get('transform')];p=np.array([r['transform']['xyz'][:2] for r in rs]);ax.scatter(p[:,0],p[:,1],marker=marker,s=27,c='green' if passed else 'crimson',label=label+(' proxy supported' if passed else ' unresolved'))
 for t in templates['templates']:
  p=t['ego']['transform']['xyz'];ax.add_patch(Circle(p[:2],40,fill=False,color='royalblue',alpha=.5));ax.text(p[0],p[1],t['id'][-2:],fontsize=11,color='blue')
 ax.set_aspect('equal');ax.set(xlabel='CARLA world X [m]',ylabel='CARLA world Y [m]',title='Town02: all 101 spawn + 60 navigation results\nFour fixed template Ego centers; circles are 40m nominal observation regions');ax.legend(fontsize=8);fig.tight_layout();p=OUT/'m8h0_native_sources.png';fig.savefig(p,dpi=160);plt.close(fig);files.append(asset(p))
 fig,axs=plt.subplots(2,2,figsize=(14,12))
 for t,w,ax in zip(templates['templates'],templates['occlusion_warnings'],axs.flat):
  center=np.array(t['ego']['transform']['xyz'][:2]);limit=43;warn_ids={r['entity_id'] for cat in ['vehicle','person'] for v in w[cat] for r in v['possible_static_occluders']}
  for e in env:
   if np.linalg.norm(np.array(e['center_world_m'])[:2]-center)>70:continue
   if e['semantic_tags'][0] in [1,2,10,11,23,24,25,26,27]:continue
   ax.add_patch(Polygon(footprint(e),fill=True,facecolor='orange' if e['entity_id'] in warn_ids else 'gray',edgecolor='orange' if e['entity_id'] in warn_ids else 'gray',alpha=.22))
  for name,color in [('ego','blue'),('vehicle','red'),('person','green')]:
   r=t[name];box=r['geometry']['hypothetical_vertical_only_box'];ax.add_patch(Polygon(footprint(box),fill=True,color=color,alpha=.7));pt=np.array(box['center_world_m'])[:2];ax.text(pt[0],pt[1],r['id'],fontsize=8,color=color)
   m=np.array(r['geometry']['hypothetical_actor_matrix']);ax.arrow(*pt,*(m[:2,0]*3),width=.1,color=color)
   if name!='ego':ax.plot([center[0],pt[0]],[center[1],pt[1]],'--',c=color,alpha=.4)
  ax.add_patch(Circle(center,20,fill=False,color='blue',ls=':'));ax.add_patch(Circle(center,40,fill=False,color='blue',ls=':'));ax.set(xlim=(center[0]-limit,center[0]+limit),ylim=(center[1]-limit,center[1]+limit),xlabel='X [m]',ylabel='Y [m]',title=t['id']+' | orange: possible OBB occluder');ax.set_aspect('equal')
 fig.suptitle('Frozen candidate templates: hypothetical support alignment, no Actor or RGB\nStatic OBB is conservative; dashed view direction does not certify visibility');fig.tight_layout();p=OUT/'m8h0_template_review.png';fig.savefig(p,dpi=160);plt.close(fig);files.append(asset(p))
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for rows,ax,title in [(vehicles,axs[0],'Native vehicle initial OBB bottom'),(people,axs[1],'Navigation point used as pedestrian center (diagnostic)')]:
  gaps=[np.mean(r['geometry']['native_initial_support_gaps_m']) for r in rows if r['geometry'].get('native_initial_support_gaps_m')];ax.hist(gaps,bins=15);ax.axvline(0,c='black');ax.axvspan(-.03,.12,color='green',alpha=.15,label='frozen gap band');ax.set(title=title,xlabel='Mean initial bottom minus queried support [m]',ylabel='Candidates');ax.legend()
 fig.suptitle('Initial native position is not a measured final Actor pose');fig.tight_layout();p=OUT/'m8h0_initial_height_semantics.png';fig.savefig(p,dpi=160);plt.close(fig);files.append(asset(p))
 write('reports/m8h0_visualizations.json',dict(**provenance('OfflinePlotsOfActualMapQueries'),files=files,no_rendered_images=True))
if __name__=='__main__':main()
