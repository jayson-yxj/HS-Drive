"""Actual raw settling and failed support evidence plots; no fabricated RGB."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m8h1_common import *

def main():
 reviews=[json.loads(p.read_text()) for p in sorted((ROOT/'capture').glob('*/ego_present_review.json'))];diag=json.loads(Path('reports/m8h1_support_diagnostic.json').read_text());files=[]
 fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
 for i,r in enumerate(reviews):
  rows=[json.loads(s) for s in Path(r['settling']['trace']['path']).read_text().splitlines()];times=np.array([x['timestamp_s'] for x in rows]);times-=times[0];pos=np.array([x['pose']['xyz'] for x in rows]);vel=np.array([x['velocity_mps'] for x in rows]);label='template '+['00','02'][i]
  axs[0].plot(times,pos[:,2],label=label);axs[1].plot(times,np.linalg.norm(vel,axis=1),label=label);axs[2].plot(times,np.linalg.norm(pos[:,:2]-np.array(r['initial_pose']['xyz'][:2]),axis=1)*1000,label=label)
 axs[0].set(ylabel='Actor Z [m]',title='Raw CARLA physics settling: native Z0.5m, no physics disabling');axs[1].set(ylabel='Raw speed [m/s]');axs[1].axhline(.02,c='red',ls='--',label='stable-window speed limit');axs[2].set(ylabel='Native XY drift [mm]',xlabel='Time from first observed snapshot [s]')
 for ax in axs:ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();p=Path('reports/m8h1_figures/m8h1_ego_settling.png');fig.savefig(p,dpi=160);plt.close(fig);files.append(asset(p))
 fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
 for ax,r in zip(axs,diag['rows']):
  xs=np.arange(15);ground=np.array([p['pre_actor_selected_surface']['xyz'][2] for p in r['point_rows']]);bottom=np.array([p['current_bottom'][2] for p in r['point_rows']]);start=np.array([p['current_query']['start'][2] for p in r['point_rows']]);ax.plot(xs,ground,'g-',label='pre-Actor surface (not current support)');ax.plot(xs,bottom,'b.-',label='actual OBB floor');ax.plot(xs,start,'rx-',label='frozen downward-ray start');ax.set(xlabel='Bottom sample index',ylabel='Z [m]',title=r['template']+' | current hits 0/15');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.suptitle('Downward rays start below prior surface; absence of hit is not proof of floating');fig.tight_layout();p=Path('reports/m8h1_figures/m8h1_support_ray_gap.png');fig.savefig(p,dpi=160);plt.close(fig);files.append(asset(p))
 write('reports/m8h1_visualizations.json',dict(**provenance('OfflinePlotsOfActualPhysicsAndGeometry'),files=files,RGB_generated=False,model_outputs=False))
if __name__=='__main__':main()
