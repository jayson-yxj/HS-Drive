"""Standalone plots from actual frozen measurements; no synthetic result graphics."""
import json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5l_common import write,asset

def main():
 out=Path('reports/m5l_figures');out.mkdir(exist_ok=True);assets=[];support=json.loads(Path('reports/m5l_sampling_failure_analysis.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,split in zip(axes,['train','val']):
  person=next(r for r in support['splits'][split]['types']if r['type']=='person');im=ax.imshow(person['p8_phases']);ax.scatter([4],[4],marker='x',color='red');ax.set_title(split+' pedestrian strict P8 phase support');ax.set_xlabel('u phase');ax.set_ylabel('v phase');fig.colorbar(im,ax=ax)
 fig.tight_layout();p=out/'m5l_pedestrian_phases.png';fig.savefig(p,dpi=160);assets.append(asset(p));plt.close(fig)
 training=json.loads(Path('reports/m5l_vehicle_training.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,seed in zip(axes,[101,102]):
  for r in training['records']:
   if r['seed']!=seed:continue
   vals=[v for v in r['history']if 'validation_loss'in v];ax.plot([v['step']for v in vals],[v['validation_loss']for v in vals],'o-',label=r['mode'])
  ax.set_title(f'Fixed budget Val loss seed{seed}');ax.set_xlabel('step');ax.legend()
 fig.tight_layout();p=out/'m5l_training.png';fig.savefig(p,dpi=160);assets.append(asset(p));plt.close(fig)
 objects=json.loads(Path('reports/m5l_object_2p5d_baseline.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,seed in zip(axes,[101,102]):
  rr=[r for r in objects['comparisons']if r['seed']==seed]
  for method in ['plane_center','dense_expected_center','dense_mode_center']:ax.plot([r['split']for r in rr],[r['errors'][method]['p50']for r in rr],'o-',label=method)
  ax.set_title(f'Conditional matched objects seed{seed}');ax.set_ylabel('Median XY error m');ax.legend(fontsize=8)
 fig.tight_layout();p=out/'m5l_object_xy.png';fig.savefig(p,dpi=160);assets.append(asset(p));plt.close(fig)
 shadow=json.loads(Path('reports/m5l_online_shadow.json').read_text());raw=json.loads(Path(shadow['raw']['path']).read_text());fig,axes=plt.subplots(1,2,figsize=(12,4))
 for r in raw['phases']:axes[0].plot(r['raw_tick_ms'],label=r['phase'],alpha=.7)
 axes[0].set_ylabel('CARLA tick wall ms');axes[0].set_xlabel('measured tick');axes[0].legend()
 rr=[r for r in raw['outputs']if 'end_to_end_ms'in r];axes[1].plot([r['queue_age_ms']for r in rr],label='queue age');axes[1].plot([r['end_to_end_ms']for r in rr],label='callback to output');axes[1].axhline(250,color='red',ls='--');axes[1].legend();axes[1].set_ylabel('ms');fig.tight_layout();p=out/'m5l_shadow_timing.png';fig.savefig(p,dpi=160);assets.append(asset(p));plt.close(fig)
 for split in ['val','test']:
  data=json.loads(Path('reports/m5l_adaptive_'+split+'.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
  for ax,seed in zip(axes,[101,102]):
   for method in ['adaptive','uniform','region_uniform']:
    rr=[r for r in data['curve']if r['seed']==seed and r['method']==method];ax.plot([r['fpr']for r in rr],[r['recall1']for r in rr],'.-',label=method)
   ax.set_xlabel('Confirmed background FPR');ax.set_ylabel('Same-layer1m recall');ax.set_title(f'{split} seed{seed}');ax.legend(fontsize=8)
  fig.tight_layout();p=out/f'm5l_adaptive_{split}.png';fig.savefig(p,dpi=160);assets.append(asset(p));plt.close(fig)
 write('reports/m5l_visualizations.json',dict(kind='plots_actual_server_outputs',assets=assets,limits='Object plots conditionalmatchedsubset, not allobjectsor voxelRecall. Adaptive curves descriptive; workpoints alreadyfrozen. Shadow rawwallmetrics notsmoothed.'))
if __name__=='__main__':main()
