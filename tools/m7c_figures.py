"""Standalone evidence plots from sealed offline records."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m7c_common import read

def main():
 out=Path('reports/m7c_figures');out.mkdir(exist_ok=False)
 latency=read('reports/m7c_planning_latency.json')['results'];p=latency['equivalence_pair'];names=list(p);fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(names))
 for off,k in [(-.2,'p50'),(0,'p95'),(.2,'max')]:ax.bar(x+off,[p[n][k] for n in names],width=.2,label=k)
 ax.axhline(200,color='r',ls='--',label='200ms');ax.axhline(150,color='grey',ls=':',label='P95 target150ms');ax.set_xticks(x,names,rotation=12);ax.set(ylabel='Offline core (ms)',title='Exact-input cache reuse: warm repeat is not moving-ego runtime');ax.legend();fig.tight_layout();fig.savefig(out/'m7c_latency.png',dpi=150);plt.close(fig)
 d=read('reports/m7c_candidate_coverage.json')['per_split'];keys=list(d);fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(keys))
 before=[sum(v for n,v in d[k]['baseline_choices'].items() if n not in ['None','D_stop']) for k in keys];after=[sum(v for n,v in d[k]['new_choices'].items() if n not in ['None','D_stop']) for k in keys]
 ax.bar(x-.18,before,.36,label='Frozen candidate library');ax.bar(x+.18,after,.36,label='+ Single slow family');ax.set_xticks(x,keys);ax.set(ylabel='Selected moving candidate scenes',title='Limited6s progress, not executed route completion');ax.legend();fig.tight_layout();fig.savefig(out/'m7c_coverage.png',dpi=150);plt.close(fig)
 r=next(r for r in read('reports/m7c_fresh_final_frozen.json')['records'] if r['scene_id']=='fresh_final_road28_braking_1');fig,ax=plt.subplots(1,2,figsize=(11,4))
 for e in r['evaluations']:
  if e['name'] not in ['A_keep_route','D_stop','E_slow_follow']:continue
  p=np.array(e['candidate']['points']);t=e['candidate']['times_s'];distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1))];label=e['name']+(' accepted' if e['accepted'] else ' rejected');ax[0].plot(t,p[:,3],label=label);ax[1].plot(t,distance,label=label)
 ax[0].set(xlabel='Snapshot-relative seconds',ylabel='Planned speed(m/s)');ax[1].set(xlabel='Snapshot-relative seconds',ylabel='Planned progress(m)');ax[0].legend(fontsize=8);fig.suptitle('Fresh road28: unchanged hypotheses, conservative stop reserve');fig.tight_layout();fig.savefig(out/'m7c_slow_follow.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
