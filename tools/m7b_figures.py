"""Standalone plots from frozen results, no new scenario/model selection."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m7b_common import read


def main():
    out=Path('reports/m7b_figures');out.mkdir(exist_ok=False)
    data=read('reports/m7b_fresh_final_frozen.json');r=next(x for x in data['records'] if x['scene_id']=='fresh_final_road2_braking_0')
    fig,ax=plt.subplots(1,2,figsize=(12,5));route=np.array(r['request']['route']);origin=route[0,:2]
    ax[0].plot(*(route[:,:2]-origin).T,'k:',label='Road centerline')
    for e in r['candidate_evaluations']:
        if e['name'] not in ['A_keep_route','D_stop']:continue
        p=np.array(e['candidate']['points']);ax[0].plot(*(p[:,:2]-origin).T,label=e['name']+(' robust' if e['robust_accepted'] else ' rejected'))
    for h in r['hypotheses']:
        if h['name'] not in ['constant_speed_on_route','braking_-3.0_delay0.0','acceleration_1.5_delay0.0']:continue
        p=np.array(h['poses']);ax[0].plot(*(p[:,:2]-origin).T,'--',alpha=.7,label=h['name'])
    ref=r['evaluator_only_references']['lead'];p=np.array(ref['poses']);ax[0].plot(*(p[:,:2]-origin).T,'r-',label='Evaluator braking reference')
    ax[0].set_aspect('equal');ax[0].set(xlabel='Relative world X (m)',ylabel='Relative world Y (m)',title='Fresh road 2: arc placement and finite futures');ax[0].legend(fontsize=7)
    for h in r['hypotheses']:
        p=np.array(h['poses']);ax[1].plot(h['times'],p[:,3],alpha=.35)
    ax[1].plot(ref['times'],p[:,3] if False else np.array(ref['poses'])[:,3],'k',lw=3,label='Evaluator only')
    ax[1].set(xlabel='Time since original snapshot (s)',ylabel='Actor speed (m/s)',title='Finite motion hypotheses; not behavior probabilities');ax[1].legend();fig.tight_layout();fig.savefig(out/'m7b_motion_hypotheses.png',dpi=150);plt.close(fig)
    result=read('reports/m7b_robust_candidate_evaluation.json')['per_split'];keys=list(result);fig,ax=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(keys))
    for offset,method in [(-.18,'A_CV'),(.18,'B_hypotheses')]:
        ax[0].bar(x+offset,[len(result[k][method]['accepted_reference_conflicts']) for k in keys],width=.36,label=method)
        ax[1].bar(x+offset,[result[k][method]['selected_scenes'] for k in keys],width=.36,label=method)
    for a in ax:a.set_xticks(x,keys,rotation=20);a.legend()
    ax[0].set(title='Accepted candidates with reference collision',ylabel='Count');ax[1].set(title='Selected scenes (includes planning stops)',ylabel='Count');fig.tight_layout();fig.savefig(out/'m7b_risk_coverage.png',dpi=150);plt.close(fig)
    lat=read('reports/m7b_planning_latency.json')['per_split'];fig,ax=plt.subplots(figsize=(8,4))
    for offset,quantile in [(-.18,'p50'),(.18,'p95')]:ax.bar(x+offset,[lat[k]['stage_ms']['B_hypotheses'][quantile] for k in keys],width=.36,label=quantile)
    ax.axhline(200,color='red',ls='--',label='5Hz budget');ax.set_xticks(x,keys,rotation=15);ax.set(ylabel='Offline component cost (ms)',title='Not an online runtime certification');ax.legend();fig.tight_layout();fig.savefig(out/'m7b_latency.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
