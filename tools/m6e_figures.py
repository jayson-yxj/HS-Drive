"""Research plots preserve raw transitions and show commanded versus applied."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(name):return [json.loads(s) for s in Path('reports/m6e_trials',name,'trace.jsonl').read_text().splitlines()]

def main():
    root=Path('reports/m6e_figures');root.mkdir(exist_ok=False)
    fig,axs=plt.subplots(4,1,figsize=(13,11),sharex=True)
    for arm in ['A','B','C']:
        r=rows(f'P0_development_p27_s301_{arm}');t=np.array([q['measurement']['t'] for q in r]);t-=t[0];v=np.array([q['signed']['v_forward_mps'] for q in r]);a=np.diff(v)/.05;j=np.diff(a)/.05
        axs[0].plot(t,v,label=arm);axs[1].plot(t[1:],a,label=arm);axs[2].plot(t[2:],j,label=arm);axs[3].plot(t,[q['measurement']['applied']['brake'] for q in r],label=arm)
    for ax,label in zip(axs,['Raw signed speed m/s','Raw acceleration m/s²','Raw jerk m/s³','Applied brake']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.3)
    axs[2].axhline(5,color='red',ls='--');axs[-1].set_xlabel('Simulation seconds');fig.tight_layout();fig.savefig(root/'m6e_interfaces.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(4,1,figsize=(13,11),sharex=True)
    for pos in [27,66]:
        r=rows(f'P0_guard_p{pos}_s301');t=np.array([q['measurement']['t'] for q in r]);t-=t[0]
        axs[0].plot(t,[q['signed']['v_forward_mps'] for q in r],label=str(pos));axs[1].plot(t,[q['guard'].get('net_backward_m',0)*1000 for q in r],label=str(pos));axs[2].step(t,[{'clear':0,'risk':1,'emergency':2}[q['guard']['level']] for q in r],label=str(pos));axs[3].step(t,[q['measurement']['applied']['brake'] for q in r],label=str(pos))
    for ax,label in zip(axs,['Signed speed m/s','Net backward mm','Guard level','Applied brake']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.3)
    axs[1].axhline(8,color='red',ls='--');axs[2].set_yticks([0,1,2],['clear','risk','emergency']);axs[-1].set_xlabel('Simulation seconds');fig.tight_layout();fig.savefig(root/'m6e_guard_live.png',dpi=140);plt.close(fig)
    if Path('reports/m6e_variant_analysis.json').exists():
        fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True)
        for version in ['P0','P1']:
            r=rows(f'{version}_validation_p170_s301_obstacle');t=np.array([q['measurement']['t'] for q in r]);t-=t[0]
            axs[0].plot(t,[q['signed']['v_forward_mps'] for q in r],label=version);axs[1].plot(t,[q['obstacle_obb_gap_m'] if q['obstacle_obb_gap_m'] is not None else np.nan for q in r],label=version);axs[2].plot(t,[q['measurement']['long_acceleration_api'] for q in r],label=version)
        for ax,label in zip(axs,['Speed m/s','Actual OBB gap m','Raw API acceleration']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.3)
        axs[1].axhline(3,color='red',ls='--');axs[-1].set_xlabel('Simulation seconds');fig.tight_layout();fig.savefig(root/'m6e_physics_safety.png',dpi=140);plt.close(fig)
        fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True)
        for version in ['P0','P1']:
            r=rows(f'{version}_validation_p66_s301_A');t=np.array([q['measurement']['t'] for q in r]);t-=t[0];v=np.array([q['signed']['v_forward_mps'] for q in r]);a=np.diff(v)/.05
            axs[0].plot(t,v,label=version);axs[1].plot(t[1:],a,label=version);axs[2].plot(t[2:],np.diff(a)/.05,label=version)
        for ax,label in zip(axs,['Speed m/s','Raw acceleration','Raw jerk']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.3)
        axs[-1].set_xlabel('Simulation seconds');fig.tight_layout();fig.savefig(root/'m6e_physics_raw.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
