"""Publication-style diagnostic plots from measured/frozen artifacts."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m7e_common import read


def main():
    out=Path('reports/m7e_figures');out.mkdir(exist_ok=False)
    episodes=['dev_constant_r3','dev_braking_r20','m7e_final_braking_r65','m7e_final_constant_r67','m7e_final_crossing_r28']
    labels={None:0,'D_stop':1,'E_slow_follow':2,'F_state_anchored_slow':2,'A_keep_route':3,'B_left':4,'C_right':5}
    fig,axes=plt.subplots(len(episodes),1,figsize=(13,12),sharex=True,sharey=True)
    for ep,ax in zip(episodes,axes):
        r=read(f'reports/m7e_replays/{ep}_B.json')['records'];t=[x['input_index']*.05 for x in r];ax.step(t,[labels[x['baseline_selected']] for x in r],where='post',label='Frozen A',alpha=.7);ax.step(t,[labels[x['selected']] for x in r],where='post',label='A + F',linestyle='--');ax.set_title(ep);ax.set_yticks([0,1,2,3,4],['Reject','Stop','Slow','Keep','Left']);ax.grid(alpha=.2);ax.legend(loc='upper right')
    fig.supxlabel('Original simulation time from episode start (s)');fig.suptitle('Offline state-based decisions, no trajectory execution');fig.tight_layout();fig.savefig(out/'m7e_decisions.png',dpi=160);plt.close(fig)
    d=read('reports/m7e_replays/dev_constant_r3_B.json')['records'];r=next(x for x in d if x['selected']=='F_state_anchored_slow' and x['baseline_selected']=='D_stop');s=r['request']['scene']['ego_xy_yaw_speed'];origin=np.array(s[:2]);yaw=s[2];rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]);fig,axes=plt.subplots(2,1,figsize=(12,8))
    for name in ['E_slow_follow','F_state_anchored_slow','D_stop']:
        c=next(e['candidate'] for e in r['evaluations'] if e['name']==name);pts=np.array(c['points']);p=(pts[:,:2]-origin)@rot;axes[0].plot(p[:,0],p[:,1],'.-',markersize=2,label=name);axes[1].plot(c['times_s'],pts[:,3],label=name)
    axes[0].scatter([0],[0],c='black',label='Measured Ego t=0',zorder=10);axes[0].set_xlabel('Forward from measured Ego (m)');axes[0].set_ylabel('Left (m)');axes[0].legend();axes[0].grid(alpha=.2);axes[1].set_xlabel('Original plan relative time (s)');axes[1].set_ylabel('Planned speed (m/s)');axes[1].legend();fig.tight_layout();fig.savefig(out/'m7e_state_connection.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ep,ax in zip(['dev_constant_r3','dev_braking_r20'],axes):
        for arm in ['A_uncached','A','B']:
            rs=read(f'reports/m7e_replays/{ep}_{arm}_clean.json')['records'];ax.plot([r['input_index']*.05 for r in rs],[r['delivery']['full_ms'] for r in rs],label=arm)
        ax.axhline(200,color='red',linestyle='--',label='200ms');ax.set_title(ep);ax.set_xlabel('Simulation time (s)');ax.legend()
    axes[0].set_ylabel('Complete serialized result latency (ms)');fig.tight_layout();fig.savefig(out/'m7e_full_latency.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
