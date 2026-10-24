"""Actual final-state high-offset diagnostic; selection affects illustration only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m7e_common import read


def main():
    rows=read('reports/m7e_replays/m7e_final_constant_r67_B.json')['records'];r=max((r for r in rows if r['selected']=='F_state_anchored_slow'),key=lambda r:abs(r['f_connection_info']['initial']['lateral_error_m']))
    s=r['request']['scene']['ego_xy_yaw_speed'];rot=np.array([[np.cos(s[2]),-np.sin(s[2])],[np.sin(s[2]),np.cos(s[2])]]);origin=np.array(s[:2]);fig,axes=plt.subplots(2,1,figsize=(11,8))
    for name in ['E_slow_follow','F_state_anchored_slow','D_stop']:
        e=next(e for e in r['evaluations'] if e['name']==name);c=e['candidate'];p=np.array(c['points']);xy=(p[:,:2]-origin)@rot
        axes[0].plot(xy[:,0],xy[:,1],'.-',label=name+(' accepted' if e['accepted'] else ' rejected'),markersize=2)
        axes[1].plot(c['times_s'],p[:,3],label=name)
    axes[0].scatter([0],[0],color='black',zorder=5,label='Actual t0');axes[0].set_xlabel('Forward from actual Ego (m)');axes[0].set_ylabel('Left (m)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].set_xlabel('Original relative time (s)');axes[1].set_ylabel('Planned speed (m/s)');axes[1].legend(fontsize=8)
    fig.suptitle(f"New final road67, original frame {r['frame']}, offline only");fig.tight_layout()
    path=Path('reports/m7e_figures/m7e_final_offset_connection.png');assert not path.exists();fig.savefig(path,dpi=170)
if __name__=='__main__':main()
