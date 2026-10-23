"""Plots of raw measurements, not smoothed acceptance surrogates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(name):
 p=Path('reports/m6d_probe_artifacts')/name/'trace.jsonl';return [json.loads(s) for s in p.read_text().splitlines()]

def main():
 out=Path('reports/m6d_figures');out.mkdir(exist_ok=False)
 fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True)
 for name in ['cold_hand_auto_015','cold_hand_manual1_015','warm_service_auto_015']:
  rows=load('development_p27_s201_'+name);r=[q for q in rows if q['phase']=='release'];t=np.array([q['measurement']['t'] for q in r]);t-=t[0];v=np.array([q['signed']['v_forward_mps'] for q in r]);a=np.diff(v)/.05;j=np.diff(a)/.05
  axes[0].plot(t,v,label=name);axes[1].plot(t[1:],a,label=name);axes[2].plot(t[2:],j,label=name);axes[3].step(t,[q['measurement']['applied']['gear'] for q in r],label=name)
 for ax,y in zip(axes,['Signed speed m/s','Raw acceleration m/s²','Raw jerk m/s³','Applied gear']):ax.set_ylabel(y);ax.grid(alpha=.3);ax.legend()
 axes[2].axhline(5,ls='--',color='red');axes[2].axhline(-5,ls='--',color='red');axes[-1].set_xlabel('seconds after hold release');fig.tight_layout();fig.savefig(out/'m6d_release_raw.png',dpi=140);plt.close(fig)
 fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
 for position in [27,126,66,170]:
  split='development' if position in [27,126] else 'validation';rows=load(f'{split}_p{position}_s201_warm_service_auto_015');t=np.array([q['measurement']['t'] for q in rows]);t-=t[0]
  axes[0].plot(t,[q['signed']['v_forward_mps'] for q in rows],label=f'position {position}')
  axes[1].plot(t,[q['command']['throttle']-q['command']['brake'] for q in rows],label=f'position {position}')
  axes[2].plot(t,[q['distance_to_anchor_m'] for q in rows],label=f'position {position}')
 for ax,y in zip(axes,['Signed speed m/s','Throttle minus brake','Signed anchor offset m']):ax.set_ylabel(y);ax.grid(alpha=.3);ax.legend()
 axes[-1].set_xlabel('simulation seconds (anchor resets at hold transition)');fig.tight_layout();fig.savefig(out/'m6d_hold_restart.png',dpi=140);plt.close(fig)
 fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
 for arm in ['C','D','F0']:
  p=Path(f'reports/m6d_trials/dev_stop_p27_s201_{arm}/trace.jsonl');rows=[json.loads(s) for s in p.read_text().splitlines()];t=np.array([q['sim_time'] for q in rows]);t-=t[0]
  axes[0].plot(t,[q['signed']['v_forward_mps'] for q in rows],label=arm)
  axes[1].plot(t,[q['stop_target']['goal_signed_remaining_m'] for q in rows],label=arm)
  axes[2].plot(t,[q['command']['throttle']-q['command']['brake'] for q in rows],label=arm)
 for ax,y in zip(axes,['Signed speed m/s','Original goal remaining m','Throttle minus brake']):ax.set_ylabel(y);ax.grid(alpha=.3);ax.legend()
 axes[-1].set_xlabel('simulation seconds');fig.tight_layout();fig.savefig(out/'m6d_f0_comparison.png',dpi=140);plt.close(fig)
 fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
 for seed in [201,202]:
  rows=load(f'guard_diagnostic_p27_s{seed}_warm_guarded_auto_015');t=np.array([q['measurement']['t'] for q in rows]);t-=t[0]
  axes[0].plot(t,[q['signed']['v_forward_mps'] for q in rows],label=str(seed));axes[1].step(t,[q['command']['brake'] for q in rows],label=str(seed));axes[2].plot(t,[q['measurement']['long_acceleration_api'] for q in rows],label=str(seed))
 for ax,y in zip(axes,['Signed speed m/s','Brake','Raw API longitudinal acceleration']):ax.set_ylabel(y);ax.grid(alpha=.3);ax.legend()
 axes[-1].set_xlabel('simulation seconds');fig.tight_layout();fig.savefig(out/'m6d_rollback_protection.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
