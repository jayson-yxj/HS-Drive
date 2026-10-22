"""Additional raw-vs-window plots; no mutation of experiment reports."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m6c_common import write,provenance,asset


def main():
    report=json.loads(Path('reports/m6c_low_speed_dynamics.json').read_text());out=Path('reports/m6c_figures');out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(13,8));labels=[]
    for d in report['trials']:
        task=d['task']
        if task['seed']!=101 or task['mode']!='stop' or task['brake']!=0 or task['speed']!=2.5:continue
        rows=[json.loads(s) for s in Path(d['trace']['path']).read_text().splitlines()][d['probe_start']:]
        t=np.array([r['t'] for r in rows]);t-=t[0];v=np.array([r['long_speed'] for r in rows]);x=np.array([r['position'][:2] for r in rows]);distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(x,axis=0),axis=1))];label=f"spawn{task['spawn']} {task['split']}"
        axes[0,0].plot(t,v,label=label);axes[0,1].plot(t,distance,label=label);axes[1,0].plot(t,[r['road']['grade'] for r in rows],label=label);axes[1,1].plot(t[:-1],np.diff(v)/np.diff(t),label=label)
    for ax,title in zip(axes.flat,['Signed speed m/s','Travel since throttle release m','Map grade (positive uphill)','Raw longitudinal acceleration m/s²']):ax.set_title(title);ax.set_xlabel('simulation s');ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'m6c_coast_grade_response.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for d in report['trials']:
        task=d['task']
        if task['seed']!=101 or task['spawn']!=126 or task['mode']!='stop' or task['speed']!=2.5 or task['brake'] not in [.01,1]:continue
        r=[json.loads(s) for s in Path(d['trace']['path']).read_text().splitlines()][d['probe_start']:];t=np.array([q['t'] for q in r]);t-=t[0];v=np.array([q['long_speed'] for q in r]);a=np.diff(v)/np.diff(t);j=np.diff(a)/np.diff(t[:-1]);label=f"Brake={task['brake']}"
        axes[0].plot(t,v,label=label);axes[1].plot(t[1:],a,label=label+' raw');axes[2].plot(t[2:],j,label=label+' raw')
        sm=np.convolve(v,np.ones(5)/5,'valid');sa=np.diff(sm)/.05;sj=np.diff(sa)/.05
        axes[1].plot(t[3:3+len(sa)],sa,'--',label=label+' 250ms');axes[2].plot(t[4:4+len(sj)],sj,'--',label=label+' 250ms')
    for ax,title in zip(axes,['Actual signed speed m/s','Acceleration m/s²','Jerk m/s³']):ax.set_ylabel(title);ax.grid(alpha=.3);ax.legend()
    axes[-1].set_xlabel('simulation s');fig.tight_layout();fig.savefig(out/'m6c_brake_raw_window.png',dpi=140);plt.close(fig)
    print('additional probe figures completed')

if __name__=='__main__':main()
