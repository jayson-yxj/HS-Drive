"""Standalone diagnostic plots from immutable raw CARLA logs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def series(rs):
    t=np.array([r['body']['t'] for r in rs]);v=np.array([r['body']['long_speed'] for r in rs]);a=np.r_[np.nan,np.diff(v)/np.diff(t)];j=np.r_[np.nan,np.diff(a)/np.diff(t)]
    return t-t[0],v,a,j


def main():
    out=Path('reports/m6f_figures');out.mkdir(exist_ok=False)
    a=rows('reports/m6f_trials_0916/responses/p27_A_401/trace.jsonl')
    b=rows('reports/m6f_trials_0916/responses/p27_B_401/trace.jsonl')
    fig,axs=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for label,rs in [('auto A',a),('manual B',b)]:
        x=[r for r in rs if r['phase']=='cold_run'];t,v,acc,jerk=series(x)
        axs[0].plot(t,v,label=label);axs[1].step(t,[r['telemetry_after']['data']['gear'] for r in x],label=label)
        axs[2].plot(t,[r['telemetry_after']['data']['engine_rpm'] for r in x],label=label)
        axs[3].plot(t,[np.mean([w['omega'] for w in r['telemetry_after']['data']['wheels']]) for r in x],label=label)
    for ax,unit in zip(axs,['body m/s','actual gear','engine rpm','mean wheel rad/s']):ax.set_ylabel(unit);ax.grid();ax.legend()
    axs[-1].set_xlabel('seconds from cold_run first post-tick sample');fig.suptitle('Actual CARLA 0.9.16 P0 / privileged / fixed .15 throttle');fig.tight_layout();fig.savefig(out/'m6f_startup_chain.png',dpi=160);plt.close(fig)
    i=next(i for i,r in enumerate(a) if r['phase']=='normal_stop');x=a[i-5:i+18];t,v,acc,jerk=series(x);t-=t[5]
    fig,axs=plt.subplots(6,1,figsize=(12,14),sharex=True)
    axs[0].plot(t,v,label='body speed m/s');axs[0].step(t,[r['requested']['brake'] for r in x],label='brake request');axs[0].legend()
    axs[1].plot(t,acc,label='raw finite-diff a');axs[1].plot(t,[r['body']['long_acceleration_api'] for r in x],label='API a');axs[1].set_ylabel('m/s²');axs[1].legend()
    axs[2].plot(t,jerk);axs[2].set_ylabel('raw jerk m/s³')
    for k,label,ax in [('omega','wheel rad/s',axs[3]),('long_slip','long slip',axs[4]),('long_force','long force N',axs[5])]:
        for w in range(4):ax.plot(t,[r['telemetry_after']['data']['wheels'][w][k] for r in x],label=f'wheel index {w}')
        ax.set_ylabel(label);ax.legend(ncol=4)
    for ax in axs:ax.grid();ax.axvline(0,color='gray',linestyle='--')
    axs[-1].set_xlabel('s relative to first post-brake sample; debug force substep unknown')
    fig.suptitle('Raw .01 brake: wheel lock/slip/force transition and body response (no smoothing)');fig.tight_layout();fig.savefig(out/'m6f_light_brake_chain.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for d in sorted(Path('reports/m6f_trials_0916/sham').glob('*401')):
        rs=rows(d/'trace.jsonl');t,v,acc,j=series(rs)
        for ax,y in zip(axs,[v,acc,j]):ax.plot(t,y,label=d.name)
    for ax,unit in zip(axs,['speed m/s','raw accel m/s²','raw jerk m/s³']):ax.set_ylabel(unit);ax.grid();ax.legend(fontsize=8)
    axs[-1].set_xlabel('elapsed sim s');fig.suptitle('Same-version sham reapply; P1 remains NoGo');fig.tight_layout();fig.savefig(out/'m6f_sham_raw.png',dpi=160);plt.close(fig)
    r=json.loads(Path('reports/m6f_runtime_overhead.json').read_text());blocks=r['paired_blocks'];fig,ax=plt.subplots(figsize=(11,5))
    names=list(blocks);ax.bar(names,[blocks[k]['work_ms']['p95'] for k in names],color=['tomato' if blocks[k]['on'] else 'steelblue' for k in names]);ax.axhline(50,color='black',linestyle='--');ax.set_ylabel('P95 loop work ms');ax.set_title('ABBA x2 stationary acquisition cost: red telemetry ON, blue OFF');fig.tight_layout();fig.savefig(out/'m6f_rpc_overhead.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for p in sorted(Path('reports/m6f_guard_0915').glob('p*/trace.jsonl')):
        rs=rows(p);start=next(i for i,r in enumerate(rs) if r['phase'] in ['guard_release','protected_stop']);x=rs[max(0,start-2):];t=np.array([r['measurement']['t'] for r in x]);t-=t[2]
        v=np.array([r['signed']['v_forward_mps'] for r in x]);pos=np.array([r['measurement']['position'][:2] for r in x]);yaw=np.radians(x[0]['measurement']['rotation'][2]);dist=(pos-pos[2])@np.array([np.cos(yaw),np.sin(yaw)])
        axs[0].plot(t,v,label=p.parent.name);axs[1].plot(t,dist*1000,label=p.parent.name);axs[2].step(t,[r['command']['brake'] for r in x],label=p.parent.name)
    for ax,label in zip(axs,['signed speed m/s','release displacement mm','brake command']):ax.set_ylabel(label);ax.grid();ax.legend()
    axs[-1].set_xlabel('seconds relative to guard release');fig.suptitle('Actual independent 0915 G1 probes; positive support evaluated separately');fig.tight_layout();fig.savefig(out/'m6f_guard_raw.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
