"""Offline applied-time plots; each plot cites acquisition trial and frames."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m6g_analyze import read


def series(tid):return [json.loads(s) for s in Path('reports/m6g_event_timelines/'+tid+'.jsonl').read_text().splitlines()]


def main():
    out=Path('reports/m6g_figures');out.mkdir(exist_ok=False)
    braking=read('reports/m6g_braking_event_analysis.json')['events']
    e=next(e for e in braking if e['event_id']=='responses__p27_A_401__command_3');s=series(e['trial_id']);a=e['applied_index'];x=s[a-5:a+16];t=np.array([v['timestamp_s']-s[a]['timestamp_s'] for v in x])
    fig,axes=plt.subplots(7,1,figsize=(12,15),sharex=True)
    axes[0].step(t,[v['request']['brake'] for v in x],where='post',label='Request');axes[0].step(t,[v['applied']['brake'] for v in x],where='post',label='Applied');axes[0].set_ylabel('brake');axes[0].legend()
    axes[1].plot(t,[v['body_signed_speed_mps'] for v in x]);axes[1].set_ylabel('body m/s')
    axes[2].plot(t,[v['acceleration_50ms_mps2'] for v in x],label='raw finite difference');axes[2].plot(t,[v['API_long_acceleration_mps2'] for v in x],linestyle='--',label='API');axes[2].set_ylabel('accel m/s²');axes[2].legend()
    for ax,key,label in zip(axes[3:],['omega','long_slip','tire_load','long_force'],['wheel rad/s','long slip','load N','debug long force N']):
        for w in range(4):ax.plot(t,[v['telemetry']['wheels'][w][key] for v in x],label=f'wheel {w}')
        ax.set_ylabel(label);ax.legend(ncol=4,fontsize=8)
    for ax in axes:ax.grid();ax.axvline(0,color='black',ls=':')
    axes[-1].set_xlabel('s relative to first observed Applied .01; true substep time unavailable')
    fig.suptitle(f"Offline M6F / {e['trial_id']} / Applied frame {s[a]['frame']}\nSource {e['acquisition_git'][:7]}, Physics {e['physics_sha256'][:12]} / no new experiment")
    fig.tight_layout();fig.savefig(out/'m6g_applied_brake_event.png',dpi=150);plt.close(fig)
    starts=read('reports/m6g_startup_transmission_analysis.json')['events'];fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for tid in ['responses__p27_A_401','responses__p27_B_401']:
        e=next(e for e in starts if e['trial_id']==tid and e['phase']=='cold_run');ss=series(tid);a=e['applied_index'];xx=ss[a:e['end_index']];tt=[s['timestamp_s']-ss[a]['timestamp_s'] for s in xx]
        for ax,key in zip(axes[:2],['engine_rpm','gear']):ax.plot(tt,[s['telemetry'][key] for s in xx],label=tid)
        axes[2].plot(tt,[np.mean([w['omega'] for w in s['telemetry']['wheels']]) for s in xx],label=tid)
        axes[3].plot(tt,[s['body_signed_speed_mps'] for s in xx],label=tid)
    for ax,label in zip(axes,['RPM','Current gear','mean wheel rad/s','body m/s']):ax.set_ylabel(label);ax.grid();ax.legend(fontsize=8)
    axes[-1].set_xlabel('s relative to first Applied throttle observation');fig.suptitle('Cold start: applied-time alignment, existing M6F traces');fig.tight_layout();fig.savefig(out/'m6g_startup_applied.png',dpi=150);plt.close(fig)
    details=read('reports/m6g_execution_logs/event_details.json');rebuild=read('reports/m6g_physics_rebuild_analysis.json')['events'];fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for e in rebuild:
        if e['repeat_id']!=401:continue
        x=e['first_ticks'];axes[0].plot(range(1,11),[s['body_velocity_world'][2] for s in x],marker='o',label=e['operation']);axes[1].plot(range(1,11),[s['body_position_world'][2] for s in x],marker='o',label=e['operation'])
    axes[0].set_ylabel('world vz m/s');axes[1].set_ylabel('world z m');axes[1].set_xlabel('post-operation tick index; not a solver substep')
    for ax in axes:ax.grid();ax.legend()
    fig.suptitle('Sham changes initial vertical motion; original vs rebuild, fixed public P0');fig.tight_layout();fig.savefig(out/'m6g_rebuild_first_ticks.png',dpi=150);plt.close(fig)
    b=read('reports/m6g_braking_event_analysis.json');fig,axes=plt.subplots(1,2,figsize=(12,5));c=b['light_brake_counts']['ordering'];axes[0].bar(['wheel first','body first','same frame'],[c.get(k,0) for k in ['wheel_decrease_observed_first','body_deceleration_observed_first','same_outer_frame']]);axes[0].set_title('37 light-brake events (33 P0 + 4 historical P1)');axes[0].set_ylabel('events, correlated samples')
    clusters=b['jerk_clusters'];axes[1].scatter([s['peak_acceleration_mps2'] for s in clusters],[abs(s['peak_jerk_mps3']) for s in clusters],c=['tomato' if s['same_frame_debug_force_insufficient'] else 'steelblue' for s in clusters],s=18);axes[1].set_xlabel('raw accel at jerk peak m/s²');axes[1].set_ylabel('|raw jerk| m/s³');axes[1].set_title('181 clusters; red: same-frame debug force <=1 N / missing')
    fig.tight_layout();fig.savefig(out/'m6g_event_evidence_limits.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
