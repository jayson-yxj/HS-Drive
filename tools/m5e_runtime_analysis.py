"""Repeated real CARLA timing attribution; affinity is a treatment, not a proof."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash
from tools.m5d_runtime_analysis import stats


def analyze_rows(rows):
    t=np.array([r['control_apply_wall'] for r in rows]);dt=np.diff(t)*1000
    speed=np.array([r['measurement']['speed'] for r in rows])[1:]
    moving_actors=np.array([max(r.get('dynamic_actor_speeds',[0]),default=0) for r in rows])[1:]>.2
    parts={}
    for name,m in [('all',np.ones(len(dt),bool)),('ego_moving',speed>.2),('ego_stationary',speed<=.2),('actor_moving',moving_actors)]:
        subset=[r for r,yes in zip(rows[1:],m) if yes]
        parts[name]=dict(interval_ms=stats(dt[m]),deadline_miss_rate=float(np.mean([r['deadline_missed'] for r in subset])) if subset else None,applied_control_interval_over50ms=int((dt[m]>50).sum()))
    tails=[]
    for i in np.argsort(dt)[-10:][::-1]:
        before,after=rows[i:i+2]
        tails.append(dict(frame=after['frame'],interval_ms=float(dt[i]),preceding_tick_ms=before['profile']['world_tick_ms'],current_compute_ms=after['compute_ms'],preceding_compute_ms=before['compute_ms'],between_cycles_ms=(after['wall_begin']-before['tick_end_wall'])*1000,wakeup_lateness_ms=after['wakeup_lateness_ms'],speed=after['measurement']['speed'],reason=after['reason']))
    return dict(phases=parts,tails=tails,wakeup_lateness_ms=stats([r['wakeup_lateness_ms'] for r in rows]),world_tick_ms=stats([r['profile']['world_tick_ms'] for r in rows]),plan_sim_age_s=stats([r['plan_sim_age'] for r in rows if r['plan_sim_age'] is not None]),plan_wall_age_s=stats([r['plan_wall_age'] for r in rows if r['plan_wall_age'] is not None]),normal_plan_wall_age_s=stats([r['plan_wall_age'] for r in rows if r['reason'] is None and r['plan_wall_age'] is not None]))


def resources(path):
    tele=json.loads(Path(path).read_text());procs={};prev={};gpus=[]
    for s in tele['records']:
        try:gpus.append([float(x) for x in s['gpu'].split(',')])
        except ValueError:pass
        for p in s['processes']:
            pid=str(p['pid']);old=prev.get(pid);prev[pid]=(s['wall'],p)
            if old is None or s['wall']<=old[0]:continue
            r=procs.setdefault(pid,dict(command=p['cmd'],cpu=[],wait=[],cores=set(),allowed=set()))
            r['cpu'].append(100*(p['cpu_ticks']-old[1]['cpu_ticks'])/tele['clock_ticks_per_s']/(s['wall']-old[0]));r['cores'].add(p['processor']);r['allowed'].add(p.get('allowed_cpus','unrecorded'))
            a,b=p['schedstat'].split(),old[1]['schedstat'].split()
            if len(a)>1 and len(b)>1:r['wait'].append((int(a[1])-int(b[1]))/1e6)
    gm=np.array(gpus)
    return dict(processes={k:dict(command=v['command'],process_cpu_one_core_percent=stats(v['cpu']),main_thread_schedule_wait_ms=stats(v['wait']),cpu_cores=sorted(v['cores']),affinity_sets=sorted(v['allowed'])) for k,v in procs.items()},gpu_utilization_percent=stats(gm[:,0]) if len(gm) else None,gpu_memory_mib=stats(gm[:,2]) if len(gm) else None)


def main():
    dest=Path('reports/m5e_runtime_attribution.json')
    if dest.exists():raise FileExistsError(dest)
    trial=json.loads(Path('reports/m5e_runtime_isolation.json').read_text());records=[]
    for case in trial['records']:
        for r in case['results']:
            rows=[json.loads(l) for l in Path(r['log']).read_text().splitlines()]
            rec=dict(case=case['case'],repeat=case['repeat'],result=r,**analyze_rows(rows),**resources(f"reports/m5e_runtime_{case['case']}_{case['repeat']}_resources.json"))
            rec['period_gate']=r['control_interval_ms']['p99']<=55 and r['control_interval_ms']['max']<=100 and r['deadline_miss_rate']<=.001 and r['normal_control_expired']==0;records.append(rec)
    faults=[]
    for r in json.loads(Path('reports/m5e_runtime_faults.json').read_text())['results']:
        rows=[json.loads(l) for l in Path(r['log']).read_text().splitlines()];faults.append(dict(result=r,**analyze_rows(rows)))
    grouped={}
    for c in sorted({r['case'] for r in records}):
        rr=[r for r in records if r['case']==c];grouped[c]=dict(repeats=len(rr),all_period_gates=all(r['period_gate'] for r in rr),max_intervals_ms=[r['result']['control_interval_ms']['max'] for r in rr],p99_ms=[r['result']['control_interval_ms']['p99'] for r in rr],deadline_miss_rates=[r['result']['deadline_miss_rate'] for r in rr],worst_max_ms=max(r['result']['control_interval_ms']['max'] for r in rr))
    atomic_json(dest,dict(kind='offline_attribution_actual_repeated_CARLA_logs',records=records,faults=faults,by_case=grouped,source_sha256=file_hash(__file__),limits='Two repeats samehost and reverseorder do not establish strict realtime guarantees; phase selection uses actualspeed. CPU affinity not exclusive cores. CPU includes all process threads; schedstat mainthread only. GPU utilization devicewide. Tick blocking not direct Unreal GPU time. Deadline=cycle finish misses50ms wall schedule; interval>50 counts separately.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,1,figsize=(13,8))
    for r in records:
        rows=[json.loads(l) for l in Path(r['result']['log']).read_text().splitlines()];t=np.array([x['control_apply_wall'] for x in rows]);label=f"{r['case']} r{r['repeat']}"
        ax[0].plot(t[1:]-t[0],np.diff(t)*1000,label=label,alpha=.55);ax[1].plot(t-t[0],[x['profile']['world_tick_ms'] for x in rows],alpha=.55)
    ax[0].axhline(50,color='k',ls='--');ax[0].set_ylabel('Applied control interval ms');ax[0].legend(fontsize=7,ncol=3);ax[1].set(xlabel='Wall time per trial s',ylabel='world.tick blocking ms');fig.tight_layout();fig.savefig('reports/m5e_figures/m5e_runtime_repeated_tails.png',dpi=150)

if __name__=='__main__':main()
