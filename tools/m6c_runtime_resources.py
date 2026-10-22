"""Associate actual client intervals with server snapshots and OS counters."""
import json
from pathlib import Path
import numpy as np
from tools.m6c_common import write,provenance,asset
from diagnostics.m6a_oracle_metrics import quantiles


def main():
    resources=[json.loads(s) for s in Path('runs/m6c_logs/resources.jsonl').read_text().splitlines()];rt=np.array([v['wall'] for v in resources]);trials=[]
    for p in sorted(Path('reports/m6c_trials').glob('runtime_*/trace.jsonl')):
        r=[json.loads(s) for s in p.read_text().splitlines()];events=[];interval=np.diff([v['timeline']['apply_control'] for v in r])*1000
        for k,ms in enumerate(interval,1):
            if ms<=55:continue
            current=r[k]['timeline'];prior=r[k-1]['timeline'];time=current['apply_control'];near=int(abs(rt-time).argmin())
            events.append(dict(frame=r[k]['frame'],interval_ms=float(ms),
                previous_tick_ms=(prior['tick_end']-prior['tick_start'])*1000,wakeup_lateness_ms=(current['wake_up']-current['scheduled'])*1000,
                preapply_wall_ms=(current['apply_control']-current['wake_up'])*1000,main_cpu_ms=r[k].get('main_cpu_s',0)*1000,
                nearest_resource_time_offset_s=float(rt[near]-time),resource=resources[near]))
        clockdiff=[q['timeline']['snapshot_end']-q['server_platform_timestamp'] for q in r]
        trials.append(dict(name=p.parent.name,events_over55=events,
            snapshot_client_minus_server_platform_s=quantiles(clockdiff),
            clock_limit='same-host server snapshot timestamp is earlier than client read; includes frame age and wall pacing, not a network clock calibration',
            trace=asset(p)))
    # schedstat reports CPU runtime, runqueue wait, timeslices cumulatively.
    bypid={}
    for row in resources:
        for p in row['processes']:
            bypid.setdefault(p['pid'],[]).append((row['wall'],p))
    process=[]
    for pid,values in bypid.items():
        if len(values)<2:continue
        try:
            a=list(map(int,values[0][1]['schedstat'].split()));b=list(map(int,values[-1][1]['schedstat'].split()))
        except ValueError:continue
        process.append(dict(pid=pid,command=values[0][1]['cmdline'],seconds=values[-1][0]-values[0][0],
            main_thread_runtime_s=(b[0]-a[0])/1e9,main_thread_runqueue_wait_s=(b[1]-a[1])/1e9,timeslices=b[2]-a[2]))
    write('reports/m6c_runtime_resources.json',dict(**provenance('actual_server_OS_GPU_observation_offline_clock_association'),source='Privileged',trials=trials,process_schedstat=process,
        resource=asset('runs/m6c_logs/resources.jsonl'),gpu=asset('runs/m6c_logs/resources.jsonl.gpu.csv'),
        load_average_range=[min(q['loadavg'][0] for q in resources),max(q['loadavg'][0] for q in resources)],
        limits='schedstat here is per-PID main thread, not all UE render/physics threads. 200ms sampling can miss transient load and cannot prove causal source of10ms tails. No new affinity sweep.'))

if __name__=='__main__':main()
