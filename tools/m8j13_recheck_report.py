"""Data-driven parked recheck summary; supports empty, failed and partial runs."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from tools.m8j13_common import write,quantiles


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def lines(path):return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def stats(v):return dict(n=len(v),min=float(min(v)),**quantiles(v)) if v else None
def ages(items):
    return dict(samples=len(items),fresh=sum(x['reason'] is None and x.get('checks_passed',True) for x in items),
        wall_expired=sum(x.get('wall_expired',x.get('wall_age_s',0)>.300001) for x in items),
        simulation_expired=sum(x.get('simulation_expired') is True for x in items),
        simulation_unconfirmed=sum(x.get('snapshot_confirmed') is False for x in items),
        reasons=dict(Counter(str(x['reason']) for x in items)))

def summarize(root,cfg):
    paths=sorted((root/'diagnostics').glob('*.json'));d=[read(p) for p in paths];ledger=lines(root.parent/'forward_budget.jsonl')
    cs=lines(root/'candidates.jsonl');responses=lines(root/'completion_received.jsonl');workers=lines(root/'worker_timing.jsonl');checks=lines(root/'snapshot_checks.jsonl')
    snaps=lines(root/'snapshots.jsonl');byframe={x['frame']:x for x in snaps};feedback=lines(root/'parked_feedback.jsonl');measurement=[x for x in feedback if x['phase'] in ('measurement','pending_drain')]
    window_start=byframe[measurement[0]['frame']]['input_wall_s'] if measurement else None
    window_end=byframe[measurement[-1]['frame']]['tick_end_wall_s'] if measurement else None
    duration=window_end-window_start if measurement else None
    received={x['request']:x for x in responses};received_d=[(x,received[p.name]) for p,x in zip(paths,d) if p.name in received]
    endpoint1=ages([dict(x['completion_age'],checks_passed=x['finite'] and x['trajectory_check_reason'] is None) for x in d])
    endpoint2=ages([dict(x['checked_age'],checks_passed=x['finite'] and x['trajectory_check_reason'] is None) for x in d])
    endpoint3=ages([dict(reason=y['age_reason'],wall_age_s=y['wall_age_s'],wall_expired=y['wall_expired'],simulation_expired=y['simulation_expired'],snapshot_confirmed=True,checks_passed=y['necessary_checks_passed']) for x,y in received_d])
    times=[y['owner_consume_check_done_wall_s'] for x,y in received_d if y['consumable_fresh']]
    gap=max(np.diff([window_start]+times+[max(window_end,times[-1]) if times else window_end])) if duration is not None else None
    waits=defaultdict(float)
    for x in checks:waits[x['input_frame']]+=x['snapshot_wait_ms']
    bycheck=defaultdict(list)
    for x in checks:bycheck[x['input_frame']].append(x)
    executed=[x for x in ledger if x['kind']=='live_pipeline']
    result=read(root/'result.json',{});final=read(root/'model_final.json',{});supervisor=read(root/'supervisor.json',{})
    schedule=read(root/'first_camera_schedule.json',{})
    reached=sum(any(m['frame']==schedule.get('first_frame',-100)+4*i for m in measurement) for i in range(cfg['candidate_count']))
    details={}
    for label,subset in [('all',d),('subsequent',d[1:])]:
        details[label]=dict(end_to_HS_PID_ms=stats([x['timings_ms']['end_to_end_result'] for x in subset]),
            end_to_required_checks_ms=stats([x['timings_ms']['end_to_end_checked'] for x in subset]),
            snapshot_wait_cumulative_ms=stats([waits[x['frame']] for x in subset]),
            phases_ms={k:stats([x['timings_ms'][k] for x in subset if k in x['timings_ms']]) for k in (d[0]['timings_ms'] if d else [])})
        indices={x['attempt'] for x in subset};pairs=[(x,y) for x,y in received_d if x['attempt'] in indices]
        details[label]['end_to_owner_consume_ms']=stats([(y['owner_consume_check_done_wall_s']-x['input_wall_s'])*1000 for x,y in pairs])
        details[label]['end_to_notification_read_ms']=stats([(y['wall_s']-x['input_wall_s'])*1000 for x,y in pairs])
    candidate_details=[];outcomes=Counter()
    wr={x['request']:x for x in workers}
    for c in cs:
        status=c['status'];reason=c.get('reason');w=wr.get(c.get('request'),{})
        if status=='Submitted':
            if w.get('status')=='ShadowComplete':category='ExecutedForward'
            elif w.get('age',{}).get('reason')=='SimulationSnapshotUnconfirmed':category='SnapshotUnconfirmed'
            elif 'Expired' in str(w.get('age',{}).get('reason')):category='AgeRejected'
            else:category=w.get('status','UnresolvedSubmitted')
        elif status=='Skipped':category='QueueCapacity'
        elif status=='NotSubmitted':category='BudgetEnded'
        elif 'Expired' in str(reason):category='AgeRejected'
        else:category='InvalidInput'
        outcomes[category]+=1;candidate_details.append(dict(candidate=c,worker=w,category=category))
    gpu=dict(allocated_max=max((x['gpu_allocated'] for x in d),default=None),reserved_max=max((x['gpu_reserved'] for x in d),default=None),scope='Model process PyTorch allocator; not whole renderer/GPU utilization')
    partial_summary=dict(status='Recorded' if result.get('status')=='Recorded' and not (root/'model_error.json').exists() else 'Partial_IntegrationOrRuntimeBlock',
        attempts=len(executed),success=len(d),failed=len(executed)-len(d),offline_attempts=sum(x['kind']!='live_pipeline' for x in ledger),
        candidate_limit=cfg['candidate_count'],reached_candidates=reached,logged_candidates=len(cs),unlogged_reached=reached-len(cs),not_reached=cfg['candidate_count']-reached,
        candidate_outcomes=dict(outcomes),owner_outcomes=dict(Counter(x['status']+':'+str(x.get('reason')) for x in cs)),
        worker_outcomes=dict(Counter(x['status'] for x in workers)),HS_PID=endpoint1,RequiredChecks=endpoint2,OwnerConsumption=endpoint3,
        first_frame_ms=d[0]['timings_ms'] if d else None,timings=details,first_snapshot_wait_ms=waits[d[0]['frame']] if d else None,
        snapshot_wait_calls=len(checks),snapshot_wait_call_ms=stats([x['snapshot_wait_ms'] for x in checks]),
        snapshot_wait_per_observation_ms=stats(list(waits.values())),snapshot_wait_by_phase_ms={p:stats([x['snapshot_wait_ms'] for x in checks if x['phase']==p]) for p in sorted({x['phase'] for x in checks})},
        measurement_wall_s=duration,measurement_sim_s=len(measurement)*.05,physics_sim_hz=20,
        actual_wall_tick_hz=len(measurement)/duration if duration else None,actual_forward_hz=len(d)/duration if duration else None,
        candidate_wall_hz=reached/duration if duration else None,fresh_owner_hz=len(times)/duration if duration else None,
        fresh_intervals_s=stats(np.diff(times).tolist()),longest_no_fresh_window_s=float(gap) if gap is not None else None,
        fresh_wall_margin_ms=stats([(0.3-y['wall_age_s'])*1000 for x,y in received_d if y['consumable_fresh']]),
        timestamp_gaps_s=stats([x['timestamp_gap_s'] for x in d if x['timestamp_gap_s'] is not None]),
        input_interval_counts_s=dict(Counter(str(round(x['timestamp_gap_s'],4)) for x in d if x['timestamp_gap_s'] is not None)),
        model_state_unchanged=final.get('fixed_state_unchanged'),model_controls=result.get('model_controls',0),training=result.get('training',0),
        measurement_speed_max_mps=max((x['speed_mps'] for x in measurement),default=None),
        brake_feedback_all_matches=all(x['matched'] and x['control']['brake']==1 and x['control']['throttle']==0 for x in feedback),
        initialization_speed_peak_mps=max((x['speed_mps'] for x in feedback if x['phase']=='initialization'),default=None),
        exit_terminal_speed_mps=result.get('exit_terminal_speed_mps'),collision_count=len(lines(root/'collisions.jsonl')),
        actual_ticks=result.get('charged_ticks'),explicit_ticks=result.get('explicit_ticks'),starts=len(read(root.parent/'budget.json',{}).get('runs',[])),
        gpu=gpu,cleanup_complete=supervisor.get('cleanup_complete'),mmap_cleanup=result.get('transport_cleanup'),
        error=result.get('error'),model_error=read(root/'model_error.json'),
        realtime_claim=False,S0='NotRun',S1='NotRun',S2='NotRun',moving_faults='NotRun')
    return partial_summary,candidate_details


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/m8j13_per_camera_recheck.json'));p.add_argument('--output',type=Path,default=Path('reports/m8j13_sources/per_camera_recheck'));a=p.parse_args()
    cfg=read(a.config);root=Path(cfg['private_root'])/'live';summary,candidates=summarize(root,cfg);a.output.mkdir(parents=True,exist_ok=True)
    # Detailed candidate records remain private, never in public summary.
    write(root.parent/'candidate_dispositions.json',candidates);write(a.output/'live_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
