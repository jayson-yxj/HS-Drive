"""Independent physical-gap, task, comfort and timing accounting; append-only."""
import json
from pathlib import Path
import numpy as np
from tools.m6c_common import write,provenance,asset
from tools.m47_common import dynamics_metrics
from diagnostics.m6a_oracle_metrics import quantiles


def load(p):return json.loads(Path(p).read_text())


def inspect(path):
    d=load(path);rows=[json.loads(s) for s in (path.parent/'trace.jsonl').read_text().splitlines()];plans=load(path.parent/'plans.json');trans=load(path.parent/'transitions.json')
    g=[q['geometry'] for q in rows];v=np.array([q['measurement']['long_speed'] for q in rows]);t=np.array([q['sim_time'] for q in rows]);acc=np.diff(v)/np.diff(t);jerk=np.diff(acc)/np.diff(t[:-1]);interval=np.diff([q['timeline']['apply_control'] for q in rows])*1000
    k=int(interval.argmax())+1;p=rows[k-1]['timeline'];c=rows[k]['timeline']
    trace=dict(name=path.parent.name,config=d['config'],source='Privileged',result=asset(path),trace=d['trace'],physics=d['physics'],
        ticks=d['ticks'],goal_passed=d['goal_passed'],goal_checks=d['goal_checks'],error_m=d['final_goal_error_m'],completion=d['completion'],
        travelled_m=float(np.linalg.norm(np.diff([q['measurement']['position'] for q in rows],axis=0),axis=1).sum()),
        terminal_geometry=g[-1],dynamics=d['dynamics'],comfort_passed=d['comfort_pass'],degradation=d['degradation'],
        collisions=d['collisions'],illegal_lanes=d['illegal_lane_events'],lane_events=d['lane_events'],boundary_ticks=sum(q['boundary'] for q in rows),
        terminal_speed=d['terminal_speed_mps'],bypass=d['bypass'],transitions=trans,
        runtime={q:d[q] for q in ['control_interval_ms','over50','over55','over100','deadline_miss','skipped_wall_slots','mean_hz','world_tick_ms','expired_normal_control','fault_to_brake_s']},
        planner_age_s=quantiles([q['plan_wall_age'] for q in rows if q['plan_wall_age'] is not None]),
        worker_execution_ms=quantiles([(q['output_wall']-q['start_wall'])*1000 for q in plans]),
        worker_queue_wait_ms=quantiles([q['queue_wait_s']*1000 for q in plans if 'queue_wait_s' in q]),
        plan_input_origin_preserved=all(q.get('planned_horizon_s',6)==6 for q in plans),
        stages_ms={name:quantiles([(q['timeline'][end]-q['timeline'][start])*1000 for q in rows]) for name,start,end in [
            ('snapshot','wake_up','snapshot_end'),('submit_poll','snapshot_end','submit_poll_end'),('safety','submit_poll_end','safety_end'),
            ('geometry_controller','safety_end','controller_end'),('apply','apply_control','apply_end'),('tick','tick_start','tick_end')]},
        wakeup_lateness_ms=quantiles([(q['timeline']['wake_up']-q['timeline']['scheduled'])*1000 for q in rows]),
        worst=dict(ms=float(interval[k-1]),frame=rows[k]['frame'],previous=p,current=c,
          parts_ms=dict(previous_apply_to_tick=(p['tick_start']-p['apply_control'])*1000,tick=(p['tick_end']-p['tick_start'])*1000,
            tail=(c['previous_sleep_start']-p['tick_end'])*1000,sleep_wakeup=(c['wake_up']-c['previous_sleep_start'])*1000,preapply=(c['apply_control']-c['wake_up'])*1000)),
        peak_jerk_windows=[dict(raw_jerk=float(jerk[i]),samples=rows[max(0,int(i)-1):int(i)+4]) for i in np.argsort(abs(jerk))[-2:]],
        road_grade=quantiles([q['road']['grade'] for q in rows]),raw_acceleration_min=float(acc.min()))
    def event(q):
        if q is None:return None
        return dict(frame=q['frame'],speed=q['measurement']['speed'],signed_speed=q['measurement']['long_speed'],
            geometry=q['geometry'],road=q['road'],command=q['command'],applied=q['measurement']['applied'],reason=q['reason'],
            plan_input_timestamp=q['geometry']['planner_input_timestamp'])
    obs=[q for q in rows if q['geometry']['obstacle_obb_clearance_m'] is not None]
    first=obs[0] if obs else None
    stopped=[q for q in obs if q['measurement']['speed']<=.08]
    if obs:
        estimate=first['geometry']['predicted_braking_m']
        initial_available=first['geometry']['available_clearance_to_safe_stop']
        trace['obstacle']=dict(initial_stopping_feasible=estimate is not None and initial_available>=estimate,
            initial_feasibility_limit='Measured .01 brake capability with conservative delay/grade, not proof this controller normally executes that action',
            final_stop_requires_safety=any(q['reason'] and q['measurement']['speed']>.2 for q in rows),
            discovery=event(first),first_normal_deceleration=event(next((q for q in obs if q['measurement']['speed']>.2 and not q['reason'] and (q['command']['brake']>0 or q['command']['mode']=='SETTLE')),None)),
            first_emergency=event(next((q for q in obs if q['measurement']['speed']>.2 and q['reason']),None)),
            minimum_relevant_OBB_gap_m=min(q['geometry']['obstacle_obb_clearance_m'] for q in obs),
            final_relevant_OBB_gap_m=obs[-1]['geometry']['obstacle_obb_clearance_m'],
            final_3m_passed=obs[-1]['geometry']['obstacle_obb_clearance_m']>=3.,
            any_InsufficientStoppingMargin=any(q['reason']=='InsufficientStoppingMargin' for q in rows),
            stopped=bool(stopped),normal_only=not any(q['reason'] and q['measurement']['speed']>.2 for q in rows))
    # Full nearest physical gap includes lateral bypass actors, not only route.
    allgaps=[o['gap_m'] for q in rows for o in q['geometry']['all_obstacle_gaps']]
    trace['minimum_all_OBB_gap_m']=min(allgaps) if allgaps else None
    if d['config']['scene']=='lead':
        trace['lead']=dict(stopped=any(q['measurement']['speed']<=.08 for q in rows[350:500]),restarted=any(q['measurement']['speed']>.3 for q in rows[520:]),goal_passed=d['goal_passed'])
    if d['config']['scene']=='removed':trace['removed']=dict(restarted=any(q['measurement']['speed']>.3 for q in rows[420:]),goal_passed=d['goal_passed'])
    if d['config'].get('fault'):
        idx=d['config']['fault_tick'];trace['fault']=dict(type=d['config']['fault'],at_speed=rows[idx]['measurement']['speed'],reaction_s=d['fault_to_brake_s'],stopped=d['terminal_speed_mps']<=.08)
    return trace,rows


def main():
    allrows=[inspect(p) for p in sorted(Path('reports/m6c_trials').glob('*/result.json'))];formal=[d for d,r in allrows if d['name'].startswith('final_')]
    stop=[d for d in formal if d['name'].startswith('final_stop')];pairs=[]
    for seed in [101,102]:
        for spawn in [27,126,170]:
            for length in [9,25]:
                subset=[d for d in stop if d['config']['spawn']==spawn and d['config']['length']==length and d['config']['seed']==seed]
                pairs.append(dict(spawn=spawn,length=length,seed=seed,physics_identical=len({d['physics']['sha256'] for d in subset})==1,
                    arms={d['config']['arm']:dict(goal=d['goal_passed'],error=d['error_m'],signed=d['terminal_geometry']['goal_signed_distance_m'],comfort=d['comfort_passed']) for d in subset}))
    write('reports/m6c_stop_abcde_comparison.json',dict(**provenance('actual_CARLA_controlled_comparison_offline_statistics'),source='Privileged',status='Partial',trials=stop,paired=pairs,
        by_arm={arm:dict(runs=sum(d['config']['arm']==arm for d in stop),goal_passed=sum(d['config']['arm']==arm and d['goal_passed'] for d in stop),comfort_passed=sum(d['config']['arm']==arm and d['comfort_passed'] for d in stop)) for arm in ['A','C','D']},
        B='Historical_M6B_goal_only_4_of8_not_same_new_matrix',E='NotRun_no_evidence_backed_comfort_execution_candidate',
        development_exclusions=load('reports/m6c_development_deviation.json'),limits='Small fixed Town03 positions, paired repeats correlated; no generalization certification.'))
    obstacles=[d for d in formal if d['config']['scene'] in ['blocked','bypass','sudden','removed']]
    write('reports/m6c_no_bypass_safety.json',dict(**provenance('actual_CARLA_privileged_OBB_stopping_gap'),source='Privileged',status='Partial',trials=obstacles,
        limits='Nearest planar Actor OBB is a geometry reference, not exact render/curb mesh. Sudden infeasibility retains failed3m score; no masking difficult trials. Normal full stop requires no moving safety fallback.'))
    runtime=[d for d,r in allrows if d['name'].startswith('runtime_')]
    worst=max((d for d,r in allrows),key=lambda d:d['runtime']['control_interval_ms']['max'])
    write('reports/m6c_runtime_tail.json',dict(**provenance('actual_CARLA_client_timeline_worker_and_OS_resource_observation'),source='Privileged',status='Partial',matched_driving_trials=runtime,
        all_trial_count=len(allrows),all_trial_max=worst,all_deadline_miss=sum(d['runtime']['deadline_miss'] for d,r in allrows),all_over100=sum(d['runtime']['over100'] for d,r in allrows),
        historical=load('reports/m6c_goal_geometry_regression.json')['historical_worst'],
        resources=asset('runs/m6c_logs/resources.jsonl') if Path('runs/m6c_logs/resources.jsonl').exists() else None,
        limitations=['CPU schedstat and load observations only; not per-UE-thread renderer/PhysX attribution.','Server snapshot platform timestamp clock relation must be measured; no assumption remote clocks are synchronized.','Both runtime arms drive same task; no stationary-minus-moving inference.','Strict20Hz and shared-visual-GPU quality remain unvalidated.']))
    write('reports/m6c_safety_regression.json',dict(**provenance('actual_CARLA_separate_task_safety_and_comfort_accounting'),source='Privileged',status='Partial',
        completed_runs=len(allrows),formal_runs=len(formal),collisions=sum(len(d['collisions']) for d,r in allrows),illegal_solid_events=sum(len(d['illegal_lanes']) for d,r in allrows),
        expired_normal_control=sum(d['runtime']['expired_normal_control'] for d,r in allrows),
        formal_traveled_m=sum(d['travelled_m'] for d in formal),
        navigation=[d for d in formal if d['config']['scene'] in ['straight','mild_curve','junction_turn']],
        lead=[d for d in formal if d['config']['scene']=='lead'],faults=[d for d in formal if 'fault' in d],
        six_second_safety_unchanged=True,reverse_parking='NotRun',curb_mesh='Unknown'))
    figures(allrows);print('analyzed',len(allrows))


def figures(allrows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path('reports/m6c_figures');out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for d,r in allrows:
        if not d['name'].startswith('final_blocked_p126_v2.5_s101_'):continue
        t=np.array([q['sim_time'] for q in r]);t-=t[0];label=d['config']['arm']
        ax[0].plot(t,[q['measurement']['speed'] for q in r],label=label)
        ax[1].plot(t,[q['geometry']['obstacle_obb_clearance_m'] for q in r],label=label)
        ax[2].plot(t,[q['command']['throttle']-q['command']['brake'] for q in r],label=label)
    for a,title in zip(ax,['Actual speed m/s','Actual relevant OBB gap m','Throttle - Brake']):a.set_ylabel(title);a.grid(alpha=.3);a.legend()
    ax[1].axhline(3,color='red',ls='--');ax[-1].set_xlabel('simulation seconds');fig.tight_layout();fig.savefig(out/'m6c_no_bypass_gap.png',dpi=140);plt.close(fig)
    report=load('reports/m6c_coast_brake_envelope.json');fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,action in zip(axes,['coast','brake_001']):
        subset=[r for r in report['trials'] if r['split']=='validation' and r['action']==action]
        for method in ['A','B','C']:
            pairs=[(r['observed_distance_m'],r['predictions'][method]['distance_m']) for r in subset if r['predictions'][method]['distance_m'] is not None]
            if pairs:x,y=np.array(pairs).T;ax.scatter(x,y,label=method)
        lim=max([r['observed_distance_m'] for r in subset],default=1);ax.plot([0,lim],[0,lim],'k--');ax.set_title(action+' held-out roads');ax.set_xlabel('Observed travel (censored if not stopped)');ax.set_ylabel('Predicted upper distance m');ax.legend()
    fig.tight_layout();fig.savefig(out/'m6c_envelope_validation.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    data=load('reports/m6c_low_speed_dynamics.json')
    for d in data['trials']:
        task=d['task']
        if task['spawn']!=27 or task['seed']!=101 or task['mode']!='startup' or task['throttle']!=.15:continue
        r=[json.loads(s) for s in Path(d['trace']['path']).read_text().splitlines()];t=np.array([q['t'] for q in r]);t-=t[0];v=np.array([q['long_speed'] for q in r]);label=task['gear']
        axes[0].plot(t,v,label=label);axes[1].plot(t[:-1],np.diff(v)/np.diff(t),label=label);axes[2].step(t,[q['applied']['gear'] for q in r],where='post',label=label)
    for ax,title in zip(axes,['Signed speed m/s','Raw acceleration m/s²','Applied gear']):ax.set_ylabel(title);ax.legend();ax.grid(alpha=.3)
    axes[-1].set_xlabel('simulation seconds');fig.tight_layout();fig.savefig(out/'m6c_startup_modes.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
