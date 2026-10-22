"""Offline summaries keep every run/failure and original clock/dynamics samples."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m6b_common import write,provenance,asset
from tools.m47_common import dynamics_metrics
from diagnostics.m6a_oracle_metrics import quantiles


def load(path):return json.loads(Path(path).read_text())


def inspect(path):
    d=load(path);root=path.parent;r=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
    plans=load(root/'plans.json');trans=load(root/'transitions.json')
    times=np.array([v['sim_time'] for v in r]);times-=times[0]
    speed=np.array([v['measurement']['long_speed'] for v in r]);acc=np.diff(speed)/np.diff(times);jerk=np.diff(acc)/np.diff(times[:-1])
    peaks=[]
    for i in np.argsort(abs(jerk))[-3:][::-1]:
        peaks.append(dict(t=float(times[i+1]),jerk=float(jerk[i]),
            samples=[dict(t=float(times[k]),v=float(speed[k]),command=r[k]['command'],applied=r[k]['measurement']['applied'],reason=r[k]['reason']) for k in range(i,i+3)]))
    stages={k:quantiles([(v['timeline'][b]-v['timeline'][a])*1000 for v in r]) for k,a,b in [
        ('snapshot','wake_up','snapshot_end'),('submit_poll','snapshot_end','submit_poll_end'),('safety','submit_poll_end','safety_end'),
        ('controller_with_remaining_query','safety_end','controller_end'),('apply','apply_control','apply_end'),('world_tick','tick_start','tick_end')]}
    intervals=np.diff([v['timeline']['apply_control'] for v in r])*1000;i=int(np.argmax(intervals))+1
    primary=dict(name=root.name,config=d['config'],passed=d['goal_passed'],goal_checks=d['goal_checks'],
        error_m=d['final_goal_error_m'],heading_rad=d['heading_error_rad'],terminal_speed=d['terminal_speed_mps'],completion=d['completion'],
        signed_remaining_m=r[-1]['current_remaining_m'],
        collision_events=len(d['collisions']),lane_events=len(d['lane_events']),illegal_lane_events=len(d['illegal_lane_events']),
        selected_path_cte=d['selected_path_cte_m'],
        dynamics=d['dynamics'],comfort_pass=d['comfort_pass'],degradation=d['degradation'],
        transitions=trans,jerk_peak_windows=peaks,
        runtime={k:d[k] for k in ['control_interval_ms','over50','over55','over100','deadline_miss','skipped_wall_slots','mean_hz','expired_normal_control','world_tick_ms','fault_to_brake_s']},
        planner_ms=quantiles([(p['output_wall']-p['start_wall'])*1000 for p in plans]),
        planner_age_s=quantiles([v['plan_wall_age'] for v in r if v['plan_wall_age'] is not None]),
        planner_failure_rate=sum(p.get('reason') is not None for p in plans)/len(plans) if plans else None,
        planner_reasons=dict(Counter(p['reason'] for p in plans if p.get('reason'))),
        planned_acceleration_min=min((p.get('planned_acceleration_min',0) for p in plans),default=None),
        planned_acceleration_max=max((p.get('planned_acceleration_max',0) for p in plans),default=None),
        planned_internal_jerk_max=max((p.get('planned_internal_jerk_max',0) for p in plans),default=None),
        full_six_second_plans=all(p.get('planned_horizon_s',6)==6 for p in plans if not p.get('error')),
        stages_ms=stages,wakeup_lateness_ms=quantiles([(v['timeline']['wake_up']-v['timeline']['scheduled'])*1000 for v in r]),
        worst_interval=dict(ms=float(intervals[i-1]),previous=r[i-1]['timeline'],current=r[i]['timeline']),
        source='Privileged',result=asset(path),trace=d['trace'],physics=d['physics'],bypass=d['bypass'])
    phases={}
    for name,low,high in [('launch',0,4),('driving',4,float(times[-1])+1)]:
        sub=[v['measurement'] for t,v in zip(times,r) if low<=t<high]
        if len(sub)>=8:phases[name]=dynamics_metrics(sub)
    primary['phase_dynamics']=phases
    if d['config']['scene']=='lead':
        # Tick-defined fixture: lead stops at200, restarts500. Separate evidence.
        mid=r[350:500];after=r[520:]
        primary['lead_capabilities']=dict(ego_stopped_during_lead_stop=any(v['measurement']['speed']<=.08 for v in mid),
            ego_moved_after_restart=any(v['measurement']['speed']>.3 for v in after),
            restart_displacement_m=float(np.linalg.norm(np.array(after[-1]['measurement']['position'])-after[0]['measurement']['position'])) if after else None,
            source='Privileged_constant_velocity_lead_not_lead_dynamics_validation')
    if d['config']['scene']=='removed':
        after=r[420:];primary['removed_recovery']=dict(moved=any(v['measurement']['speed']>.3 for v in after),goal_passed=d['goal_passed'])
    if d['config']['scene'] in ('blocked','bypass','removed','lead'):
        from simulator.oracle_driver import footprint,polygons_intersect
        proxy_gaps=[];intersections=0
        route=np.array(load(root/'route.json')['selected'])
        from simulator.motion_oracle import route_projection
        for row in r:
            pos=row['measurement']['position'];xy=np.array(pos[:2])*[1,-1]
            yaw=-np.radians(row['measurement']['rotation'][2]);ego=footprint(*xy,yaw,2.396,1.082)
            s,_,_,_=route_projection(route,xy)
            for o in row['obstacles']:
                other=footprint(*o['xy'],o['yaw'],o['half_length'],o['half_width'])
                intersections+=int(polygons_intersect(ego,other))
                if d['config']['scene']!='bypass':
                    so,_,_,lat=route_projection(route,o['xy'])
                    if so>s and lat<2.5:proxy_gaps.append(so-s-o['half_length']-2.396)
        primary['obstacle_audit']=dict(discrete_nominal_OBB_intersection_frames=intersections,
            minimum_route_arc_gap_proxy_m=min(proxy_gaps) if proxy_gaps else None,
            reference='Actor OBB; no mesh/contact safety guarantee')
        if d['config']['scene']=='blocked':
            primary['refusal_passed']=d['bypass']['status']=='NoFeasibleBypass' and d['terminal_speed_mps']<=.08 and not d['collisions'] and not d['illegal_lane_events'] and intersections==0 and not any(v['boundary'] for v in r)
    return primary,r


def main():
    paths=sorted(Path('reports/m6b_trials').glob('*/result.json'));allrows=[inspect(p) for p in paths]
    finals=[d for d,r in allrows if d['name'].startswith('final_stop')]
    paired=[]
    for spawn in [27,170]:
        for length in [9,25]:
            for seed in [101,102]:
                sub=[d for d in finals if d['config']['spawn']==spawn and d['config']['length']==length and d['config']['seed']==seed]
                paired.append(dict(spawn=spawn,length=length,seed=seed,arms={d['config']['arm']:dict(error=d['error_m'],passed=d['passed'],decel=d['dynamics']['max_decel_raw'],jerk=d['dynamics']['jerk_peak_raw']) for d in sub},
                    physics_identical=len({d['physics']['sha256'] for d in sub})==1))
    write('reports/m6b_low_speed_regression.json',dict(**provenance('actual_CARLA_matched_trials_offline_analysis'),source='Privileged',
        status='Partial',trials=finals,paired=paired,
        by_arm={a:dict(runs=sum(d['config']['arm']==a for d in finals),goal_passed=sum(d['passed'] and d['config']['arm']==a for d in finals),
                      comfort_passed=sum(d['comfort_pass'] and d['config']['arm']==a for d in finals)) for a in 'ABC'},
        limits='Two positions, two execution orders; seeds do not make deterministic repeats independent. Same Physics and route within pair. A uses old planner/controller on common new wall-slot harness; not identical M6-A scheduler.',
        old_test_read_only=True))
    maneuvers=[d for d,r in allrows if d['name'].startswith('final_') and not d['name'].startswith('final_stop')]
    write('reports/m6b_oracle_maneuver_results.json',dict(**provenance('actual_CARLA_privileged_maneuvers'),source='Privileged',status='Partial',trials=maneuvers))
    bypass=[d for d in maneuvers if d['config']['scene'] in ('bypass','blocked','removed')]
    write('reports/m6b_bypass_feasibility.json',dict(**provenance('actual_CARLA_map_audit_and_closed_loop'),source='Privileged',status='Partial',trials=bypass,
        limits=['Static rotated rectangles with padding; no general dynamic obstacle avoidance.',
                'Dense discrete body road grid and original six-second Safety; not exact continuous curb mesh.',
                'Broken crossings permitted only for audited same-direction lanes; full lane event logs retained.',
                'NoFeasibleBypass must be explicit, then stop; endpoint failure is not failure to reject unsafe bypass.']))
    runtime=[d for d,r in allrows if d['name'].startswith('runtime_')]
    write('reports/m6b_runtime_tail.json',dict(**provenance('actual_CARLA_monotonic_wall_time_and_offline_tail_analysis'),source='Privileged',status='Partial',trials=runtime,
        historical=asset('reports/m6b_goal_stop_geometry.json'),
        scheduler='fixed epoch+slot*.05; skip missed wall slots explicitly; no deadline reset and no catchup ticks; missed slots reported, not claimed successful cycles',
        A_limit='Stationary tick-only scaffold with snapshot/logging/sensors but no Worker/Controller. Not same motion workload as B/C/D; cannot isolate server rendering vs physics.',
        attribution='Client tick wall time cannot identify internal UE rendering/physics/scheduling. CPU affinity only bounds process placement, not CPU exclusivity or GPU isolation.',
        long_term_or_shared_visual_gpu='NotRun_inherited_Partial'))
    write('reports/m6b_safety_regression.json',dict(**provenance('actual_CARLA_faults_and_offline_safety_accounting'),source='Privileged',status='Partial',
        total_trials=len(allrows),collisions=sum(d['collision_events'] for d,r in allrows),
        illegal_lane_events=sum(d['illegal_lane_events'] for d,r in allrows),
        expired_normal_control=sum(d['runtime']['expired_normal_control'] for d,r in allrows),
        fault_trials=[d for d in maneuvers if d['config']['scene'] in ('emergency','crash','timeout','invalid')],
        unchanged_original_safety=True,horizon_s=6,visual_control=False,curb_contacts='Unknown_no_dedicated_mesh_reference',reverse_bay_parking='NotRun'))
    figures(allrows)
    print('analyzed',len(allrows),'trials')


def figures(allrows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('reports/m6b_figures');root.mkdir(exist_ok=True)
    fig,ax=plt.subplots(2,2,figsize=(13,8))
    for d,r in allrows:
        if not d['name'].startswith('final_stop_p27_l9_s101_'):continue
        t=np.array([q['sim_time'] for q in r]);t-=t[0];v=np.array([q['measurement']['long_speed'] for q in r]);label=d['config']['arm']
        ax[0,0].plot(t,v,label=label);ax[0,1].plot(t,[q['goal_error_m'] for q in r],label=label)
        ax[1,0].plot(t[:-1],np.diff(v)/np.diff(t),label=label)
        ax[1,1].plot(t,[q['command']['throttle']-q['command']['brake'] for q in r],label=label)
    for a,title in zip(ax.flat,['Actual speed m/s','Goal distance m','Raw acceleration m/s²','Throttle - Brake']):a.set_title(title);a.set_xlabel('simulation s');a.legend();a.grid(alpha=.3)
    fig.suptitle('Privileged A/B/C matched 9m trial; raw peaks retained');fig.tight_layout();fig.savefig(root/'m6b_stop_abc.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,spawn in zip(axes,[126,170]):
        d,r=next((d,r) for d,r in allrows if d['name']==f'final_bypass_p{spawn}_s101')
        route=load(Path(d['result']['path']).parent/'route.json');xy=np.array([[q['measurement']['position'][0],-q['measurement']['position'][1]] for q in r])
        for k in ['original','selected']:
            p=np.array(route[k]);ax.plot(p[:,0],p[:,1],label=k)
        ax.plot(xy[:,0],xy[:,1],label='actual ego');ax.set_title(f'Privileged spawn{spawn} completed {d["completion"]:.1%}');ax.axis('equal');ax.legend()
    fig.tight_layout();fig.savefig(root/'m6b_bypass.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    for d,r in allrows:
        if not d['name'].startswith('runtime_'):continue
        x=np.diff([q['timeline']['apply_control'] for q in r])*1000
        ax.plot(np.arange(len(x)),x,lw=.5,label=d['name'])
    ax.axhline(50,color='k',ls='--');ax.set_ylabel('actual control/tick interval ms');ax.set_xlabel('cycle');ax.legend(ncol=4);fig.tight_layout();fig.savefig(root/'m6b_runtime_tails.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
