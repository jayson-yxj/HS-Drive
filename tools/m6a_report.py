"""Aggregate every M6-A CARLA trial, including failures, without rerunning models."""
import json
from pathlib import Path
import numpy as np
from diagnostics.m6a_oracle_metrics import analyze, quantiles
from tools.m6a_common import write, provenance, asset


def read(path):
    return json.loads(Path(path).read_text())


def main():
    trials=[]
    traces=[]
    for path in sorted(Path('reports').glob('m6a_*artifacts/*_result.json')):
        result=read(path)
        rows=[json.loads(s) for s in Path(result['log']).read_text().splitlines()]
        route=np.array(read(result['route'])['route_world_hs'])
        plans=read(result['worker_history'])
        record=analyze(rows,plans,route,result)
        record.update(trial=path.parent.name, source_snapshot=asset(path.parent/'source/manifest.json'),
                      vehicle_physics=asset(path.parent/(result['scene']+'_physics.json')),
                      dynamics=result['actual_dynamics'], result=asset(path),
                      load_description=result['load_description'],
                      lead_motion=result.get('lead_motion','actuator' if result['scene']=='dynamic_lead' else None))
        if path.parent.name != 'm6a_schema_smoke_artifacts':
            record['provenance_correction']='Original envelope default said hs_ego but snapshot state/obstacles are hs_world; measurement is native CARLA world. Offline metrics use explicit Y flip. Original trace retained; no numeric control or calibration change. Final schema smoke verifies corrected metadata.'
        if path.parent.name=='m6a_oracle_remaining_artifacts' and result['scene']=='low_speed_stop':
            record['load_correction']='Partially concurrent full test suite; not isolated. Clean repeat retained separately; do not discard either.'
        intervals=np.diff([r['control_apply_wall'] for r in rows])*1000
        worst=int(np.argmax(intervals))+1
        record['worst_interval_diagnostic']=dict(
            interval_ms=float(intervals[worst-1]),previous_profile_ms=rows[worst-1]['profile'],
            current_profile_ms=rows[worst]['profile'],current_wakeup_lateness_ms=rows[worst]['wakeup_lateness_ms'])
        trials.append(record);traces.append((record,rows,route))
    unexpired=sum(t['expired_normal_control'] for t in trials)
    write('reports/m6a_oracle_planner_regression.json',dict(
        **provenance('actual_CARLA_closed_loop_with_offline_trace_metrics'),source='Privileged',
        status='Partial',decision='Go_OracleEngineeringOnly_NotSafetyCertification',trials=trials,
        total_trials=len(trials),total_ticks=sum(t['ticks'] for t in trials),
        collisions=sum(len(t['collisions']) for t in trials),
        lane_invasions=sum(len(t['lane_invasions']) for t in trials),expired_normal_control=unexpired,
        runtime_contract_unchanged=True,visual_input_to_control=False,
        bypass_status='NotImplemented_static_obstacle_trial_stops_and_fails_route_completion',
        reverse_bay_parking_status='NotRun_NotImplemented',
        curb_contact_status='Unknown_no_independent_curb_contact_reference',
        repetitions='Normal routes one each; short stop repeated; lead actuator and scripted fixtures differ; not iid replicates',
        failed_batches=['runs/m6a_logs/oracle.log: missing dynamic_lead route alias after three completed routes'],
        evaluation_limits=['Traffic lights forced green; no red-light/rule validation.',
                           'Collision sensor events are contact events, not unique accident counts.',
                           'Road footprint checks are corners, not exact continuous mesh checks.',
                           'All trials use Oracle route/map/actor snapshots; no RGB safety conclusion.',
                           'Following completion intentionally limited by stopped lead; not a complete navigation pass.',
                           'No dedicated curb reference; no zero-curb claim.']))
    write('reports/m6a_runtime_status.json',dict(
        **provenance('actual_CARLA_wall_clock_measurements'),source='Privileged',status='Partial',
        shared_gpu_strict_20hz='NotRevalidated_inherited_Partial',
        selected_periods_s=dict(planner=.2,controller=.05,max_plan_age=.35,safety_horizon=6),
        strict_20hz_passed=all(t['strict_20hz'] for t in trials),
        expired_normal_control=unexpired,
        trials=[{k:t[k] for k in ['trial','scene','ticks','mean_wall_hz','control_interval_ms','control_intervals_over_50ms','control_intervals_over_55ms','completion_deadline_misses','controller_timeout_count','planner_ms','stale_plan_control_ticks','stale_received_plan_count','degradation_ticks','plan_wall_age_s','worst_interval_diagnostic','load_description']} for t in trials],
        limits='CPU scheduling and CARLA tick tails retained. No shared visual workload test this stage; short runs cannot certify hard realtime. Queue put timing excludes feeder serialization.'))
    figures(traces)
    print('wrote',len(trials),'trials',flush=True)


def figures(traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path('reports/m6a_figures');out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    choices=[t for t in traces if (t[0]['trial']=='m6a_oracle_artifacts' or
                t[0]['trial']=='m6a_follow_scripted_artifacts' or
                t[0]['trial']=='m6a_low_speed_clean_artifacts' or
                (t[0]['trial']=='m6a_oracle_remaining_artifacts' and t[0]['scene']=='static_obstacle'))]
    for ax,(d,rows,route) in zip(axes.flat,choices):
        xy=np.array([[r['measurement']['position'][0],-r['measurement']['position'][1]] for r in rows])
        ax.plot(route[:,0],route[:,1],label='Privileged route')
        ax.plot(xy[:,0],xy[:,1],label='actual ego')
        ax.set_title(f"{d['scene']} {d['route_completion']:.1%}\nCTE P95 {d['cross_track_m']['p95']:.3f}m")
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('HS world X (m)');ax.set_ylabel('HS world Y (m)');ax.legend(fontsize=7)
    fig.suptitle('Actual CARLA / Privileged inputs; no visual control');fig.tight_layout();fig.savefig(out/'m6a_routes.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    for d,rows,_ in traces:
        if d['trial'] not in ('m6a_low_speed_clean_artifacts','m6a_follow_scripted_artifacts'):
            continue
        t=np.array([r['sim_time'] for r in rows]);t-=t[0]
        v=np.array([r['measurement']['speed'] for r in rows])
        axes[0].plot(t,v,label=d['scene']+' ego measured')
        if d['scene']=='dynamic_lead':axes[0].plot(t,[r['dynamic_actor_speeds'][0] for r in rows],label='scripted lead then brake')
        a=np.diff(v)/np.diff(t);j=np.diff(a)/np.diff(t)[1:]
        axes[1].plot(t[2:],j,label=d['scene']+' raw jerk')
    axes[0].set_ylabel('measured speed (m/s)');axes[1].set_ylabel('raw dv/dt jerk (m/s³)')
    for ax in axes:ax.set_xlabel('simulation seconds');ax.legend();ax.grid(alpha=.3)
    fig.suptitle('Actual speed and unfiltered finite-difference jerk; comfort NOT passed');fig.tight_layout();fig.savefig(out/'m6a_motion.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5))
    for d,rows,_ in traces:
        if d['trial'] not in ('m6a_oracle_artifacts','m6a_low_speed_clean_artifacts'):continue
        dt=np.diff([r['control_apply_wall'] for r in rows])*1000
        ax.plot(np.arange(len(dt))*.05,dt,lw=.6,label=d['scene'])
    ax.axhline(50,color='k',ls='--',label='50ms budget');ax.set_xlabel('nominal tick time (s)');ax.set_ylabel('actual apply interval (ms)');ax.legend();ax.set_title('Wall-clock tails retained; not strict 20Hz');fig.tight_layout();fig.savefig(out/'m6a_runtime.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
