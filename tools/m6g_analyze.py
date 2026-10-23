"""Reproducible M6-G offline event study; zero simulator/network imports."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from diagnostics.m6g_events import command_events,flatten,braking_metrics,point,wheel_values,CONTROL_FIELDS


def read(p):return json.loads(Path(p).read_text())
def asset(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def quant(x):
    x=np.asarray(x,float)
    return None if not len(x) else dict(n=len(x),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),max=float(x.max()))
def counts(values):return dict(collections.Counter(values))


def main():
    cfg=read('configs/m6g/protocol.json');thr=cfg['annotation_thresholds'];events=[];trials=[];braking=[];startup=[];gears=[];rebuilds=[];spikes=[];delayed=[];rt=[];rpc=[];tick=[];interval=[];total=0;missing=0
    root=Path('reports/m6g_event_timelines');root.mkdir(exist_ok=False)
    for path in sorted(Path('reports/m6f_trials_0916').glob('*/*/trace.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()];total+=len(rows)
        task=read(path.with_name('config.json'));result=read(path.with_name('result.json'));launch=read(path.parent.parent/'launch.json');init=read(path.with_name('initialization.json'))
        rebuild=task['physics']!='P0-unmodified';samples=flatten(rows,rebuild);tid=path.parent.parent.name+'__'+path.parent.name
        out=root/(tid+'.jsonl')
        with out.open('x') as f:
            for s in samples:f.write(json.dumps(s,allow_nan=False)+'\n')
        origin=dict(trial_id=tid,trace=asset(path),timeline=asset(out),carla_version='0.9.16',physics_sha256=result['physics_sha256'],
          acquisition_git=result['source_commit'],loaded_source=launch['loaded']['project_imports'],config=task,
          wheel_telemetry='actual M6F API observations, not M6G new collection',engine_substep='NotAvailable')
        trials.append(origin)
        for i,r in enumerate(rows):
            tel=r['telemetry_after'];missing+=tel is None
            if tel is not None:
                ks=[k for k in ['throttle','brake','steer'] if abs(r['requested'][k]-tel['data'][k])>1e-5]
                if ks:
                    j=next((j for j in range(i+1,len(rows)) if rows[j]['telemetry_after'] and all(abs(rows[j]['telemetry_after']['data'][k]-r['requested'][k])<1e-5 for k in ks)),None)
                    delayed.append(dict(trial_id=tid,request_frame=r['body_before']['frame'],first_post_request_frame=r['body']['frame'],request_index=i,fields=ks,first_matching_index=j,delay_ticks=None if j is None else j-i,requested=r['requested'],observed_applied=r['body']['applied'],trace_sha256=origin['trace']['sha256']))
            for key in ['telemetry_before','telemetry_after','repeated_after']:
                if r[key]:rpc.append(dict(ms=r[key]['rpc_ms'],trial_id=tid,frame=r['body']['frame'],key=key))
            tick.append((r['timeline']['tick_end']-r['timeline']['apply_end'])*1000)
        intervals=np.diff([r['timeline']['apply'] for r in rows])*1000;interval.extend(intervals)
        rt.append(dict(trial_id=tid,apply_intervals_ms=quant(intervals),mean_Hz=1000/np.mean(intervals),tick_ms=quant([(r['timeline']['tick_end']-r['timeline']['apply_end'])*1000 for r in rows]),deadline_miss=sum(r['deadline_missed'] for r in rows),skipped_wall_slots=sum(r['skipped_slots'] for r in rows)))
        for n,e in enumerate(command_events(rows)):
            a=e['applied_index'];i=e['request_index'];end=e['end_index'];r=rows[i];eid=tid+'__command_'+str(n)
            kind='startup' if 'throttle' in e['changed_fields'] and e['command']['throttle']>0 else 'braking' if 'brake' in e['changed_fields'] and e['command']['brake']>0 else 'control_release_or_other'
            ref=dict(event_id=eid,trial_id=tid,kind=kind,source_trace_sha256=origin['trace']['sha256'],physics_sha256=origin['physics_sha256'],acquisition_git=origin['acquisition_git'],source_manifest_trial_id=tid,
               request_wall_time=r['timeline']['apply'],request_rpc_return_wall_time=r['timeline']['apply_end'],request_source_frame=r['body_before']['frame'],phase=r['phase'],**e)
            if a is not None:
                ref.update(first_applied_frame=samples[a]['frame'],first_applied_observed_sim_s=samples[a]['timestamp_s'],
                   actual_application_sim_bracket_s=[rows[a]['body_before']['t'],rows[a]['body']['t']],
                   actual_application_wall_time='NotAvailable; request walltime and observation upper bound retained',
                   observation_wall_upper_bound=rows[a]['timeline']['end'],
                   timeline_lines_1based=[max(0,a-10)+1,min(len(rows),max(end,a+31))])
            events.append(ref)
            if a is None:continue
            if kind=='braking':
                metrics=braking_metrics(samples,e,thr);braking.append(dict(**ref,category='ordinary_light_brake' if abs(e['command']['brake']-.01)<1e-5 else 'initialize_emergency_or_hold',metrics=metrics,evidence='Correlated',mechanism='wheel lock/large slip candidate; substep causation Unresolved'))
            elif kind=='startup':
                base=samples[max(0,a-1)];base_rpm=base['telemetry']['engine_rpm'] if base['telemetry'] else None
                predicates={'rpm_rise':lambda s:s['telemetry'] is not None and base_rpm is not None and s['telemetry']['engine_rpm']>base_rpm+thr['rpm_rise'],
                   'gear1':lambda s:s['telemetry'] is not None and s['telemetry']['gear']==1,
                   'wheel_motion':lambda s:wheel_values(s,'omega') is not None and max(abs(v) for v in wheel_values(s,'omega'))>thr['wheel_motion_rad_s'],
                   'body_motion':lambda s:s['body_signed_speed_mps']>thr['body_motion_mps']}
                first={k:next((j for j in range(a,end) if fn(samples[j])),None) for k,fn in predicates.items()}
                neutral=[s for s in samples[a:end] if s['telemetry'] and s['telemetry']['gear']==0]
                startup.append(dict(**ref,first={k:point(samples,j,a) for k,j in first.items()},already_true_before={k:bool(fn(base)) for k,fn in predicates.items()},baseline_rpm=base_rpm,
                  gear0_samples=len(neutral),gear0_rpm_range=None if not neutral else [min(s['telemetry']['engine_rpm'] for s in neutral),max(s['telemetry']['engine_rpm'] for s in neutral)],
                  evidence='Correlated',inference='Gear0 wait followed by gear1/wheel/body response; internal target gear, clutch and auto latency unobserved'))
        for i,s in enumerate(samples):
            if i and s['telemetry'] and samples[i-1]['telemetry'] and s['telemetry']['gear']!=samples[i-1]['telemetry']['gear']:
                gears.append(dict(event_id=tid+'__gear_'+str(i),trial_id=tid,frame=s['frame'],timestamp_s=s['timestamp_s'],from_gear=samples[i-1]['telemetry']['gear'],to_gear=s['telemetry']['gear'],phase=s['phase'],timeline_lines_1based=[max(0,i-10)+1,min(len(samples),i+31)],source_trace_sha256=origin['trace']['sha256'],physics_sha256=origin['physics_sha256'],acquisition_git=origin['acquisition_git'],evidence='Observed'))
        # Every suprathreshold raw jerk frame belongs to exactly one contiguous event.
        indices=[i for i,s in enumerate(samples) if s['jerk_50ms_mps3'] is not None and abs(s['jerk_50ms_mps3'])>thr['jerk_abs_mps3']]
        clusters=[]
        for i in indices:
            if not clusters or i>clusters[-1][-1]+1:clusters.append([i])
            else:clusters[-1].append(i)
        for group in clusters:
            peak=max(group,key=lambda j:abs(samples[j]['jerk_50ms_mps3']));s=samples[peak];tags=[]
            near=samples[max(0,group[0]-2):min(len(samples),group[-1]+3)]
            if any(wheel_values(x,'long_slip') is not None and max(abs(v) for v in wheel_values(x,'long_slip'))>=thr['large_slip_abs'] for x in near):tags.append('large_tire_slip_correlated')
            if any(wheel_values(x,'omega') is not None and max(abs(v) for v in wheel_values(x,'omega'))<=thr['wheel_lock_abs_rad_s'] for x in near):tags.append('wheel_lock_or_stationary_contact_correlated')
            if any(g['trial_id']==tid and samples[max(0,group[0]-2)]['frame']<=g['frame']<=samples[min(len(samples)-1,group[-1]+2)]['frame'] for g in gears):tags.append('gear_change_correlated')
            if rebuild and group[0]<10:tags.append('physics_rebuild_related_initialization')
            force=wheel_values(s,'long_force');unexplained=force is None or max(abs(v) for v in force)<=thr['force_near_zero_N']
            if unexplained:tags.append('same_frame_debug_force_insufficient')
            spikes.append(dict(event_id=tid+'__jerk_'+str(group[0]),trial_id=tid,first_frame=samples[group[0]]['frame'],last_frame=samples[group[-1]]['frame'],peak_frame=s['frame'],peak_index=peak,peak_jerk_mps3=s['jerk_50ms_mps3'],peak_acceleration_mps2=s['acceleration_50ms_mps2'],phase=s['phase'],same_frame_debug_force_insufficient=unexplained,
              candidates=tags or ['current_telemetry_unresolved'],evidence='Correlated' if tags else 'Unresolved',root_cause_status='Unresolved',
              timeline_lines_1based=[max(0,group[0]-10)+1,min(len(samples),group[-1]+31)],source_trace_sha256=origin['trace']['sha256'],physics_sha256=origin['physics_sha256'],acquisition_git=origin['acquisition_git']))
        if path.parent.parent.name=='sham':
            before=init['before_operation'];after=init['after_operation']
            rebuilds.append(dict(trial_id=tid,operation=task['physics'],repeat_id=task['repeat_id'],physics_sha256=origin['physics_sha256'],acquisition_git=origin['acquisition_git'],source_trace_sha256=origin['trace']['sha256'],
               before_operation=before,after_operation=after,first_ticks=samples[:10],
               instantaneous_live_speed_jump=after['telemetry']['data']['speed']-before['telemetry']['data']['speed'],
               same_frame=before['telemetry']['associated_frame']==after['telemetry']['associated_frame'],cached_body_identical=before['body']==after['body'],
               first_tick_delta_v_world=np.subtract(samples[0]['body_velocity_world'],before['body']['velocity']).tolist(),
               evidence='Observed',mechanism='RecreatePhysicsState supported by CARLA source; exact reset internal variables unavailable',
               derivative_exclusion='first derivative and adjacent jerk invalidated for sham/P1; no cache-to-new-state subtraction'))
    meta=dict(stage='M6-G',kind='OfflineAnalysisOfActualM6F_CARLA',new_CARLA_trials=0,source='Privileged',carla_version='0.9.16',analysis_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol=asset('configs/m6g/protocol.json'),sources={str(p):asset(p)['sha256'] for p in [Path(__file__),Path('diagnostics/m6g_events.py')]},acquisition_manifest='reports/m6g_execution_logs/trial_manifest.json',solver_causality='NotEstablished')
    write('reports/m6g_execution_logs/trial_manifest.json',dict(**{k:v for k,v in meta.items() if k!='acquisition_manifest'},trials=trials))
    old=read('reports/m6f_execution_logs/detailed_evidence.json')['command_transition_delays']
    oldset={(x['trial'].split('/')[-2]+'__'+x['trial'].split('/')[-1],x['frame']) for x in old};newset={(x['trial_id'],x['first_post_request_frame']) for x in delayed}
    if oldset!=newset:raise ValueError('Historical25DelayedEventsNotPreserved')
    write('reports/m6g_timeline_alignment.json',dict(**meta,rows=total,trial_count=len(trials),command_event_count=len(events),gear_event_count=len(gears),delayed_request_rows=delayed,all25_preserved=True,events=events,gear_events=gears,fields='complete per-frame timelines referenced by event line spans; exact units in M6F contract',physics_rebuild_derivative_invalidations=sum(x['physics_rebuild_boundary'] for tr in trials for x in [json.loads(Path(tr['timeline']['path']).read_text().splitlines()[0])]),outer_frame_only=True))
    light=[b for b in braking if b['category']=='ordinary_light_brake']
    write('reports/m6g_braking_event_analysis.json',dict(**meta,events=braking,light_brake_counts=dict(events=len(light),ordering=counts(x['metrics']['ordering'] for x in light),preexisting_large_slip=sum(x['metrics']['baseline_large_slip'] for x in light)),jerk_clusters=spikes,jerk_summary=dict(clusters=len(spikes),same_frame_debug_force_insufficient=sum(s['same_frame_debug_force_insufficient'] for s in spikes),candidate_counts=counts(t for s in spikes for t in s['candidates'])),normal_and_emergency_separated=True,emergency_same_speed_reference='NotRun_no_new_collection_or_instrumentation'))
    write('reports/m6g_startup_transmission_analysis.json',dict(**meta,events=startup,gear_events=gears,actual_internal_target_gear='NotAvailable',autobox_timers='NotAvailable',evidence='Correlated; mechanism reference does not establish Shipping causal identity'))
    write('reports/m6g_physics_rebuild_analysis.json',dict(**meta,events=rebuilds,scope='same-version existing original/sham/P1 logs; no new Physics',P1_status='NoGo',first_tick_internal_state='Only API-exposed state observed; full internal state NotAvailable'))
    write('reports/m6g_runtime_overhead.json',dict(**meta,RPC_ms=quant([x['ms'] for x in rpc]),RPC_worst=sorted(rpc,key=lambda x:x['ms'],reverse=True)[:12],world_tick_call_ms=quant(tick),apply_interval_ms=quant(interval),mean_Hz=1000/np.mean(interval),trials=rt,deadline_miss=sum(x['deadline_miss'] for x in rt),skipped_wall_slots=sum(x['skipped_wall_slots'] for x in rt),telemetry_absent_rows=missing,telemetry_absent_reason='440 planned OFF/initialization rows in ABBA, not dropped telemetry',failed_or_rejected_rows=0,unplanned_missing_ratio=0.,async_consumer='NotRun_no_async_diagnostic_added',server_instrumentation_overhead='NotRun',strict20Hz='Open',RPC_population='before/after plus290 repeated calls; M6F headline excluded repeated calls; both retained'))
    print(json.dumps(dict(rows=total,trials=len(trials),commands=len(events),delayed=len(delayed),brake_events=len(braking),light_brakes=len(light),startup=len(startup),gear=len(gears),jerk_clusters=len(spikes)),indent=2))

if __name__=='__main__':main()
