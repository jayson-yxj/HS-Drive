"""Synthetic raw-stream audits; no model/PID/server execution or private payloads."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from simulator.m8j13_logic_runtime import STOP
from simulator.m8j13_logic_shadow import ShadowChannel,PROTOCOL,CHECKS,ENDPOINTS
from tools.m8j13_logic_shadow_report import audit


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def stream(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def synthetic_run(tmp_path,n=24,hold=21,normal=True):
    root=tmp_path/'run';root.mkdir();identity=['run',7,'g'];actors={f'camera{i}':i for i in range(6)}
    cfg=dict(shadow_protocol=PROTOCOL,planned_targets=24,max_active_ticks=800,max_forward_attempts=24,max_pid_calls=24,preflight_pid_calls=0,
             tick_caps=dict(initialization=60,stationary=95,fault_stop=120))
    dump(root/'frozen_config.json',cfg);dump(root/'request.json',dict(commit='synthetic',config_sha256='synthetic'))
    snaps=[];ops=[];tick_reqs=[];events=[dict(event='phase',phase='Initialization',wall_s=-1.)];seq=0
    command=dict(source='InitialProtectionBrake',control=STOP,observation=None)
    def operation(kind,args,start,end,reply,limit=.5):
        nonlocal seq
        seq+=1;o=dict(sequence=seq,kind=kind,started=start,deadline=start+limit,operation_limit_s=limit,phase_deadline=start+limit,run_deadline=200.)
        r=dict(sequence=seq,kind=kind,args=args,operation=o,owner_submitted_wall_s=start)
        rep=dict(sequence=seq,kind=kind,operation=o,success=True,io_received_wall_s=start+.0001,call_end_wall_s=end-.0001,processing_complete_wall_s=end-.0001,**reply)
        dump(root/'io_requests'/f'{seq:06}.json',r);dump(root/'io_responses'/f'{seq:06}.json',rep)
        value=dict(operation=o,reply=rep,owner_consumed_wall_s=end);ops.append(value);return value
    def tick(frame,wall,phase='stationary'):
        s=dict(frame=frame,sim_s=frame*.05,pose=np.eye(4).tolist(),velocity=[0,0,0],speed=0.,speed_norm=0.,position=[0,0,0],input_wall_s=wall,tick_end_wall_s=wall+.001,phase=phase,command=copy.deepcopy(command),full_control=copy.deepcopy(STOP),feedback_matches=True)
        snaps.append(s);tick_reqs.append(dict(index=len(snaps),phase=phase,expected_frame=frame,wall_s=wall,command=copy.deepcopy(command)))
        operation('tick',dict(command=copy.deepcopy(command),phase=phase),wall,wall+.002,dict(snapshot=s,rpc_start_wall_s=wall+.0002,rpc_return_wall_s=wall+.0003),limit=2 if phase=='initialization' else .5)
        return s
    operation('setup',{},-2,-1.9,dict(identity=identity,sensor_actors=actors))
    operation('control',dict(command=command),-1.8,-1.79,dict(server_processed=True,due_tick_cue=False,frame_before=0,frame_after=0,rpc_start_wall_s=-1.799,rpc_return_wall_s=-1.795),limit=.2)
    for i in range(21):tick(i+1,i*.05,'initialization')
    dump(root/'target_schedule.json',dict(frames=[22+4*i for i in range(24)]))
    channel=ShadowChannel(identity);consumed=[];attempts=[];pids=[];timing=[]
    for i in range(n):
        frame=22+4*i;wall=2+i*.6
        if i:tick(frame-1,wall-.05)
        operation('register',dict(frame=frame),wall-.01,wall-.005,{})
        obs=tick(frame,wall);tick(frame+1,wall+.05);frozen=tick(frame+2,wall+.1)
        row=dict(frame_id=frame,timestamp_s=obs['sim_s'],imu_frame=frame,imu_timestamp_s=obs['sim_s'],ego_world_matrix=np.eye(4).tolist(),target_world_xy=[20,0],speed_mps=0.)
        metadata=[dict(camera=name,actor_id=actor,run=identity[0],episode=identity[1],sensor_generation=identity[2],frame=frame,timestamp_s=obs['sim_s'],receive_monotonic_s=wall+.08,copied_monotonic_s=wall+.081) for name,actor in actors.items()]
        operation('collect',{},wall+.103,wall+.11,dict(observation=obs,frozen=frozen,shadow_capture=dict(row=row,camera_metadata=metadata,archive_done_wall_s=wall+.109),capture_complete_wall_s=wall+.108))
        t=channel.request(obs,frozen,wall+.12,200.)
        dump(root/'shadow_tasks'/f'{i+1:06}.json',dict(task=t,row=row))
        times={k:wall+.35 for k in ENDPOINTS}
        times.update(worker_received=wall+.121,read_done=wall+.125,preprocess_start=wall+.13,preprocess_done=wall+.2,transfer_done=wall+.21,forward_start=wall+.22,publish_start=wall+.36)
        reply=dict(protocol=PROTOCOL,identity=identity,sequence=i+1,frame=frame,input_sim_s=t['input_sim_s'],input_wall_s=wall,task_deadline=t['deadline_wall_s'],validation_start=times['forward_done'],validation_deadline=times['forward_done']+.1,checks={k:True for k in CHECKS},times=times,model_forward=True)
        dump(root/'shadow_completions'/f'{i+1:06}.json',reply)
        operation('frozen',dict(expected=frozen),wall+.362,wall+.365,dict(snapshot=frozen))
        channel.consume(reply,frozen,wall+.369);channel.finalize_return(reply,wall+.37)
        consumed.append(dict(sequence=i+1,frame=frame,consume_reference_wall_s=wall+.369,consume_return_wall_s=wall+.37,received_wall_s=wall+.361,old_realtime_reason='PlanWallExpired',sim_age_s=.1))
        events.append(dict(event='shadow_task_submitted',task=t,frozen_confirmed_wall_s=wall+.115,wall_s=wall+.12))
        path=np.column_stack([np.zeros(15),np.arange(1,16)]).tolist()
        d=dict(task=t,times=times,memory_before=dict(cached=i>0,history_timestamp=None if i==0 else [(frame-4)*.05]),memory_after=dict(cached=True,history_timestamp=[frame*.05]),native_path=path,native_speeds=[0.]*6,hs_points=np.zeros((6,4)).tolist(),hs_timestamp=row['timestamp_s'],transport=dict(transported_path=path,current_navigation_target=[0,20],current_speed_mps=0.),raw_pid=dict(steer=0,throttle=1,brake=0),native_agent_control=dict(steer=0,throttle=1,brake=0),pid_metadata=dict(desired_speed=0),published_wall_s=wall+.36,gpu_transfer_ms=10,gpu_allocated=1,gpu_reserved=2)
        dump(root/'diagnostics'/f'{i+1:06}.json',d)
        attempts.append(dict(sequence=i+1,frame=frame));pids.append(dict(sequence=i+1,frame=frame));timing.append(dict(sequence=i+1,archive_ms=1))
    fault_time=2+n*.6;fault=dict(reason='NormalShadowCompletion' if normal else 'IODeadline:control',wall_s=fault_time)
    dump(root/'watchdog_fault.json',fault)
    brake=dict(reason=fault['reason'],control=STOP,server_processed=True,handoff_uncertain=False,submission_wall_s=fault_time+.001,return_wall_s=fault_time+.01,operation=dict(deadline=fault_time+.201))
    dump(root/'fault_brake.json',brake)
    command=dict(source='FaultBrake',control=STOP,observation=None)
    last=snaps[-1]['frame']
    for i in range(hold):tick(last+i+1,fault_time+.02+i*.05,'fault_stop')
    phase_counts={k:sum(s['phase']==k for s in snaps) for k in {s['phase'] for s in snaps}}
    dump(root/'result.json',dict(fault=fault,normal_completed=normal,initialization=dict(stable_ticks=21),phase_ticks=phase_counts,charged_ticks=len(snaps)+60,settings_charges=dict(settings_startup=30,settings_restore=30),model_controls=0,remaining_processes=[],server_advance_unknown=False,stop=dict(status='PhysicalStopObserved' if hold>=21 else 'ResponseObserved_HoldIncomplete')))
    dump(root/'owner_state.json',dict(control_uncertainty=None))
    dump(root/'model_final.json',dict(attempts=n,successful_forwards=n,completed=n,pid_calls=n,fixed_state_unchanged=True,model_controls=0,training=0,optimizer_steps=0))
    dump(root/'model_ready.json',dict(initial_memory=dict(history_timestamp=None,cached=False),parameters_frozen=True,eval=True,fp32=True,load_wall_s=1))
    dump(root/'cleanup_steps.json',dict(sensor_stops=[dict(passed=True)]*8,actor_destruction=[dict(destroyed=True)]*9,settings_restored=True,remaining=[],settings=dict(synchronous_mode=False),settings_expected=dict(synchronous_mode=False),complete=True))
    dump(root/'supervisor.json',dict(assets_unchanged=True,process_cleanup=True,model_exit_code=0,ports_after=dict(p=False),groups_after=[[],[]]))
    for name,data in [('io_consumption.jsonl',ops),('snapshots.jsonl',snaps),('tick_requests.jsonl',tick_reqs),('shadow_consumed.jsonl',consumed),('forward_attempts.jsonl',attempts),('pid_attempts.jsonl',pids),('worker_timing.jsonl',timing),('owner_events.jsonl',events)]:stream(root/name,data)
    stream(root.parent/'budget.jsonl',[dict(event='model_start',wall_s=-4),dict(event='carla_start',wall_s=-3),dict(event='finished',wall_s=fault_time+3)])
    return root


def test_24_complete_requires_raw_chain_and_terminal_hold(tmp_path):
    root=synthetic_run(tmp_path);r=audit(root)
    assert r['failed_checks']==[],r['evidence_errors']
    assert r['status']=='Passed_FiniteParkedLogicalShadowOnly'
    assert r['counts']['owner_complete_timely']==24
    assert r['termination']['terminal_hold_verified'] and r['termination']['post_fault_ticks']==21
    assert r['S0_review_ready'] and r['S0_driving_authorized'] is False


@pytest.mark.parametrize('hold',[0,1,20])
def test_preexisting_stationary_status_cannot_replace_terminal_hold(tmp_path,hold):
    root=synthetic_run(tmp_path,hold=hold);result=json.loads((root/'result.json').read_text());result['stop']['status']='PhysicalStopObserved';dump(root/'result.json',result)
    r=audit(root)
    assert r['status']=='Partial' and not r['termination']['terminal_hold_verified']


def test_unpublished_forward_keeps_independent_counts(tmp_path):
    root=synthetic_run(tmp_path,n=3,normal=False)
    (root/'shadow_completions'/'000003.json').unlink()
    r=audit(root)
    assert r['counts']['forward_attempts']==r['counts']['forward_success']==3
    assert r['counts']['published_notifications']==2
    assert r['counts']['owner_consumptions_recorded']==3 and r['counts']['owner_complete_timely']==2
    assert r['status']=='Partial' and r['unmatched_records']['forward_attempts']==[3]


def test_zero_forward_and_missing_evidence_report_partial(tmp_path):
    root=synthetic_run(tmp_path,n=0,hold=0,normal=False)
    (root/'model_final.json').unlink()
    r=audit(root)
    assert r['status']=='Partial' and r['counts']['forward_attempts']==0
    assert r['performance']['gpu_peak_allocated'] is None
    assert r['output']['native_speed_range_mps'] is None
    assert any(x.get('error')=='MissingEvidence' for x in r['evidence_errors'])


def test_hold_requires_resolved_handoff_even_with_brake_ack(tmp_path):
    root=synthetic_run(tmp_path)
    dump(root/'owner_state.json',dict(control_uncertainty=dict(sequence=10,reason='IODeadline:control')))
    r=audit(root)
    assert r['termination']['fault_brake_server_processed']
    assert r['termination']['handoff_uncertain'] and not r['termination']['terminal_hold_verified']


def test_earlier_operation_timeout_not_hidden_by_final_status(tmp_path):
    root=synthetic_run(tmp_path)
    rows=[json.loads(x) for x in (root/'io_consumption.jsonl').read_text().splitlines()]
    item=next(x for x in rows if x['operation']['kind']=='control');item['owner_consumed_wall_s']=item['operation']['started']+.3
    stream(root/'io_consumption.jsonl',rows)
    r=audit(root)
    assert r['status']=='Partial' and r['handoff']['failed']==1
    assert r['handoff']['over_200_through_500ms']==1
    assert not r['termination']['terminal_hold_verified']


def test_failed_runtime_status_alone_does_not_establish_pass(tmp_path):
    root=synthetic_run(tmp_path,n=9,hold=0,normal=False)
    result=json.loads((root/'result.json').read_text());result['status']='Passed';dump(root/'result.json',result)
    r=audit(root)
    assert r['status']=='Partial' and r['counts']['owner_complete_timely']==9
    assert r['counts']['unreached_targets']==15 and not r['S0_review_ready']


def test_fault_brake_late_success_cannot_admit_terminal_hold(tmp_path):
    root=synthetic_run(tmp_path)
    brake=json.loads((root/'fault_brake.json').read_text())
    brake['return_wall_s']=brake['submission_wall_s']+.3
    dump(root/'fault_brake.json',brake)
    rows=[json.loads(x) for x in (root/'snapshots.jsonl').read_text().splitlines()]
    for s in rows:
        if s['phase']=='fault_stop':
            s['input_wall_s']+=.4;s['tick_end_wall_s']+=.4
    stream(root/'snapshots.jsonl',rows)
    r=audit(root)
    assert r['status']=='Partial' and not r['termination']['terminal_hold_verified']


def test_terminal_hold_requires_consecutive_observed_ticks(tmp_path):
    root=synthetic_run(tmp_path)
    rows=[json.loads(x) for x in (root/'snapshots.jsonl').read_text().splitlines()]
    terminal=[s for s in rows if s['phase']=='fault_stop']
    stream(root/'snapshots.jsonl',[s for s in rows if s['phase']!='fault_stop']+[terminal[0],terminal[-1]])
    r=audit(root)
    assert r['status']=='Partial' and not r['termination']['terminal_hold_verified']


def test_missing_stationary_feedback_cannot_pass_vacuous_all(tmp_path):
    root=synthetic_run(tmp_path)
    rows=[json.loads(x) for x in (root/'snapshots.jsonl').read_text().splitlines()]
    stream(root/'snapshots.jsonl',[s for s in rows if s['phase']!='stationary'])
    assert audit(root)['status']=='Partial'


def test_one_missing_measurement_tick_prevents_complete_acceptance(tmp_path):
    root=synthetic_run(tmp_path)
    rows=[json.loads(x) for x in (root/'snapshots.jsonl').read_text().splitlines()]
    rows.remove(next(s for s in rows if s['phase']=='stationary'))
    stream(root/'snapshots.jsonl',rows)
    result=audit(root)
    assert result['status']=='Partial' and 'MeasurementSnapshotCoverage' in result['failed_checks']


def test_rpc_timeline_after_owner_consumption_cannot_pass(tmp_path):
    root=synthetic_run(tmp_path)
    rows=[json.loads(x) for x in (root/'io_consumption.jsonl').read_text().splitlines()]
    control=next(x for x in rows if x['operation']['kind']=='control')
    control['reply']['rpc_start_wall_s']=control['operation']['deadline']+1
    control['reply']['rpc_return_wall_s']=control['operation']['deadline']+2
    stream(root/'io_consumption.jsonl',rows)
    assert audit(root)['status']=='Partial'


def test_missing_model_final_does_not_invent_forward_failure(tmp_path):
    root=synthetic_run(tmp_path,n=1,hold=0,normal=False)
    (root/'model_final.json').unlink()
    r=audit(root)
    assert r['counts']['forward_attempts']==1
    assert r['counts']['forward_success'] is None and r['counts']['forward_failed'] is None
    assert r['counts']['forward_success_count_source'].startswith('Unknown:')


def test_zero_forward_count_names_actual_missing_final_evidence(tmp_path):
    root=synthetic_run(tmp_path,n=0,hold=0,normal=False)
    (root/'model_final.json').unlink()
    r=audit(root)
    assert r['counts']['forward_success']==r['counts']['forward_failed']==0
    assert r['counts']['forward_success_count_source'].startswith('No published model task')


def test_unconsumed_tick_retains_operation_identity(tmp_path):
    root=synthetic_run(tmp_path)
    rows=[json.loads(x) for x in (root/'io_consumption.jsonl').read_text().splitlines()]
    last=rows.pop();assert last['operation']['kind']=='tick'
    stream(root/'io_consumption.jsonl',rows)
    r=audit(root);sequence=last['operation']['sequence']
    assert r['status']=='Partial'
    assert r['unmatched_records']['io_requests_without_owner_consumption']==[sequence]
    failure=r['deadline']['failed_operations'][0]
    assert failure['sequence']==sequence and failure['kind']=='tick'
    assert failure['response_recorded'] and not failure['owner_consumed']
    assert failure['whole_operation_ms'] is None
    assert not r['termination']['terminal_hold_verified']


def planned_return_fixture(root,late):
    rows=[json.loads(x) for x in (root/'io_consumption.jsonl').read_text().splitlines()]
    c=next(x for x in rows if x['operation']['kind']=='control');o=c['operation']
    o.update(operation_limit_s=.5,deadline=o['started']+.5,phase_deadline=o['started']+.5)
    c['reply']['operation']=copy.deepcopy(o)
    q=json.loads((root/'io_requests'/f"{o['sequence']:06}.json").read_text());q['operation']=copy.deepcopy(o)
    proof=dict(phase='FiniteExecution',phase_started=o['started'],phase_deadline=o['phase_deadline'])
    q['args']['parked_handoff']=proof
    dump(root/'io_requests'/f"{o['sequence']:06}.json",q)
    stream(root/'io_consumption.jsonl',rows)
    publication=c['owner_consumed_wall_s']-.00005
    stream(root/'io_publications.jsonl',[dict(sequence=o['sequence'],publication_done_wall_s=publication)])
    returned=o['started']+(.501 if late else .3)
    stream(root/'planned_handoff_returns.jsonl',[dict(operation_sequence=o['sequence'],source=q['args']['command']['source'],owner_return_wall_s=returned,passed=True,phase_started=proof['phase_started'],phase_deadline=proof['phase_deadline'])])
    return o,returned,publication


def test_final_planned_return_after_deadline_rejects_timely_io(tmp_path):
    root=synthetic_run(tmp_path);operation,returned,_=planned_return_fixture(root,late=True)
    r=audit(root)
    assert r['status']=='Partial' and r['handoffs']['failed']==1
    failure=r['deadline']['failed_operations'][0]
    assert failure['sequence']==operation['sequence'] and failure['final_owner_return_recorded']
    assert failure['whole_operation_ms']==pytest.approx(501)
    assert failure['owner_after_deadline_ms']==pytest.approx(1)
    assert not r['termination']['terminal_hold_verified']


def test_planned_handoff_exports_endpoints_and_final_return_duration(tmp_path):
    root=synthetic_run(tmp_path);operation,returned,publication=planned_return_fixture(root,late=False)
    r=audit(root)
    assert r['status']=='Passed_FiniteParkedLogicalShadowOnly'
    assert r['handoffs']['over_200_through_500ms']==1
    record=r['handoffs']['records'][0]
    assert record['whole_operation_ms']==pytest.approx(300)
    assert record['timing_endpoints']['owner_return_wall_s']==returned
    assert record['timing_endpoints']['publication_done_wall_s']==publication
    assert record['timing_endpoints']['effective_deadline_wall_s']==operation['deadline']
    assert r['handoffs']['endpoint_from_submit_ms']['owner_return_wall_s']['p50']==pytest.approx(300)
