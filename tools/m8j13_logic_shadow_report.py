"""Read-only event audit for completed, failed and empty logical shadow runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.m8j13_logic_init_report import lines,distribution
from simulator.m8j13_logic_shadow import ShadowChannel,memory_valid,require_parked_execution,require_notification
from simulator.m8j13_logic_runtime import frozen_equal,canonical,STOP
from simulator.m8j13_sensor_audit import check_group
from simulator.m8j13_safety import sanitize_control,validate_trajectory,transport_native_path,PlanStamp,freshness
from datasets.m8j12_contract import native_target


def load(p):return json.loads(p.read_text()) if p.exists() else {}
def dist(v):return dict(first=v[0] if v else None,following=distribution(v[1:]),all=distribution(v))
def index_files(folder,key):return {load(p)[key]:load(p) for p in sorted(folder.glob('*.json'))}
def ratio(n,d):return n/d if d and d>0 else None


def termination_evidence(snaps,fault,brake,uncertain=False):
    reason=fault.get('reason');rows=[s for s in snaps if s.get('phase') in ('fault_stop','exit') and s.get('input_wall_s',-1)>=fault.get('wall_s',float('inf'))]
    stable=[]
    for s in rows:
        legal=(s.get('feedback_matches') is True and s.get('command',{}).get('source')=='FaultBrake' and canonical(s.get('full_control'))==canonical(STOP))
        if stable and (s['frame']!=stable[-1]['frame']+1 or abs(s['sim_s']-stable[-1]['sim_s']-.05)>1e-5):stable=[]
        if legal and s['speed_norm']<=.1:stable.append(s)
        else:stable=[]
    hold=stable[-1]['sim_s']-stable[0]['sim_s'] if stable else 0.
    processed=brake.get('server_processed') is True
    timely=processed and brake.get('return_wall_s',float('inf'))<=brake.get('operation',{}).get('deadline',float('-inf'))
    ordinary_known=not uncertain and not brake.get('handoff_uncertain',True)
    ordered=bool(rows) and all(s['input_wall_s']>=brake.get('return_wall_s',float('inf')) for s in rows)
    bounded=bool(rows) and len(rows)<=120 and rows[-1]['tick_end_wall_s']-fault.get('wall_s',0)<=10
    passed=processed and timely and ordinary_known and ordered and bounded and hold>=1.-1e-6
    return dict(normal=reason=='NormalShadowCompletion',first_reason=reason,post_fault_ticks=len(rows),terminal_hold_sim_s=hold,terminal_hold_verified=passed,
                fault_brake_server_processed=processed,fault_brake_timely=timely,ordinary_handoff_confirmed=ordinary_known,handoff_uncertain=not ordinary_known,
                status='PhysicalHoldObserved' if passed else ('HandoffUncertain' if not ordinary_known else 'HoldNotVerified'))


def audit(root):
    root=Path(root)
    cfg=load(root/'frozen_config.json');result=load(root/'result.json');sup=load(root/'supervisor.json');final=load(root/'model_final.json');ready=load(root/'model_ready.json');request=load(root/'request.json')
    ops=lines(root,'io_consumption.jsonl');snaps=lines(root,'snapshots.jsonl');consumed=lines(root,'shadow_consumed.jsonl');attempts=lines(root,'forward_attempts.jsonl');pid=lines(root,'pid_attempts.jsonl');events=lines(root,'owner_events.jsonl')
    reqs={int(p.stem):load(p) for p in (root/'io_requests').glob('*.json')};publications={x['sequence']:x for x in lines(root,'io_publications.jsonl')}
    io_responses={int(p.stem):load(p) for p in (root/'io_responses').glob('*.json')}
    handoff_returns={x['operation_sequence']:x for x in lines(root,'planned_handoff_returns.jsonl')}
    tasks={load(p)['task']['sequence']:load(p) for p in (root/'shadow_tasks').glob('*.json')};replies=index_files(root/'shadow_completions','sequence')
    diagnostics={load(p)['task']['sequence']:load(p) for p in (root/'diagnostics').glob('*.json')};timing={x['sequence']:x for x in lines(root,'worker_timing.jsonl')}
    setup=next((x['reply'] for x in ops if x['operation']['kind']=='setup' and x['reply'].get('success')),{});identity=setup.get('identity');actors=setup.get('sensor_actors',{})
    captures=[x for x in ops if x['operation']['kind']=='collect' and x['reply'].get('success')];byframe={x['reply']['observation']['frame']:x for x in captures}
    fault=load(root/'first_fault.json') or load(root/'watchdog_fault.json') or result.get('fault') or {};brake=load(root/'fault_brake.json');errors=[]
    if not fault and (root/'server_exited.json').exists():fault=dict(reason='ServerExited',wall_s=load(root/'server_exited.json')['wall_s'],source='supervisor; no owner/watchdog terminal record')
    def check(name,value):
        if not value:errors.append(name)
        return bool(value)
    def validate(name,fn):
        try:fn();return True
        except (AssertionError,KeyError,ValueError,TypeError,StopIteration,IndexError) as e:errors.append(name+':'+str(e));return False
    evidence_errors=[]
    for name in ('frozen_config.json','result.json','supervisor.json','model_final.json','model_ready.json','request.json','cleanup_steps.json'):
        if not (root/name).exists():
            evidence_errors.append(dict(file=name,error='MissingEvidence'));check('RequiredEvidence:'+name,False)
    planned=cfg.get('planned_targets',24);limits=load(Path('configs/m8j13_protocol.json'))['trajectory_checks']
    metrics={k:[] for k in ('six_camera_arrival_ms','six_camera_copy_ms','archive_before_submit_ms','freeze_confirmation_ms','submission_to_worker_ms','e2e_pid_ms','e2e_checks_ms','e2e_owner_ms','capture_ms','worker_read_ms','cpu_preprocess_ms','gpu_transfer_ms','forward_decode_ms','hs_ms','transport_ms','pid_ms','necessary_checks_ms','publish_ms','owner_after_publication_ms','task_to_consume_ms','validation_to_consume_ms','freeze_wait_ms','archive_ms')}
    passed_outputs=set();passed_consumption=set();old_pass=0;memory_checked=0;previous=None
    for seq,packet in sorted(tasks.items()):
        t=packet['task'];row=packet['row'];d=diagnostics.get(seq);reply=replies.get(seq)
        if not d:continue
        def output_checks():
            assert t['identity']==identity and t['frame']==row['frame_id'] and row['timestamp_s']==t['input_sim_s']
            memory_valid(d['memory_before'],d['memory_after'],previous,row['timestamp_s'])
            transported=transport_native_path(np.asarray(d['native_path']),row['ego_world_matrix'],t['frozen']['pose'])
            assert np.array_equal(transported,np.asarray(d['transport']['transported_path']))
            assert np.array_equal(native_target(t['frozen']['pose'],row['target_world_xy']),np.asarray(d['transport']['current_navigation_target']))
            assert d['transport']['current_speed_mps']==t['frozen']['speed'] and d['hs_timestamp']==row['timestamp_s']
            assert sanitize_control(d['raw_pid'])[1] is None and sanitize_control(d['native_agent_control'])[1] is None
            assert validate_trajectory(d['hs_points'],row['speed_mps'],limits)[0] is None
            assert reply is not None;require_notification(reply)
        if validate('Output:'+str(seq),output_checks):passed_outputs.add(seq);memory_checked+=1
        previous=row['timestamp_s']
    channel=ShadowChannel(identity or [])
    for c in consumed:
        seq=c['sequence'];packet=tasks.get(seq);reply=replies.get(seq);d=diagnostics.get(seq)
        def consumption_checks():
            nonlocal old_pass
            assert packet and reply and d and seq in passed_outputs
            t=packet['task'];cap=byframe[t['frame']];obs=cap['reply']['observation']
            assert channel.request(obs,t['frozen'],t['requested_wall_s'],t['run_deadline'])==t
            conf=next(x['reply']['snapshot'] for x in ops if x['operation']['kind']=='frozen' and x['reply'].get('success') and reply['times']['publish_start']<=x['operation']['started']<=c['consume_return_wall_s'])
            channel.consume(reply,conf,c['consume_reference_wall_s']);channel.finalize_return(reply,c['consume_return_wall_s'])
            assert c['consume_return_wall_s']>=c['consume_reference_wall_s']>=c['received_wall_s']
            assert not any(t['requested_wall_s']<=s['input_wall_s']<=c['consume_return_wall_s'] for s in snaps)
            stamp=PlanStamp(identity[0],t['frame'],t['input_sim_s'],t['input_wall_s'],reply['times']['forward_start'],reply['times']['checks_done'],'frozen')
            raw=freshness(stamp,identity[0],conf['frame'],conf['sim_s'],c['consume_return_wall_s'])
            assert raw==c['old_realtime_reason'];old_pass+=raw is None
        if not validate('Consumption:'+str(seq),consumption_checks):continue
        passed_consumption.add(seq);t=packet['task'];cap=byframe[t['frame']];tt=d['times'];origin=t['input_wall_s']
        event=next(e for e in events if e['event']=='shadow_task_submitted' and e['task']['sequence']==seq)
        vals=dict(six_camera_arrival_ms=(max(x['receive_monotonic_s'] for x in cap['reply']['shadow_capture']['camera_metadata'])-origin)*1000,six_camera_copy_ms=(max(x['copied_monotonic_s'] for x in cap['reply']['shadow_capture']['camera_metadata'])-origin)*1000,archive_before_submit_ms=(cap['reply']['shadow_capture']['archive_done_wall_s']-cap['reply']['capture_complete_wall_s'])*1000,freeze_confirmation_ms=(event['frozen_confirmed_wall_s']-cap['owner_consumed_wall_s'])*1000,submission_to_worker_ms=(tt['worker_received']-t['requested_wall_s'])*1000,
            e2e_pid_ms=(tt['pid_done']-origin)*1000,e2e_checks_ms=(tt['checks_done']-origin)*1000,e2e_owner_ms=(c['consume_return_wall_s']-origin)*1000,capture_ms=(cap['owner_consumed_wall_s']-origin)*1000,worker_read_ms=(tt['read_done']-tt['worker_received'])*1000,cpu_preprocess_ms=(tt['preprocess_done']-tt['preprocess_start'])*1000,gpu_transfer_ms=d['gpu_transfer_ms'],forward_decode_ms=(tt['forward_done']-tt['forward_start'])*1000,hs_ms=(tt['hs_done']-tt['forward_done'])*1000,transport_ms=(tt['transport_done']-tt['hs_done'])*1000,pid_ms=(tt['pid_done']-tt['transport_done'])*1000,necessary_checks_ms=(tt['checks_done']-tt['pid_done'])*1000,publish_ms=(d['published_wall_s']-tt['checks_done'])*1000,owner_after_publication_ms=(c['consume_return_wall_s']-d['published_wall_s'])*1000,task_to_consume_ms=(c['consume_return_wall_s']-t['requested_wall_s'])*1000,validation_to_consume_ms=(c['consume_return_wall_s']-reply['validation_start'])*1000,freeze_wait_ms=(c['consume_return_wall_s']-event['frozen_confirmed_wall_s'])*1000)
        if seq in timing:vals['archive_ms']=timing[seq]['archive_ms']
        for k,v in vals.items():metrics[k].append(v)
    for cap in captures:
        def capture_checks():
            r=cap['reply'];obs=r['observation'];row=r['shadow_capture']['row'];assert r['frozen']['frame']==obs['frame']+2
            assert check_group(r['shadow_capture']['camera_metadata'],dict(frame=obs['frame'],elapsed_seconds=obs['sim_s']),identity,actors)['valid']
            assert row['imu_frame']==obs['frame'] and abs(row['imu_timestamp_s']-obs['sim_s'])<=1e-5
            assert cap['owner_consumed_wall_s']<=obs['input_wall_s']+.5
        validate('Capture',capture_checks)
    check('FixedCadence',all(b['reply']['observation']['frame']-a['reply']['observation']['frame']==4 for a,b in zip(captures,captures[1:])))
    failures=[];handoffs=[];all_handoffs=[]
    consumed_ops={x['operation']['sequence']:x for x in ops}
    check('UniqueIOConsumption',len(consumed_ops)==len(ops))
    unconsumed=sorted(set(reqs)-set(consumed_ops))
    # Submitted requests remain evidence even when the owner/server exits before
    # producing a response or consumption record. Never infer a successful call.
    for sequence in sorted(set(reqs)|set(consumed_ops)):
        q=reqs.get(sequence,{});x=consumed_ops.get(sequence);o=x['operation'] if x else q['operation']
        r=x['reply'] if x else io_responses.get(sequence,{})
        consume_end=x['owner_consumed_wall_s'] if x else None
        ret=handoff_returns.get(sequence);end=ret['owner_return_wall_s'] if ret else consume_end
        proof=q.get('args',{}).get('parked_handoff');source=q.get('args',{}).get('command',{}).get('source')
        identity_ok=check('OperationIdentity',q.get('operation')==o and (not r or r.get('operation')==o))
        times=[o['started']]+[r[k] for k in ('io_received_wall_s','pre_control_snapshot_wall_s','parked_proof_checked_wall_s','rpc_start_wall_s','rpc_return_wall_s','post_control_snapshot_wall_s','snapshot_available_wall_s','feedback_complete_wall_s','frozen_checked_wall_s','processing_complete_wall_s','call_end_wall_s','publication_started_wall_s') if k in r]
        if consume_end is not None:times.append(consume_end)
        if ret:times.append(end)
        timeline_ok=check('OperationTimeline',all(a<=b for a,b in zip(times,times[1:])))
        limits=[o['started']+o['operation_limit_s'],o['phase_deadline'],o['run_deadline']]
        obs=q.get('args',{}).get('command',{}).get('observation') or q.get('args',{}).get('observation')
        if obs:limits.append(obs['input_wall_s']+4.)
        if proof:limits.append(proof['phase_deadline'])
        effective_deadline=min([o['deadline']]+limits)
        deadline_ok=check('OperationDeadlineBounds',o['deadline']<=min(limits)+1e-8)
        return_ok=True
        if ret:
            return_ok=check('PlannedHandoffReturnIdentity',proof is not None and ret['source']==source and ret['phase_started']==proof['phase_started'] and ret['phase_deadline']==proof['phase_deadline'])
            return_ok=return_ok and ret.get('passed') is True
        ok=bool(x and r.get('success') and end<=effective_deadline and identity_ok and timeline_ok and deadline_ok and return_ok)
        def elapsed(value,start=o['started']):return None if value is None else (value-start)*1000
        item=dict(sequence=sequence,operation=o,kind=o['kind'],source=source,operation_limit_s=o['operation_limit_s'],effective_deadline_wall_s=effective_deadline,whole_operation_ms=elapsed(end),io_queue_ms=elapsed(r.get('io_received_wall_s')),rpc_ms=None if 'rpc_start_wall_s' not in r or 'rpc_return_wall_s' not in r else (r['rpc_return_wall_s']-r['rpc_start_wall_s'])*1000,owner_after_deadline_ms=elapsed(end,effective_deadline),error='OwnerConsumptionUnconfirmed' if not x else r.get('error'),server_processing_confirmed=r.get('server_processed',False),owner_consumed=x is not None,final_owner_return_recorded=ret is not None,passed=ok)
        if not x:item['response_recorded']=bool(r)
        if not ok:failures.append(item)
        if o['kind']=='control':all_handoffs.append(item)
        if proof:
            check('HandoffPhaseDeadline',effective_deadline<=min(o['started']+.5,proof['phase_deadline'],o['run_deadline']))
            publication=publications.get(sequence,{}).get('publication_done_wall_s')
            frozen_after=next((v for v in ops if v['operation']['kind']=='frozen' and v['reply'].get('success') and consume_end is not None and consume_end<=v['operation']['started']<=proof['phase_deadline']),None) if source=='PlannedHoldBrake' else None
            frozen_end=frozen_after['owner_consumed_wall_s'] if frozen_after else r.get('post_control_snapshot_wall_s')
            endpoints=dict(phase_started_wall_s=proof['phase_started'],phase_deadline_wall_s=proof['phase_deadline'],owner_submitted_wall_s=q.get('owner_submitted_wall_s',o['started']),io_received_wall_s=r.get('io_received_wall_s'),pre_control_snapshot_wall_s=r.get('pre_control_snapshot_wall_s'),parked_proof_checked_wall_s=r.get('parked_proof_checked_wall_s'),rpc_start_wall_s=r.get('rpc_start_wall_s'),rpc_return_wall_s=r.get('rpc_return_wall_s'),post_control_snapshot_wall_s=r.get('post_control_snapshot_wall_s'),processing_complete_wall_s=r.get('processing_complete_wall_s'),publication_started_wall_s=r.get('publication_started_wall_s'),publication_done_wall_s=publication,owner_consumed_wall_s=consume_end,owner_return_wall_s=None if ret is None else end,frozen_confirmation_wall_s=frozen_end,effective_deadline_wall_s=effective_deadline)
            item.update(phase=proof['phase'],response_publication_ms=elapsed(publication),timing_endpoints=endpoints,endpoint_from_submit_ms={k:elapsed(v) for k,v in endpoints.items()})
            handoffs.append(item)
    for q in reqs.values():
        if q['kind'] in ('tick','control'):validate('BrakeOnly',lambda q=q:require_parked_execution(cfg,q['args']['command'],q['kind']=='tick'))
    measured=[x for x in snaps if x.get('phase')=='stationary'];check('Stationary',bool(measured) and all(x['speed_norm']<=.1 for x in measured))
    expected_measured=list(range(captures[0]['reply']['observation']['frame'],captures[-1]['reply']['frozen']['frame']+1)) if captures else []
    check('MeasurementSnapshotCoverage',bool(expected_measured) and [s['frame'] for s in measured]==expected_measured)
    check('Feedback',all(x['feedback_matches'] and canonical(x['full_control'])==canonical(STOP) and x['command']['source'] in ('InitialProtectionBrake','FaultBrake') for x in snaps))
    collision_count=len(lines(root,'collisions.jsonl'));check('NoCollision',collision_count==0)
    cl=load(root/'cleanup_steps.json');cleanup=dict(sensor_stops=sum(x.get('passed',False) for x in cl.get('sensor_stops',[])),actor_destructions=sum(x.get('destroyed',False) for x in cl.get('actor_destruction',[])),settings_restored=cl.get('settings_restored') is True,remaining_actors=None if cl.get('remaining') is None else len(cl['remaining']),subprocesses_and_ports=sup.get('process_cleanup') is True,model_exited=sup.get('model_exit_code')==0,shared_memory='NotUsed',late_tick_rpc_cleanup='NotTriggeredUnlessExplicitUnknownAdvance_PriorNoServerEvidence')
    clean=cleanup['sensor_stops']==8 and cleanup['actor_destructions']==9 and cleanup['settings_restored'] and cleanup['remaining_actors']==0 and cleanup['subprocesses_and_ports'] and cleanup['model_exited'] and result.get('remaining_processes')==[]
    owner_state=load(root/'owner_state.json')
    uncertainty=(result.get('stop',{}).get('status')=='HandoffUncertain' or bool(load(root/'io_handoff_uncertain.json')) or result.get('server_advance_unknown',False)
                 or owner_state.get('handoff_uncertain',False) or bool(owner_state.get('failed_operation') or owner_state.get('control_uncertainty'))
                 or any(not x['passed'] for x in all_handoffs) or any(x['kind']=='tick' and not x['owner_consumed'] for x in failures))
    termination=termination_evidence(snaps,fault,brake,uncertainty)
    ledger=lines(root.parent,'budget.jsonl');starts=sum(x['event']=='carla_start' for x in ledger);model_starts=sum(x['event']=='model_start' for x in ledger)
    # When owner was terminated, retain the conservative supervisor charge and
    # independently enumerate submitted attempts. Never fabricate result.json.
    phases={}
    for q in reqs.values():
        if q['kind']=='tick':phases[q['args']['phase']]=phases.get(q['args']['phase'],0)+1
    charges={}
    for kind,key in (('setup','settings_startup'),('cleanup','settings_restore')):
        if any(q['kind']==kind for q in reqs.values()):charges[key]=cfg.get('tick_caps',{}).get(key,30)
    settings=result.get('settings_charges',charges);phase_ticks=result.get('phase_ticks',phases)
    ledger_charge=next((x.get('charged_ticks') for x in reversed(ledger) if x['event']=='finished'),None)
    charged=ledger_charge if ledger_charge is not None else result.get('charged_ticks')
    no_model_tasks=bool(ready) and not tasks and not attempts and not pid and not diagnostics and not replies and not consumed
    def final_count(key):return final.get(key,0 if no_model_tasks else None)
    budget=dict(starts=starts,max_starts=1,model_starts=model_starts,active_ticks=sum(phase_ticks.values()),observed_ticks=len(snaps),phase_ticks=phase_ticks,settings_charges=settings,charged_ticks=charged,charge_policy='owner final accounting' if result else 'supervisor conservative whole reservation; owner final missing',enumerated_attempts_and_settings=sum(phases.values())+sum(charges.values()),unconfirmed_tick_attempts=max(0,sum(phases.values())-len(snaps)),max_ticks=800,full_forward_attempts=len(attempts),max_forwards=24,pid_calls_total=len(pid)+cfg.get('preflight_pid_calls',0),max_pid_calls=24,training=final_count('training'),optimizer_steps=final_count('optimizer_steps'),model_controls=final_count('model_controls'),driving_budget_consumed=0)
    budget_ok=starts==model_starts==1 and budget['charged_ticks'] is not None and budget['charged_ticks']==budget['active_ticks']+sum(budget['settings_charges'].values())<=800 and budget['full_forward_attempts']<=24 and budget['pid_calls_total']<=24
    successes=final_count('successful_forwards');unchanged=final.get('fixed_state_unchanged') is True and sup.get('assets_unchanged') is True
    success_source=('model_final counter; independently backed by output diagnostics when present' if 'successful_forwards' in final else 'No published model task and no forward/PID/diagnostic/completion evidence; model ready observed' if no_model_tasks else 'Unknown: model_final successful_forwards counter absent; attempts are not proof of success or failure')
    counts=dict(planned_targets=planned,collected=len(captures),submitted=len(tasks),not_submitted_after_collection=max(0,len(captures)-len(tasks)),unreached_targets=max(0,planned-len(captures)),forward_attempts=len(attempts),forward_success=successes,forward_failed=None if successes is None else max(0,len(attempts)-successes),forward_success_count_source=success_source,published=len(replies),necessary_output_pid_checks=len(passed_outputs),owner_complete_timely=len(passed_consumption),old_realtime_passed=old_pass,model_controls=final.get('model_controls'),pid_live=len(pid),pid_preflight=cfg.get('preflight_pid_calls',0),pid_total=budget['pid_calls_total'])
    counts.update(published_notifications=len(replies),owner_consumptions_recorded=len(consumed))
    tick_requests=[q for q in reqs.values() if q['kind']=='tick']
    tick_responses=[load(root/'io_responses'/('%06d.json'%q['sequence'])) for q in tick_requests]
    tick_confirmed=[x for x in ops if x['operation']['kind']=='tick' and x['reply'].get('success') and x['owner_consumed_wall_s']<=x['operation']['deadline']]
    budget.update(ledger_charge=ledger_charge,owner_enumerated_charge=result.get('charged_ticks'),
                  conservative_reservation=charged is not None and charged>sum(phases.values())+sum(charges.values()),
                  tick_submissions=len(tick_requests),tick_response_files=sum(bool(x) for x in tick_responses),
                  tick_rpc_return_recorded=sum('rpc_return_wall_s' in x for x in tick_responses),
                  tick_owner_confirmed=len(tick_confirmed))
    counts['model_controls']=budget['model_controls']
    initial_states=[s for s in snaps if s.get('phase')=='initialization']
    check('InitializationAdmission',len(initial_states)>=21 and all(s['speed_norm']<=.1 for s in initial_states[-21:]))
    if cfg.get('parked_handoff_revision'):
        check('FixedHandoffCounts',len(handoffs)==planned*2 and all(sum(x['source']==source for x in handoffs)==planned for source in ('PlannedHoldBrake','InitialProtectionBrake')))
    complete=all(counts[k]==planned for k in ('collected','submitted','forward_attempts','forward_success','necessary_output_pid_checks','owner_complete_timely','pid_live'))
    passed=complete and not errors and not failures and termination['normal'] and termination['terminal_hold_verified'] and clean and budget_ok and unchanged and final.get('model_controls')==final.get('training')==final.get('optimizer_steps')==0
    origin=min((p['task']['input_wall_s'] for p in tasks.values()),default=None);end=consumed[-1]['consume_return_wall_s'] if consumed else None;span=None if origin is None or end is None else end-origin
    mspan=measured[-1]['tick_end_wall_s']-measured[0]['input_wall_s'] if measured else None
    speeds=[v for d in diagnostics.values() for v in d.get('native_speeds',[])];targets=[d['pid_metadata']['desired_speed'] for d in diagnostics.values()]
    init=[x for x in ops if x['operation']['kind']=='tick' and reqs[x['operation']['sequence']]['args']['phase']=='initialization']
    out=dict(status='Passed_FiniteParkedLogicalShadowOnly' if passed else 'Partial',protocol=cfg.get('shadow_protocol'),J13_A='Partial',run_commit=request.get('commit'),config_sha256=request.get('config_sha256'),run_id=root.name,counts=counts,audit_errors=errors,failed_checks=errors,evidence_errors=evidence_errors,
      unmatched_records=dict(forward_attempts=[x['sequence'] for x in attempts if x['sequence'] not in replies],io_requests_without_owner_consumption=unconsumed),
      handoff=dict(count=len(all_handoffs),failed=sum(not x['passed'] for x in all_handoffs),over_200_through_500ms=sum(x['whole_operation_ms'] is not None and 200<x['whole_operation_ms']<=500 for x in all_handoffs)),
      deadline=dict(logic_complete_samples=len(passed_consumption),old_realtime_wall_expired=sum(x.get('old_realtime_reason')=='PlanWallExpired' for x in consumed),sim_age_s=distribution([x['sim_age_s'] for x in consumed]),whole_run_acceptance=passed,first_fault=fault.get('reason'),active_target=fault.get('active_target'),failed_operation=fault.get('failed_operation') or (failures[0]['operation'] if failures else None),failed_operations=failures),
      handoffs=dict(count=len(handoffs),within_200ms=sum(x['passed'] and x['whole_operation_ms']<=200 for x in handoffs),over_200_through_500ms=sum(x['passed'] and 200<x['whole_operation_ms']<=500 for x in handoffs),failed=sum(not x['passed'] for x in handoffs),operation_ms=distribution([x['whole_operation_ms'] for x in handoffs if x['whole_operation_ms'] is not None]),source_counts={n:sum(x['source']==n for x in handoffs) for n in ('PlannedHoldBrake','InitialProtectionBrake')},endpoint_from_submit_ms={k:distribution([x['endpoint_from_submit_ms'][k] for x in handoffs if x['endpoint_from_submit_ms'][k] is not None]) for k in (handoffs[0]['endpoint_from_submit_ms'] if handoffs else [])},records=handoffs,no_causal_claim_from_single_run=True),
      timings={k:dist(v) for k,v in metrics.items()},performance=dict(measurement_to_last_consume_wall_s=span,forward_frequency_over_same_window_hz=ratio(len(attempts),span),timely_completion_frequency_hz=ratio(len(passed_consumption),span),measurement_tick_span_wall_s=mspan,measurement_physics_advance_s=len(measured)*.05,measurement_sim_to_wall_ratio=ratio(len(measured)*.05,mspan),total_lifecycle_wall_s=ledger[-1]['wall_s']-ledger[0]['wall_s'] if ledger else None,model_load_wall_s=ready.get('load_wall_s'),total_confirmed_freeze_wait_s=sum(metrics['freeze_wait_ms'])/1000,gpu_peak_allocated=max((d['gpu_allocated'] for d in diagnostics.values()),default=None),gpu_peak_reserved=max((d['gpu_reserved'] for d in diagnostics.values()),default=None),small_sample=True),
      initialization=dict(ticks=len(init),maximum_operation_ms=max(((x['owner_consumed_wall_s']-x['operation']['started'])*1000 for x in init),default=None)),
      output=dict(native_speed_range_mps=[min(speeds),max(speeds)] if speeds else None,pid_target_speed_range_mps=[min(targets),max(targets)] if targets else None,full_throttle_count=sum(d['raw_pid']['throttle']==1 for d in diagnostics.values()),speed_horizon_exceeds_5mps_count=sum(max(d['native_speeds'])>5 for d in diagnostics.values()),memory_checked=memory_checked,params_assets_unchanged=unchanged,moving_transport_validation=False),
      isolation=dict(model_controls=budget['model_controls'],zero_count_evidence='No published model task, no forward/PID ledger, brake-only execution requests' if no_model_tasks else 'model final and execution requests',measurement_max_speed_mps=max((s['speed_norm'] for s in measured),default=None),initial_fall_separate=True,hold_physics_ticks=sum(s['command']['source']=='PlannedHoldBrake' for s in snaps),collisions=collision_count),termination=termination,cleanup=cleanup,budget=budget,S0_review_ready=passed,S0_driving_authorized=False,limitations=['Finite parked evidence only; no moving handoff reliability or real-time claim','Old failures preserved','No retries'])
    out['shutdown_evidence']=dict(first_fault=fault,processes=(sup.get('shutdown') or {}).get('processes'),
        missing_child_evidence=sup.get('missing_child_evidence'),finalization_errors=sup.get('finalization_errors'),
        actor_settings_verified=clean,owner_final_present=bool(result),model_final_present=bool(final),
        model_normal_exit=((sup.get('shutdown') or {}).get('processes',{}).get('model',{}).get('normal_exit')
                           if 'shutdown' in sup else True if final and sup.get('model_exit_code')==0 else None))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(status=r['status'],counts=r['counts'],handoffs=r['handoffs'],termination=r['termination'],audit_errors=r['audit_errors']),indent=2))
if __name__=='__main__':main()
