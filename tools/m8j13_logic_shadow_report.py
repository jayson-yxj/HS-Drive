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
        if legal and s['speed_norm']<=.1:stable.append(s)
        else:stable=[]
    hold=stable[-1]['sim_s']-stable[0]['sim_s'] if stable else 0.
    processed=brake.get('server_processed') is True
    ordinary_known=not uncertain and not brake.get('handoff_uncertain',True)
    ordered=bool(rows) and all(s['input_wall_s']>=brake.get('return_wall_s',float('inf')) for s in rows)
    bounded=bool(rows) and len(rows)<=120 and rows[-1]['tick_end_wall_s']-fault.get('wall_s',0)<=10
    passed=processed and ordinary_known and ordered and bounded and hold>=1.-1e-6
    return dict(normal=reason=='NormalShadowCompletion',first_reason=reason,post_fault_ticks=len(rows),terminal_hold_sim_s=hold,terminal_hold_verified=passed,
                fault_brake_server_processed=processed,ordinary_handoff_confirmed=ordinary_known,handoff_uncertain=not ordinary_known,
                status='PhysicalHoldObserved' if passed else ('HandoffUncertain' if not ordinary_known else 'HoldNotVerified'))


def audit(root):
    cfg=load(root/'frozen_config.json');result=load(root/'result.json');sup=load(root/'supervisor.json');final=load(root/'model_final.json');ready=load(root/'model_ready.json');request=load(root/'request.json')
    ops=lines(root,'io_consumption.jsonl');snaps=lines(root,'snapshots.jsonl');consumed=lines(root,'shadow_consumed.jsonl');attempts=lines(root,'forward_attempts.jsonl');pid=lines(root,'pid_attempts.jsonl');events=lines(root,'owner_events.jsonl')
    reqs={int(p.stem):load(p) for p in (root/'io_requests').glob('*.json')};publications={x['sequence']:x for x in lines(root,'io_publications.jsonl')}
    tasks={load(p)['task']['sequence']:load(p) for p in (root/'shadow_tasks').glob('*.json')};replies=index_files(root/'shadow_completions','sequence')
    diagnostics={load(p)['task']['sequence']:load(p) for p in (root/'diagnostics').glob('*.json')};timing={x['sequence']:x for x in lines(root,'worker_timing.jsonl')}
    setup=next((x['reply'] for x in ops if x['operation']['kind']=='setup' and x['reply'].get('success')),{});identity=setup.get('identity');actors=setup.get('sensor_actors',{})
    captures=[x for x in ops if x['operation']['kind']=='collect' and x['reply'].get('success')];byframe={x['reply']['observation']['frame']:x for x in captures}
    fault=load(root/'watchdog_fault.json') or result.get('fault') or {};brake=load(root/'fault_brake.json');errors=[]
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
    for x in ops:
        o=x['operation'];r=x['reply'];q=reqs.get(o['sequence'],{});end=x['owner_consumed_wall_s'];ok=r.get('success') and end<=o['deadline']
        check('OperationIdentity',r.get('operation')==o and q.get('operation')==o)
        item=dict(kind=o['kind'],source=q.get('args',{}).get('command',{}).get('source'),operation_limit_s=o['operation_limit_s'],whole_operation_ms=(end-o['started'])*1000,io_queue_ms=(r['io_received_wall_s']-o['started'])*1000,rpc_ms=None if 'rpc_start_wall_s' not in r else (r['rpc_return_wall_s']-r['rpc_start_wall_s'])*1000,owner_after_deadline_ms=(end-o['deadline'])*1000,error=r.get('error'),server_processing_confirmed=r.get('server_processed',False),passed=bool(ok))
        if not ok:failures.append(item)
        if o['kind']=='control':all_handoffs.append(item)
        proof=q.get('args',{}).get('parked_handoff')
        if proof:
            check('HandoffPhaseDeadline',o['deadline']<=min(o['started']+.5,proof['phase_deadline'],o['run_deadline']))
            item.update(phase=proof['phase'],response_publication_ms=None if o['sequence'] not in publications else (publications[o['sequence']]['publication_done_wall_s']-o['started'])*1000)
            handoffs.append(item)
    for q in reqs.values():
        if q['kind'] in ('tick','control'):validate('BrakeOnly',lambda q=q:require_parked_execution(cfg,q['args']['command'],q['kind']=='tick'))
    measured=[x for x in snaps if x.get('phase')=='stationary'];check('Stationary',all(x['speed_norm']<=.1 for x in measured))
    check('Feedback',all(x['feedback_matches'] and canonical(x['full_control'])==canonical(STOP) and x['command']['source'] in ('InitialProtectionBrake','FaultBrake') for x in snaps))
    collision_count=len(lines(root,'collisions.jsonl'));check('NoCollision',collision_count==0)
    cl=load(root/'cleanup_steps.json');cleanup=dict(sensor_stops=sum(x.get('passed',False) for x in cl.get('sensor_stops',[])),actor_destructions=sum(x.get('destroyed',False) for x in cl.get('actor_destruction',[])),settings_restored=cl.get('settings_restored') is True,remaining_actors=None if cl.get('remaining') is None else len(cl['remaining']),subprocesses_and_ports=sup.get('process_cleanup') is True,model_exited=sup.get('model_exit_code')==0,shared_memory='NotUsed',late_tick_rpc_cleanup='NotTriggeredUnlessExplicitUnknownAdvance_PriorNoServerEvidence')
    clean=cleanup['sensor_stops']==8 and cleanup['actor_destructions']==9 and cleanup['settings_restored'] and cleanup['remaining_actors']==0 and cleanup['subprocesses_and_ports'] and cleanup['model_exited'] and result.get('remaining_processes')==[]
    owner_state=load(root/'owner_state.json')
    uncertainty=(result.get('stop',{}).get('status')=='HandoffUncertain' or bool(load(root/'io_handoff_uncertain.json')) or result.get('server_advance_unknown',False)
                 or owner_state.get('handoff_uncertain',False) or bool(owner_state.get('failed_operation') or owner_state.get('control_uncertainty'))
                 or any(not x['passed'] for x in all_handoffs))
    termination=termination_evidence(snaps,fault,brake,uncertainty)
    ledger=lines(root.parent,'budget.jsonl');starts=sum(x['event']=='carla_start' for x in ledger);model_starts=sum(x['event']=='model_start' for x in ledger)
    budget=dict(starts=starts,max_starts=1,model_starts=model_starts,active_ticks=sum(result.get('phase_ticks',{}).values()),observed_ticks=len(snaps),phase_ticks=result.get('phase_ticks',{}),settings_charges=result.get('settings_charges',{}),charged_ticks=result.get('charged_ticks'),max_ticks=800,full_forward_attempts=len(attempts),max_forwards=24,pid_calls_total=len(pid)+cfg.get('preflight_pid_calls',0),max_pid_calls=24,training=final.get('training'),optimizer_steps=final.get('optimizer_steps'),model_controls=final.get('model_controls'),driving_budget_consumed=0)
    budget_ok=starts==model_starts==1 and budget['charged_ticks'] is not None and budget['charged_ticks']==budget['active_ticks']+sum(budget['settings_charges'].values())<=800 and budget['full_forward_attempts']<=24 and budget['pid_calls_total']<=24
    successes=final.get('successful_forwards',0);unchanged=final.get('fixed_state_unchanged') is True and sup.get('assets_unchanged') is True
    counts=dict(planned_targets=planned,collected=len(captures),submitted=len(tasks),not_submitted_after_collection=max(0,len(captures)-len(tasks)),unreached_targets=max(0,planned-len(captures)),forward_attempts=len(attempts),forward_success=successes,forward_failed=max(0,len(attempts)-successes),forward_success_count_source='model_final counter; independently backed by output diagnostics when present',published=len(replies),necessary_output_pid_checks=len(passed_outputs),owner_complete_timely=len(passed_consumption),old_realtime_passed=old_pass,model_controls=final.get('model_controls'),pid_live=len(pid),pid_preflight=cfg.get('preflight_pid_calls',0),pid_total=budget['pid_calls_total'])
    counts.update(published_notifications=len(replies),owner_consumptions_recorded=len(consumed))
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
      unmatched_records=dict(forward_attempts=[x['sequence'] for x in attempts if x['sequence'] not in replies]),
      handoff=dict(count=len(all_handoffs),failed=sum(not x['passed'] for x in all_handoffs),over_200_through_500ms=sum(200<x['whole_operation_ms']<=500 for x in all_handoffs)),
      deadline=dict(logic_complete_samples=len(passed_consumption),old_realtime_wall_expired=sum(x.get('old_realtime_reason')=='PlanWallExpired' for x in consumed),sim_age_s=distribution([x['sim_age_s'] for x in consumed]),whole_run_acceptance=passed,first_fault=fault.get('reason'),active_target=fault.get('active_target'),failed_operation=fault.get('failed_operation'),failed_operations=failures),
      handoffs=dict(count=len(handoffs),within_200ms=sum(x['passed'] and x['whole_operation_ms']<=200 for x in handoffs),over_200_through_500ms=sum(x['passed'] and 200<x['whole_operation_ms']<=500 for x in handoffs),failed=sum(not x['passed'] for x in handoffs),operation_ms=distribution([x['whole_operation_ms'] for x in handoffs]),source_counts={n:sum(x['source']==n for x in handoffs) for n in ('PlannedHoldBrake','InitialProtectionBrake')},no_causal_claim_from_single_run=True),
      timings={k:dist(v) for k,v in metrics.items()},performance=dict(measurement_to_last_consume_wall_s=span,forward_frequency_over_same_window_hz=ratio(len(attempts),span),timely_completion_frequency_hz=ratio(len(passed_consumption),span),measurement_tick_span_wall_s=mspan,measurement_physics_advance_s=len(measured)*.05,measurement_sim_to_wall_ratio=ratio(len(measured)*.05,mspan),total_lifecycle_wall_s=ledger[-1]['wall_s']-ledger[0]['wall_s'] if ledger else None,model_load_wall_s=ready.get('load_wall_s'),total_confirmed_freeze_wait_s=sum(metrics['freeze_wait_ms'])/1000,gpu_peak_allocated=max((d['gpu_allocated'] for d in diagnostics.values()),default=None),gpu_peak_reserved=max((d['gpu_reserved'] for d in diagnostics.values()),default=None),small_sample=True),
      initialization=dict(ticks=len(init),maximum_operation_ms=max(((x['owner_consumed_wall_s']-x['operation']['started'])*1000 for x in init),default=None)),
      output=dict(native_speed_range_mps=[min(speeds),max(speeds)] if speeds else None,pid_target_speed_range_mps=[min(targets),max(targets)] if targets else None,full_throttle_count=sum(d['raw_pid']['throttle']==1 for d in diagnostics.values()),speed_horizon_exceeds_5mps_count=sum(max(d['native_speeds'])>5 for d in diagnostics.values()),memory_checked=memory_checked,params_assets_unchanged=unchanged,moving_transport_validation=False),
      isolation=dict(model_controls=final.get('model_controls'),measurement_max_speed_mps=max((s['speed_norm'] for s in measured),default=None),initial_fall_separate=True,hold_physics_ticks=sum(s['command']['source']=='PlannedHoldBrake' for s in snaps),collisions=collision_count),termination=termination,cleanup=cleanup,budget=budget,S0_review_ready=passed,S0_driving_authorized=False,limitations=['Finite parked evidence only; no moving handoff reliability or real-time claim','Old failures preserved','No retries'])
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(status=r['status'],counts=r['counts'],handoffs=r['handoffs'],termination=r['termination'],audit_errors=r['audit_errors']),indent=2))
if __name__=='__main__':main()
