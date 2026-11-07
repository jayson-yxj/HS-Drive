"""Read-only logical shadow audit, never imports model or executes PID/CARLA."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.m8j13_logic_init_report import lines,distribution
from tools.m8j13_common import digest
from simulator.m8j13_logic_shadow import ShadowChannel,memory_valid,require_parked_execution
from simulator.m8j13_logic_runtime import frozen_equal
from simulator.m8j13_sensor_audit import check_group
from simulator.m8j13_safety import sanitize_control,validate_trajectory,transport_native_path
from datasets.m8j12_contract import native_target


def load(p):return json.loads(p.read_text())
def dist(values):
    return dict(first=values[0] if values else None,following=distribution(values[1:]),all=distribution(values))


def audit(root):
    cfg=load(root/'frozen_config.json');result=load(root/'result.json');sup=load(root/'supervisor.json');final=load(root/'model_final.json');ready=load(root/'model_ready.json');request=load(root/'request.json')
    ops=lines(root,'io_consumption.jsonl');snaps=lines(root,'snapshots.jsonl');consumed=lines(root,'shadow_consumed.jsonl');attempts=lines(root,'forward_attempts.jsonl');pid=lines(root,'pid_attempts.jsonl');events=lines(root,'owner_events.jsonl');timing=lines(root,'worker_timing.jsonl')
    reqs={int(p.stem):load(p) for p in (root/'io_requests').glob('*.json')}
    tasks=[load(p) for p in sorted((root/'shadow_tasks').glob('*.json'))];replies=[load(p) for p in sorted((root/'shadow_completions').glob('*.json'))];diagnostics=[load(p) for p in sorted((root/'diagnostics').glob('*.json'))]
    captures=[x for x in ops if x['operation']['kind']=='collect' and x['reply']['success']]
    setup=next(x['reply'] for x in ops if x['operation']['kind']=='setup');identity=setup['identity'];actors=setup['sensor_actors'];channels=ShadowChannel(identity)
    assert len(tasks)==len(replies)==len(consumed)==len(diagnostics)==len(attempts)==len(pid)==final['attempts']==final['successful_forwards']
    assert final['fixed_state_unchanged'] and sup['assets_unchanged'] and final['model_controls']==final['training']==final['optimizer_steps']==0
    assert ready['initial_memory']['history_timestamp'] is None and not ready['initial_memory']['cached']
    assert all(ready[k] for k in ('parameters_frozen','eval','fp32'))
    assert all(x['feedback_matches'] and x['command']['source']=='InitialProtectionBrake' and x['full_control']['throttle']==0 and x['full_control']['brake']==1 for x in snaps)
    for r in reqs.values():
        if r['kind'] in ('control','tick'):require_parked_execution(cfg,r['args']['command'],r['kind']=='tick')
    measured=[x for x in snaps if x['phase']=='stationary'];assert all(x['speed_norm']<=.1 for x in measured)
    assert not lines(root,'collisions.jsonl')
    previous=None;metrics={k:[] for k in ('six_camera_arrival_ms','six_camera_copy_ms','archive_before_submit_ms','freeze_confirmation_ms','submission_to_worker_ms','e2e_pid_ms','e2e_checks_ms','e2e_owner_ms','capture_ms','worker_read_ms','cpu_preprocess_ms','gpu_transfer_ms','forward_decode_ms','hs_ms','transport_ms','pid_ms','necessary_checks_ms','publish_ms','owner_after_publication_ms','task_to_consume_ms','validation_to_consume_ms','freeze_wait_ms','archive_ms')}
    for i,(packet,reply,d,c) in enumerate(zip(tasks,replies,diagnostics,consumed)):
        t=packet['task'];row=packet['row'];cap=captures[i]['reply'];obs=cap['observation']
        assert t['identity']==identity and t['sequence']==i+1 and obs['frame']==row['frame_id']==t['frame']
        generated=channels.request(obs,t['frozen'],t['requested_wall_s'],t['run_deadline']);assert generated==t
        conf=next(x['reply']['snapshot'] for x in ops if x['operation']['kind']=='frozen' and reply['times']['publish_start']<=x['operation']['started']<=c['consume_return_wall_s'])
        channels.consume(reply,conf,c['consume_reference_wall_s']);channels.finalize_return(reply,c['consume_return_wall_s'])
        assert c['consume_return_wall_s']>=c['consume_reference_wall_s']>=c['received_wall_s'] and c['old_realtime_reason']=='PlanWallExpired'
        assert not any(t['requested_wall_s']<=s['input_wall_s']<=c['consume_return_wall_s'] for s in snaps)
        assert frozen_equal(t['frozen'],conf)
        memory_valid(d['memory_before'],d['memory_after'],previous,row['timestamp_s']);previous=row['timestamp_s']
        original=np.asarray(d['native_path']);transported=transport_native_path(original,row['ego_world_matrix'],t['frozen']['pose'])
        assert np.array_equal(transported,np.asarray(d['transport']['transported_path']))
        assert np.array_equal(native_target(t['frozen']['pose'],row['target_world_xy']),np.asarray(d['transport']['current_navigation_target']))
        assert d['transport']['current_speed_mps']==t['frozen']['speed'] and d['hs_timestamp']==row['timestamp_s']
        assert sanitize_control(d['raw_pid'])[1] is None and sanitize_control(d['native_agent_control'])[1] is None
        limits=load(Path('configs/m8j13_protocol.json'))['trajectory_checks'];assert validate_trajectory(d['hs_points'],row['speed_mps'],limits)[0] is None
        tt=d['times'];input_wall=t['input_wall_s'];event=next(e for e in events if e['event']=='shadow_task_submitted' and e['task']['sequence']==t['sequence'])
        values=dict(six_camera_arrival_ms=(max(x['receive_monotonic_s'] for x in cap['shadow_capture']['camera_metadata'])-input_wall)*1000,six_camera_copy_ms=(max(x['copied_monotonic_s'] for x in cap['shadow_capture']['camera_metadata'])-input_wall)*1000,archive_before_submit_ms=(cap['shadow_capture']['archive_done_wall_s']-cap['capture_complete_wall_s'])*1000,freeze_confirmation_ms=(event['frozen_confirmed_wall_s']-captures[i]['owner_consumed_wall_s'])*1000,submission_to_worker_ms=(tt['worker_received']-t['requested_wall_s'])*1000,e2e_pid_ms=(tt['pid_done']-input_wall)*1000,e2e_checks_ms=(tt['checks_done']-input_wall)*1000,e2e_owner_ms=(c['consume_return_wall_s']-input_wall)*1000,
            capture_ms=(captures[i]['owner_consumed_wall_s']-input_wall)*1000,worker_read_ms=(tt['read_done']-tt['worker_received'])*1000,cpu_preprocess_ms=(tt['preprocess_done']-tt['preprocess_start'])*1000,
            gpu_transfer_ms=d['gpu_transfer_ms'],forward_decode_ms=(tt['forward_done']-tt['forward_start'])*1000,hs_ms=(tt['hs_done']-tt['forward_done'])*1000,transport_ms=(tt['transport_done']-tt['hs_done'])*1000,
            pid_ms=(tt['pid_done']-tt['transport_done'])*1000,necessary_checks_ms=(tt['checks_done']-tt['pid_done'])*1000,publish_ms=(d['published_wall_s']-tt['checks_done'])*1000,
            owner_after_publication_ms=(c['consume_return_wall_s']-d['published_wall_s'])*1000,task_to_consume_ms=(c['consume_return_wall_s']-t['requested_wall_s'])*1000,
            validation_to_consume_ms=(c['consume_return_wall_s']-t['validation_start'])*1000 if 'validation_start' in t else (c['consume_return_wall_s']-reply['validation_start'])*1000,
            freeze_wait_ms=(c['consume_return_wall_s']-event['frozen_confirmed_wall_s'])*1000,archive_ms=timing[i]['archive_ms'])
        for k,v in values.items():metrics[k].append(v)
    for cap in captures:
        r=cap['reply'];row=r['shadow_capture']['row'];obs=r['observation'];assert r['frozen']['frame']==obs['frame']+2
        check=check_group(r['shadow_capture']['camera_metadata'],dict(frame=obs['frame'],elapsed_seconds=obs['sim_s']),identity,actors);assert check['valid']
        assert row['imu_frame']==obs['frame'] and abs(row['imu_timestamp_s']-obs['sim_s'])<=1e-5
        assert cap['owner_consumed_wall_s']<=obs['input_wall_s']+.5
    assert all(b['reply']['observation']['frame']-a['reply']['observation']['frame']==4 for a,b in zip(captures,captures[1:]))
    overdue=[x for x in ops if x['owner_consumed_wall_s']>x['operation']['deadline'] or not x['reply']['success']]
    failure=[]
    for x in overdue:
        o=x['operation'];r=x['reply'];failure.append(dict(kind=o['kind'],source=reqs[o['sequence']]['args'].get('command',{}).get('source'),whole_operation_ms=(x['owner_consumed_wall_s']-o['started'])*1000,
            io_queue_ms=(r['io_received_wall_s']-o['started'])*1000,rpc_ms=None if 'rpc_start_wall_s' not in r else (r['rpc_return_wall_s']-r['rpc_start_wall_s'])*1000,
            owner_after_deadline_ms=(x['owner_consumed_wall_s']-o['deadline'])*1000,error=r.get('error'),server_processing_confirmed=r.get('server_processed',False)))
    cl=load(root/'cleanup_steps.json');assert cl['complete'] and cl['remaining']==[] and cl['settings_restored'] and len(cl['sensor_stops'])==8 and len(cl['actor_destruction'])==9
    assert all(x['passed'] for x in cl['sensor_stops']) and all(x['destroyed'] for x in cl['actor_destruction'])
    assert sup['process_cleanup'] and result['remaining_processes']==[] and sup['model_exit_code']==0
    ledger=lines(root.parent,'budget.jsonl');assert len([x for x in ledger if x['event']=='carla_start'])==1 and len([x for x in ledger if x['event']=='model_start'])==1
    assert len(snaps)==sum(result['phase_ticks'].values()) and result['charged_ticks']==len(snaps)+60<=800
    span=(consumed[-1]['consume_return_wall_s']-tasks[0]['task']['input_wall_s']) if consumed else None
    measured_span=measured[-1]['tick_end_wall_s']-measured[0]['input_wall_s'] if measured else None
    rawspeed=[x for d in diagnostics for x in d['native_speeds']];targets=[d['pid_metadata']['desired_speed'] for d in diagnostics]
    report=dict(status='Partial',protocol=cfg['shadow_protocol'],J13_A='Partial',run_commit=request['commit'],config_sha256=request['config_sha256'],run_id=root.name,
        counts=dict(planned_targets=24,collected=len(captures),not_submitted_after_collection=len(captures)-len(tasks),unreached_targets=24-len(captures),forward_attempts=len(attempts),forward_success=final['successful_forwards'],forward_failed=len(attempts)-final['successful_forwards'],necessary_output_pid_checks=len(replies),owner_complete_timely=len(consumed),old_realtime_passed=sum(x['old_realtime_reason'] is None for x in consumed),model_controls=0,pid_live=len(pid),pid_preflight=cfg['preflight_pid_calls'],pid_total=len(pid)+cfg['preflight_pid_calls']),
        deadline=dict(logic_complete_samples=len(consumed),old_realtime_wall_expired=len(consumed),sim_age_s=distribution([x['sim_age_s'] for x in consumed]),whole_run_acceptance=False,first_fault=result['fault']['reason'],failed_operations=failure),
        timings={k:dist(v) for k,v in metrics.items()},performance=dict(measurement_to_last_consume_wall_s=span,forward_frequency_over_same_window_hz=None if not span else len(attempts)/span,timely_completion_frequency_hz=None if not span else len(consumed)/span,measurement_tick_span_wall_s=measured_span,measurement_physics_advance_s=len(measured)*.05,measurement_sim_to_wall_ratio=None if not measured_span else len(measured)*.05/measured_span,total_lifecycle_wall_s=ledger[-1]['wall_s']-ledger[0]['wall_s'],model_load_wall_s=ready['load_wall_s'],total_confirmed_freeze_wait_s=sum(metrics['freeze_wait_ms'])/1000,partial_final_freeze_wait_s=None,final_failed_handoff_wall_s=None if not failure else failure[0]['whole_operation_ms']/1000,wall_tick_frequency_hz=None if not measured_span else len(measured)/measured_span,full_activity_to_fault_wall_s=result['fault']['wall_s']-measured[0]['input_wall_s'],forward_frequency_full_activity_hz=len(attempts)/(result['fault']['wall_s']-measured[0]['input_wall_s']),gpu_peak_allocated=max(d['gpu_allocated'] for d in diagnostics),gpu_peak_reserved=max(d['gpu_reserved'] for d in diagnostics),small_sample=True),
        output=dict(native_speed_range_mps=[min(rawspeed),max(rawspeed)],pid_target_speed_range_mps=[min(targets),max(targets)],full_throttle_count=sum(d['raw_pid']['throttle']==1 for d in diagnostics),speed_horizon_exceeds_5mps_count=sum(max(d['native_speeds'])>5 for d in diagnostics),trajectory_failed=0,finite_passed=len(diagnostics),current_speed_mps=0.,memory_passed=True,input_interval_sim_s=.2,params_assets_unchanged=True,moving_transport_validation=False,raw_and_agent_equal=all(d['raw_pid']==d['native_agent_control'] for d in diagnostics)),
        isolation=dict(model_controls=0,all_observed_ticks_independent_brake=True,measurement_max_speed_mps=max(s['speed_norm'] for s in measured),initial_fall_separate=True,hold_physics_ticks=0,collisions=0),
        termination=dict(normal=False,status=result['stop']['status'],fault_brake_server_processed=result['stop']['brake']['server_processed'],post_fault_ticks=0,terminal_hold_verified=False,reason='Ordinary hold control timed out; no physics continuation allowed while handoff uncertain'),
        cleanup=dict(sensor_stops=8,actor_destructions=9,settings_restored=True,remaining_actors=0,subprocesses_and_ports=True,model_exited=True,shared_memory='NotUsed',late_tick_rpc_cleanup='NotTriggered_PriorNoServerEvidenceOnly',ordinary_control_timeout_cleanup='Actual bounded cleanup after failed control response; no cancellation or stop claim'),
        budget=dict(starts=1,max_starts=1,model_starts=1,active_ticks=len(snaps),phase_ticks=result['phase_ticks'],settings_charges=result['settings_charges'],charged_ticks=result['charged_ticks'],max_ticks=800,full_forward_attempts=len(attempts),max_forwards=24,pid_calls_total=len(pid)+cfg['preflight_pid_calls'],max_pid_calls=24,training=0,optimizer_steps=0,model_controls=0,driving_budget_consumed=0),
        limitations=['24 targets not completed','Unplanned control handoff deadline failure','No terminal physical hold evidence','One synthetic preflight PID call consumed total budget; not actual stopping cause','No real-time or driving admission','No retries'],S0_review_ready=False)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(counts=r['counts'],deadline=r['deadline'],budget=r['budget']),indent=2))
if __name__=='__main__':main()
