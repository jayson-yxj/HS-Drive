"""Read-only scenario audit. Aggregate export; private event streams stay private."""
import json
from collections import Counter
from pathlib import Path
from tools.m8j13_logic_init_report import lines,distribution
from simulator.m8j13_logic_faults import REASONS


def load(p):return json.loads(p.read_text()) if p.exists() else {}


def audit(root):
    cfg=load(root/'frozen_config.json');name=cfg['fault_injection']['scenario'];request=load(root/'request.json')
    result=load(root/'result.json');supervisor=load(root/'supervisor.json');fault=load(root/'watchdog_fault.json');brake=load(root/'fault_brake.json')
    snaps=lines(root,'snapshots.jsonl');consumption=lines(root,'io_consumption.jsonl');events=lines(root,'owner_events.jsonl')
    requests=[load(p) for p in sorted((root/'io_requests').glob('*.json'))]
    tasks=[load(p) for p in sorted((root/'tasks').glob('*.json'))];outputs=[load(p) for p in sorted((root/'task_results').glob('*.json'))]
    checks={}
    def check(key,condition):checks[key]=bool(condition)
    check('first_fault',fault.get('reason')==REASONS[name] and result.get('fault')==fault)
    check('initialization',result.get('initialization',{}).get('stable_ticks',0)>=21)
    check('six_stationary',len(result.get('stationary_cycles',[]))==6)
    pre=result.get('injection',{}).get('boundary',{})
    check('forward_motion',1<=pre.get('speed',-1)<1.7 and pre.get('speed_norm',100)<1.7)
    check('no_collision',not lines(root,'collisions.jsonl'))
    check('one_operation_at_a_time',all(a['owner_consumed_wall_s']<=b['operation']['started'] for a,b in zip(consumption,consumption[1:])))
    allowed_error=lambda x:name=='missing_camera' and x['operation']['kind']=='collect' and x['reply'].get('rejection_reason')=='MissingCamera'
    check('operation_identity_and_deadline',bool(consumption) and len(consumption)==len(requests) and all(x['reply']['operation']==x['operation'] and x['owner_consumed_wall_s']<=x['operation']['deadline'] and (x['reply']['success'] or allowed_error(x)) for x in consumption))
    normal_after=[r for r in requests if r['kind']=='control' and r['args']['command']['source']!='FaultBrake' and r['owner_submitted_wall_s']>=fault.get('wall_s',float('inf'))]
    check('no_normal_after_latch',not normal_after and not [x for x in events if x['event']=='task_accepted' and x['wall_s']>=fault.get('wall_s',float('inf'))])
    check('brake_processed',brake.get('server_processed') and not brake.get('handoff_uncertain') and brake.get('return_wall_s',float('inf'))<=brake.get('operation',{}).get('deadline',0))
    after=[s for s in snaps if s['phase']=='fault_stop'];ordinary=[s for s in snaps if s['phase']!='fault_stop']
    check('no_hold_physics',all(s['command']['source']!='PlannedHoldBrake' for s in snaps))
    check('all_feedback',bool(snaps) and all(s['feedback_matches'] for s in snaps))
    check('fault_brake_feedback',bool(after) and all(s['command']['source']=='FaultBrake' and s['full_control']['throttle']==0 and s['full_control']['brake']==1 and s['input_wall_s']>=brake.get('return_wall_s',float('inf')) for s in after))
    check('no_uncertain_resume',not result.get('server_advance_unknown',True) and not result.get('stop_error') and result.get('stop',{}).get('status')=='PhysicalStopObserved')
    # Compute physical stop and hold from snapshots, never from runtime status alone.
    stable=[]
    for s in after:
        if s['speed_norm']<=.1:stable.append(s)
        else:stable=[]
    hold=stable[-1]['sim_s']-stable[0]['sim_s'] if stable else 0
    check('physical_hold',hold>=1-1e-6)
    observation_wall=after[-1]['tick_end_wall_s']-fault['wall_s'] if after and fault else None
    check('stop_budget',bool(after) and len(after)<=120 and observation_wall<=10)
    check('ordinary_before_fault',all(s['tick_end_wall_s']<fault.get('wall_s',0) for s in ordinary))
    check('unchanged_original_age',all(s['sim_s']<=s['command']['observation']['sim_s']+.3+1e-6 and s['tick_end_wall_s']<=s['command']['observation']['input_wall_s']+4 for s in snaps if s['command']['observation']))
    registered=[r['args']['frame'] for r in requests if r['kind']=='register']
    check('fixed_target_cadence',len(registered)>=7 and all(b-a==4 for a,b in zip(registered,registered[1:])))
    collect=[x for x in consumption if x['operation']['kind']=='collect']
    check('two_tick_capture',all(x['reply']['frozen']['frame']==x['reply']['observation']['frame']+2 for x in collect if x['reply']['success']))
    for t in tasks:
        ends=[x['worker_completed_wall_s'] for x in outputs if x['sequence']==t['sequence']]
        end=min(ends+[fault.get('wall_s',float('inf'))])
        check('task_no_physics_'+str(t['sequence']),not any(t['requested_wall_s']<=s['input_wall_s']<end for s in snaps))
    injection_time=None;injection_detail={}
    if name=='missing_camera':
        armed=load(root/'camera_injection_armed.json');rejection=load(root/'capture_rejection.json')
        rows=lines(root,'camera_metadata.jsonl');suppressed=[r for r in rows if r['disposition']=='TestSuppressed']
        key=armed.get('key',{});sel=[r for r in rows if r['frame']==key.get('frame') and r['disposition'] in ('Selected','PendingSnapshot')]
        check('injection_actual',len(suppressed)==1 and suppressed[0]['camera']=='CAM_FRONT' and suppressed[0]['actor_id']==key.get('actor_id') and suppressed[0]['frame']==key.get('frame') and [suppressed[0]['run'],suppressed[0]['episode'],suppressed[0]['sensor_generation']]==key.get('identity'))
        check('other_five_intact',len(sel)==5 and len({r['camera'] for r in sel})==5 and 'CAM_FRONT' not in {r['camera'] for r in sel})
        check('capture_rejection_path',rejection.get('reason')=='MissingCamera' and len(rejection.get('groups',[]))==5 and rejection.get('imu',{}).get('frame')==key.get('frame') and rejection.get('snapshot',{}).get('frame')==key.get('frame') and rejection.get('current',{}).get('frame')==key.get('frame',0)+2)
        check('return_margin',bool(rejection) and rejection['wall_s']<=rejection['operation']['deadline'] and fault.get('wall_s',float('inf'))<=rejection['original_start_wall_s']+.5)
        if suppressed:injection_time=suppressed[0]['receive_monotonic_s']
        injection_detail=dict(suppressed=len(suppressed),retained_cameras=len(sel),reserved_return_s=cfg['collect_return_reserve_s'])
    elif name=='invalid_output':
        mark=load(root/'worker_injection.json');bad=[x for x in outputs if x['sequence']==mark.get('sequence')]
        check('injection_actual',len(bad)==1 and bad[0]['control']['throttle']==1.1 and bad[0]['control']['brake']==0 and bad[0]['control']['steer']==0)
        check('illegal_never_rpc',not any(r['kind']=='control' and r['args']['command']['control']['throttle']==1.1 for r in requests))
        if bad:
            injection_time=bad[0]['worker_completed_wall_s'];t=next(t for t in tasks if t['sequence']==bad[0]['sequence'])
            check('legal_output_identity_and_time',bad[0]['identity']==t['identity'] and bad[0]['frame']==t['frame'] and bad[0]['source']==t['source'] and injection_time<t['deadline_wall_s'] and injection_time-bad[0]['worker_received_wall_s']>=.6)
        injection_detail=dict(throttle=1.1,scope='Only finite out-of-range throttle; no general model trajectory claim')
    else:
        mark=load(root/'manual_injection.json');entry=load(root/'EMERGENCY_STOP');t=tasks[-1] if tasks else {}
        injection_time=mark.get('request_wall_s')
        check('injection_actual',bool(mark) and mark==entry and mark.get('source')=='IndependentTestDriverViaManualEntry')
        check('early_live_manual',bool(mark) and mark['request_wall_s']<t.get('requested_wall_s',0)+.5 and mark['request_wall_s']<t.get('deadline_wall_s',0)-1 and mark['request_wall_s']-mark['owner_heartbeat']<.3 and mark['request_wall_s']-mark['worker_heartbeat']['wall_s']<.3)
        check('late_normal_rejected',result.get('late_result')==dict(rejected=True,reason='LateResultLocked') and any(x['sequence']==t.get('sequence') and not x['late'] and x['worker_completed_wall_s']>fault.get('wall_s',float('inf')) for x in outputs))
        injection_detail=dict(scope='Automated injection via manual emergency entry; not a human keypress',late_normal_rejected=result.get('late_result'))
    cl=load(root/'cleanup_steps.json')
    check('sensor_stop',len(cl.get('sensor_stops',[]))==8 and all(s.get('passed') for s in cl['sensor_stops']))
    check('actor_destroy',len(cl.get('actor_destruction',[]))==9 and all(s.get('destroyed') for s in cl['actor_destruction']))
    check('settings_and_remaining',cl.get('complete') and cl.get('settings_restored') and cl.get('remaining')==[] and cl.get('settings')==cl.get('settings_expected'))
    check('processes_ports',supervisor.get('cleanup_complete') and result.get('remaining_processes')==[] and not result.get('cleanup_error'))
    tick_reqs=lines(root,'tick_requests.jsonl');owner_attempts=sum(result.get('phase_ticks',{}).values());charged=result.get('charged_ticks',800)
    check('tick_accounting',len(tick_reqs)==len(snaps)==owner_attempts and charged==owner_attempts+60 and charged<=800)
    check('phase_caps',all(v<=cfg['tick_caps'][k] for k,v in result.get('phase_ticks',{}).items()))
    check('zero_model_pid',result.get('model_controls')==result.get('model_forwards')==result.get('training')==result.get('pid_executions')==0)
    before=ordinary[-1] if ordinary else None;stopped=stable[0] if stable else None
    import math
    summary=dict(scenario=name,status='Passed_FiniteInjectedFaultOnly' if all(checks.values()) else 'Partial',commit=request.get('commit'),run_id=root.name,
        checks=checks,failed_checks=[k for k,v in checks.items() if not v],injection=injection_detail,first_fault=fault.get('reason'),injection_forward_speed_mps=pre.get('speed'),injection_speed_norm_mps=pre.get('speed_norm'),
        injection_to_detection_ms=None if injection_time is None or not fault else (fault['wall_s']-injection_time)*1000,
        brake_server_processed=brake.get('server_processed',False),brake_processing_ms=(brake.get('return_wall_s',0)-brake.get('submission_wall_s',0))*1000,
        fault_feedback_ticks=len(after),first_fault_feedback_speed_mps=after[0]['speed_norm'] if after else None,terminal_speed_mps=after[-1]['speed_norm'] if after else None,
        stop_sim_s=None if not before or not stopped else stopped['sim_s']-before['sim_s'],stop_distance_m=None if not before or not stopped else math.dist(stopped['position'],before['position']),hold_sim_s=hold,observation_wall_s=observation_wall,
        normal_control_submissions_after_fault=len(normal_after),hold_physics_ticks=sum(s['command']['source']=='PlannedHoldBrake' for s in snaps),
        cleanup=dict(sensor_stops=sum(s.get('passed',False) for s in cl.get('sensor_stops',[])),actor_destructions=sum(s.get('destroyed',False) for s in cl.get('actor_destruction',[])),settings_restored=cl.get('settings_restored',False),remaining=cl.get('remaining'),processes_ports=checks['processes_ports']),
        budget=dict(starts=1,attempted_ticks=owner_attempts,observed_ticks=len(snaps),phase_ticks=result.get('phase_ticks',{}),settings_charges=result.get('settings_charges',{}),charged_ticks=charged,max_ticks=800,model_forwards=0,model_controls=0,pid_executions=0,training=0),
        limits=['Single finite injection','No model/PID/real-time/driving claim','Late RPC cleanup not intentionally tested','J13-A Partial'])
    return summary
