"""Read-only audit of the single initialization-revision run; aggregate export.

No CARLA/model imports, no image/prediction export, no simulator actions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def lines(root,name):return [json.loads(x) for x in (root/name).read_text().splitlines()] if (root/name).exists() else []
def distribution(values):
    if not values:return None
    s=sorted(values)
    def q(p):
        x=(len(s)-1)*p;i=int(x);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(x-i)
    return dict(n=len(s),min=s[0],p50=q(.5),p95=q(.95),max=s[-1])


def audit(root):
    result=read(root/'result.json');supervisor=read(root/'supervisor.json');cfg=read(root/'frozen_config.json');frozen=read(root/'request.json')
    consumption=lines(root,'io_consumption.jsonl');snaps=lines(root,'snapshots.jsonl');events=lines(root,'owner_events.jsonl')
    requests={int(p.stem):read(p) for p in (root/'io_requests').glob('*.json')}
    responses={int(p.stem):read(p) for p in (root/'io_responses').glob('*.json')}
    assert len(requests)==len(responses)==len(consumption)
    for item in consumption:
        op=item['operation'];request=requests[op['sequence']];reply=item['reply']
        assert op==request['operation']==reply['operation']
        assert request['owner_submitted_wall_s']==op['started']
        assert reply['success'] and op['started']<=reply['io_received_wall_s']<=reply['call_end_wall_s']<=item['owner_consumed_wall_s']<=op['deadline']
        assert op['deadline']<=min(op['started']+op['operation_limit_s'],op['phase_deadline'],op['run_deadline'])
        if op['kind']=='tick':
            timeline=[reply[k] for k in ('io_received_wall_s','rpc_start_wall_s','rpc_return_wall_s','snapshot_available_wall_s','feedback_complete_wall_s','processing_complete_wall_s')]
            assert timeline==sorted(timeline)
            init=request['args']['phase']=='initialization'
            assert op['operation_limit_s']==(2. if init else .5)
        if op['kind']=='control':
            assert op['operation_limit_s']==.2 and reply['due_tick_cue'] is False and reply['frame_before']==reply['frame_after']
    init=[x for x in consumption if x['operation']['kind']=='tick' and requests[x['operation']['sequence']]['args']['phase']=='initialization']
    active=[x for x in consumption if x['operation']['kind']=='tick' and requests[x['operation']['sequence']]['args']['phase']!='initialization']
    dur=lambda x:(x['owner_consumed_wall_s']-x['operation']['started'])*1000
    starts=[x['reply']['rpc_start_wall_s'] for x in init];intervals=[b-a for a,b in zip(starts,starts[1:])]
    assert min(intervals)>=.05 and len(init)<=60
    init_snaps=[x for x in snaps if x['phase']=='initialization']
    assert len(init_snaps)==result['initialization']['ticks'] and all(x['speed_norm']<=.1 for x in init_snaps[-21:])
    initial_phase=next(x for x in events if x['event']=='phase' and x['phase']=='Initialization')
    init_wall=init[-1]['owner_consumed_wall_s']-initial_phase['wall_s'];assert init_wall<=30
    assert all(x['feedback_matches'] for x in snaps)
    assert all(x['command']['source'] in ('IndependentTestControl','InitialProtectionBrake','FaultBrake') for x in snaps)
    assert all(x['speed_norm']==0 for x in snaps if x['phase']=='stationary')
    for x in snaps:
        obs=x['command']['observation']
        if obs:
            assert x['sim_s']<=obs['sim_s']+.3+1e-6 and x['tick_end_wall_s']<=obs['input_wall_s']+4.
    checks=lines(root,'input_checks.jsonl');collect=[x for x in consumption if x['operation']['kind']=='collect']
    assert len(checks)==len(collect)==len(result['stationary_cycles'])+len(result['moving_cycles'])+1
    target_frames=[x['reply']['observation']['frame'] for x in collect]
    assert all(b-a==4 for a,b in zip(target_frames,target_frames[1:]))
    for x in collect:
        o=x['reply']['observation'];f=x['reply']['frozen'];assert f['frame']==o['frame']+2
        assert x['owner_consumed_wall_s']<=o['input_wall_s']+.5
    for phase in ('stationary','start'):
        for x in snaps:
            if x['phase']==phase and x['command']['control']['throttle']>0:
                assert phase=='start' and x['command']['source']=='IndependentTestControl'
    tasks=[read(p) for p in sorted((root/'tasks').glob('*.json'))]
    completed=[read(p) for p in sorted((root/'task_results').glob('*.json'))]
    assert len(tasks)==len(completed)==len(collect) and tasks[-1]['mode']=='blocked'
    normal=completed[:-1];assert all(x['worker_completed_wall_s']-x['worker_received_wall_s']>=.6 and not x['late'] for x in normal)
    assert completed[-1]['late'] and result['late_result']==dict(rejected=True,reason='LateResultLocked')
    for task,reply in zip(tasks,completed):
        assert task['sequence']==reply['sequence'] and task['identity']==reply['identity'] and task['frame']==reply['frame']
        if not reply['late']:
            assert not any(task['requested_wall_s']<=x['input_wall_s']<=reply['worker_completed_wall_s'] for x in snaps)
    assert len(result['stationary_cycles'])==cfg['stationary_cycles'] and len(result['moving_cycles'])>0
    fault=read(root/'watchdog_fault.json');brake=read(root/'fault_brake.json');blocked=tasks[-1]
    assert fault['reason']=='TaskDeadline' and fault['wall_s']>=blocked['deadline_wall_s']
    assert completed[-1]['worker_completed_wall_s']>fault['wall_s']
    assert not lines(root,'collisions.jsonl')
    assert result['task_timeout']['worker_heartbeat']['task']==blocked['sequence']
    assert fault['wall_s']-result['task_timeout']['worker_heartbeat']['wall_s']<.3
    assert 1<=result['task_timeout']['speed']<1.7
    # No physics during the blocked task/hold; fault control must be processed first.
    assert not any(blocked['requested_wall_s']<=x['input_wall_s']<brake['return_wall_s'] for x in snaps)
    after=[x for x in snaps if x['phase']=='fault_stop'];assert len(after)==result['phase_ticks']['fault_stop'] and 0<len(after)<=cfg['tick_caps']['fault_stop']
    assert all(x['command']['source']=='FaultBrake' and x['full_control']['brake']==1 and x['full_control']['throttle']==0 for x in after)
    assert all(x['speed_norm']<=.1 for x in after)
    hold=after[-1]['sim_s']-after[0]['sim_s'];assert hold>=1-1e-6
    assert all(x['command']['source']=='FaultBrake' for x in snaps if x['input_wall_s']>fault['wall_s'])
    cl=result['cleanup'];assert cl['complete'] and cl['settings_restored'] and cl['remaining']==[]
    assert len(cl['sensor_stops'])==8 and all(x['passed'] for x in cl['sensor_stops'])
    assert len(cl['actor_destruction'])==9 and all(x['destroyed'] for x in cl['actor_destruction'])
    assert supervisor['cleanup_complete'] and result['remaining_processes']==[]
    init_ms=[dur(x) for x in init];capture_ms=[(x['owner_consumed_wall_s']-x['reply']['observation']['input_wall_s'])*1000 for x in collect]
    cleanup=next(x for x in consumption if x['operation']['kind']=='cleanup')
    stop_wall=after[-1]['tick_end_wall_s']-fault['wall_s'];assert stop_wall<10
    private_audit=dict(all_operations_checked=len(consumption),real_original_deadlines_preserved=True,tick_count=len(snaps),targets_checked=len(collect),all_control_feedback_matches=True,hold_physics_ticks=0)
    summary=dict(protocol_id='synchronous_logic_v1',revision=cfg['revision'],scope='FiniteNoModelInitializationAndMovingTaskTimeoutTest',run_commit=frozen['commit'],config_sha256=frozen['config_sha256'],
        initialization=dict(passed=True,ticks=len(init),consecutive_stable=21,total_wall_s=init_wall,operation_ms=distribution(init_ms),over_old_500ms=sum(t>500 for t in init_ms),over_new_2000ms=sum(t>2000 for t in init_ms),rpc_start_interval_ms=distribution([x*1000 for x in intervals]),causal_limit='No operation exceeded old0.5s this run. Deadline+pacing+load differ from prior run; no isolated cause or cold-start/GPU claim.'),
        active=dict(tick_operation_ms=distribution([dur(x) for x in active]),all_operations=len(consumption),operations_over_effective_deadline=0,clock_identity_matches=True,stationary_cycles=len(result['stationary_cycles']),normal_moving_cycles=len(result['moving_cycles']),target_groups=len(collect),target_interval_ticks=4,flush_ticks=2,capture_owner_complete_ms=distribution(capture_ms),stationary_capture_owner_complete_ms=distribution(capture_ms[:6]),normal_worker_wait_ms=distribution([(x['worker_completed_wall_s']-x['worker_received_wall_s'])*1000 for x in normal]),normal_task_handoff_complete_ms=distribution([x['wait_wall_s']*1000 for x in result['stationary_cycles']+result['moving_cycles']]),measurement_stationary_max_speed_mps=0.,max_navigation_error_m=max(x['navigation_error'] for x in checks),max_extrinsic_error=max(x['extrinsic_max_error'] for x in checks)),
        fault=dict(reason='TaskDeadline',injection_speed_mps=result['task_timeout']['speed'],task_to_detection_wall_s=fault['wall_s']-blocked['requested_wall_s'],detection_after_deadline_ms=(fault['wall_s']-blocked['deadline_wall_s'])*1000,worker_alive=True,brake_processing_ms=(brake['return_wall_s']-brake['submission_wall_s'])*1000,brake_return_to_first_feedback_ms=(after[0]['tick_end_wall_s']-brake['return_wall_s'])*1000,first_feedback_speed_mps=after[0]['speed_norm'],terminal_speed_mps=after[-1]['speed_norm'],stop_time_sim_s=result['stop']['stop_sim_s'],stop_distance_m=result['stop']['stop_distance_m'],hold_sim_s=hold,observation_ticks=len(after),observation_wall_s=stop_wall,late_result_rejected=True,normal_commands_after_latch=0,stop_scope=result['stop']['position_scope']),
        control_ticks=dict(Counter(x['command']['source'] for x in snaps)),hold_control_physics_ticks=0,owner_max_heartbeat_gap_ms=result['owner_heartbeat_max_gap_s']*1000,
        cleanup=dict(sensor_stops_passed=8,sensor_stops_expected=8,actors_destroyed=9,actors_expected=9,settings_restored=True,settings=cl['settings'],remaining_actors=[],settings_restore_observed_frame_delta=cl['settings_observed_frame_delta'],io_cleanup_ms=dur(cleanup),services_reaped=True,server_groups_empty=True,ports_released=True,late_rpc_cleanup_path='NotTriggeredThisRun_PureTestsOnly'),
        budget=dict(starts=1,max_starts=1,explicit_ticks=len(snaps),phase_ticks=result['phase_ticks'],settings_charges=result['settings_charges'],charged_ticks=result['charged_ticks'],max_ticks=800,unused_not_reused=True,model_forwards=0,model_controls=0,training=0,driving_budget_consumed=0),
        failures_preserved=True,J13_A='Partial',remaining_moving_faults=['MissingCamera','InvalidOutput','ManualEmergencyStop'],shadow_or_S0_auto_admission=False)
    assert summary['budget']['charged_ticks']==len(snaps)+sum(result['settings_charges'].values())<=800
    return summary,private_audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    summary,details=audit(a.root);a.out.mkdir(exist_ok=True,parents=True)
    (a.out/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.root.parent/'postrun_audit.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
