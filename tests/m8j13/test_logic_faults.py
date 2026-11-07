"""Actual receive/collect/worker/owner/manual-entry wiring, no server or model."""
import copy
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as S
import pytest
from simulator.m8j13_logic_faults import CameraSuppression,receive_camera,collect_input,synthetic_reply,scenario,validate_manual_request,reserve_scenario
from simulator.m8j13_logic_runtime import TargetCapture,TaskChannel,STOP,OperationDeadline,supervision_reason
from tools.m8j13_logic_services import test_worker as run_worker,manual_driver,watchdog
from tools.m8j13_logic_runtime import Runtime
from tools.m8j13_common import atomic
from tests.m8j13.test_logic_runtime import image,snapshot,IDENTITY,IDS,CAMERAS,response


def cfg(name='missing_camera'):
    return json.loads(Path('configs/m8j13_logic_fault_'+name+'.json').read_text())

def wait(path,seconds=3):
    end=time.monotonic()+seconds
    while not path.exists():
        assert time.monotonic()<end,path
        time.sleep(.002)
    return json.loads(path.read_text())

def camera_chain():
    t=TargetCapture(IDENTITY,IDS);t.register(10);t.start_tick(10,1.);t.confirm(snapshot())
    i=S(groups=defaultdict(list),payloads={},imu={10:dict(frame=10)})
    h=CameraSuppression(cfg(),IDENTITY,IDS);key=dict(identity=list(IDENTITY),camera='CAM_FRONT',actor_id=IDS['CAM_FRONT'],frame=10);h.arm(key)
    return t,i,h,key

def payload(name):
    return S(stamp=S(record=lambda:image(name)),width=704,height=396,bgra=bytes(704*396*4),copied_monotonic_s=1.1)


def test_actual_receive_suppresses_one_raw_payload_before_target_and_collect_rejects_early():
    t,i,h,key=camera_chain();records=[]
    for name in CAMERAS:receive_camera(payload(name),t,i,h,records.append)
    assert len(t.groups)==len(i.groups[10])==len(i.payloads)==5
    assert h.hit['camera']=='CAM_FRONT' and len([r for r in records if r['disposition']=='TestSuppressed'])==1
    now=[1.15];budget=OperationDeadline(dict(kind='collect',deadline=1.5),lambda:now[0])
    with pytest.raises(ValueError,match='^MissingCamera$'):
        collect_input(t,i,lambda:snapshot(12),lambda:None,budget,lambda:now[0],lambda dt:now.__setitem__(0,now[0]+dt),.1)
    assert 1.4<=now[0]<1.41  # Diagnostic available before outer deadline, no third tick.
    assert t.accepted is False
    state=dict(armed=True,owner_wall_s=now[0],phase='BoundedCapture',phase_deadline=1.5,run_deadline=9.,fault='MissingCamera',operation=None)
    assert supervision_reason(state,now[0]+.02)=='MissingCamera'
    with pytest.raises(ValueError,match='DuplicateInjected'):receive_camera(payload('CAM_FRONT'),t,i,h,records.append)


@pytest.mark.parametrize('field',['identity','actor_id','camera'])
def test_injection_key_mismatch_rejected(field):
    t,i,h,key=camera_chain();h=CameraSuppression(cfg(),IDENTITY,IDS);key[field]=['other',42,'old'] if field=='identity' else (-1 if field=='actor_id' else 'CAM_BACK')
    with pytest.raises(ValueError,match='Identity'):h.arm(key)


def test_disabled_unknown_and_duplicate_injection():
    assert scenario({}) is None
    with pytest.raises(ValueError):scenario({'fault_injection':dict(enabled=True,scenario='unknown')})
    h=CameraSuppression({},IDENTITY,IDS)
    with pytest.raises(ValueError):h.arm(dict(identity=IDENTITY,camera='CAM_FRONT',actor_id=100,frame=10))
    t,i,h,key=camera_chain()
    with pytest.raises(ValueError):h.arm(key)


@pytest.mark.parametrize('control,reason',[(dict(STOP,throttle=1.1,brake=0),'ControlOutOfRange'),(dict(STOP,steer=float('nan')),'NonfiniteControl'),({'steer':0},'InvalidControl'),(dict(STOP,throttle=.1),'ConflictingControl')])
def test_public_sanitizer_precedes_equality_in_real_task_channel(control,reason):
    ch=TaskChannel(IDENTITY);t=ch.request(snapshot(),snapshot(12),1.2,'normal',STOP);r=response(t);r['control']=control
    with pytest.raises(ValueError,match='^'+reason+'$'):ch.consume(r,snapshot(12),1.9)
    assert not ch.consumed


def test_actual_worker_returns_illegal_value_without_calling_control(tmp_path):
    for n in ('tasks','task_results'):(tmp_path/n).mkdir()
    c=cfg('invalid_output');errors=[]
    def worker():
        try:run_worker(tmp_path,c)
        except BaseException as e:errors.append(e)
    thread=threading.Thread(target=worker);thread.start()
    try:
        wait(tmp_path/'test_worker_ready.json');now=time.monotonic();ch=TaskChannel(IDENTITY)
        t=ch.request(dict(snapshot(),input_wall_s=now),snapshot(12),now,'normal',dict(STOP,throttle=.5,brake=0))
        t['injection']='invalid_output';atomic(tmp_path/'owner_state.json',dict(identity=list(IDENTITY),task=dict(sequence=1)));atomic(tmp_path/'tasks'/'000001.json',t)
        r=wait(tmp_path/'task_results'/'000001.json')
        assert r['worker_completed_wall_s']-r['worker_received_wall_s']>=.6 and r['control']['throttle']==1.1
        with pytest.raises(ValueError,match='ControlOutOfRange'):ch.consume(r,snapshot(12),time.monotonic())
        assert wait(tmp_path/'worker_injection.json')['sequence']==1
    finally:
        (tmp_path/'test_worker_stop').touch();thread.join(3)
    assert not thread.is_alive() and not errors


def test_real_owner_receives_collect_failure_before_deadline_then_independent_reason(tmp_path):
    for n in ('io_requests','io_responses'):(tmp_path/n).mkdir()
    r=Runtime(tmp_path,cfg());now=time.monotonic();r.phase('BoundedCapture',now+.5);r.state['armed']=True
    def fake_io():
        path=tmp_path/'io_requests'/'000001.json';q=wait(path);time.sleep(.08)
        atomic(tmp_path/'io_responses'/path.name,dict(sequence=q['sequence'],operation=q['operation'],success=False,error="ValueError('MissingCamera')",rejection_reason='MissingCamera'))
    thread=threading.Thread(target=fake_io);thread.start()
    with pytest.raises(RuntimeError,match='^MissingCamera$'):r.call('collect',timeout=.5)
    thread.join();assert r.pending is None and not r.guard.uncertain
    r.state['fault']='MissingCamera';r.pump();assert supervision_reason(r.state,time.monotonic())=='MissingCamera'


def test_collect_late_response_preserves_first_timeout_never_claims_missing(tmp_path):
    for n in ('io_requests','io_responses'):(tmp_path/n).mkdir()
    r=Runtime(tmp_path,cfg());r.phase('BoundedCapture',time.monotonic()+.03)
    with pytest.raises(TimeoutError):r.call('collect',timeout=.5)
    assert r.pending is not None
    r.terminal=dict(reason='PhaseDeadline:BoundedCapture',wall_s=time.monotonic());r.guard.latch()
    op=r.pending;atomic(tmp_path/'io_responses'/'000001.json',dict(sequence=1,operation=op,success=False,error='MissingCamera'))
    assert r.reconcile_for_cleanup() and r.terminal['reason']=='PhaseDeadline:BoundedCapture'
    assert not r.result['late_io_cleanup_reconciliation']['physics_resumption_allowed']


def test_manual_driver_and_real_watchdog_entry_no_owner_latch(monkeypatch,tmp_path):
    import sys
    from tools import m8j13_logic_services as services
    calls=[];errors=[]
    client=S(set_timeout=lambda x:None,get_client_version=lambda:'0.9.16',get_server_version=lambda:'0.9.16')
    monkeypatch.setitem(sys.modules,'carla',S(Client=lambda *a:client,libcarla=S(__file__='synthetic-client')))
    monkeypatch.setattr(services,'batch_control',lambda client,ego,control,budget:(calls.append(dict(control)) or dict(server_processed=True,physics_confirmed=False,due_tick_cue=False)))
    now=time.monotonic();task=dict(sequence=1,frame=10,requested_wall_s=now,deadline_wall_s=now+2,mode='normal')
    state=dict(identity=list(IDENTITY),armed=True,phase='ExpectedComputeWait',task=task,operation=None,fault=None,epoch=0,ego_id=123,owner_wall_s=now,phase_deadline=now+2,run_deadline=now+240)
    atomic(tmp_path/'owner_state.json',state);atomic(tmp_path/'test_worker_heartbeat.json',dict(task=1,wall_s=now))
    atomic(tmp_path/'manual_request.json',dict(identity=IDENTITY,task=task,forward_speed_mps=1.1))
    def run(fn):
        try:fn(tmp_path,cfg('manual_emergency_stop'))
        except BaseException as e:errors.append(e)
    threads=[threading.Thread(target=run,args=(fn,)) for fn in (watchdog,manual_driver)]
    for t in threads:t.start()
    try:
        mark=wait(tmp_path/'manual_injection.json');fault=wait(tmp_path/'watchdog_fault.json');brake=wait(tmp_path/'fault_brake.json')
        assert fault['reason']=='ManualEmergencyStop' and brake['server_processed'] and calls==[STOP]
        assert mark['request_wall_s']<task['deadline_wall_s']-1
        # Even after another fault signal, first watchdog fault remains immutable.
        state['fault']='TaskDeadline';atomic(tmp_path/'owner_state.json',state);time.sleep(.03)
        assert wait(tmp_path/'watchdog_fault.json')==fault
    finally:
        (tmp_path/'manual_stop').touch();(tmp_path/'logic_watchdog_stop').touch()
        for t in threads:t.join(3)
    assert not errors and all(not t.is_alive() for t in threads)


@pytest.mark.parametrize('kind',['pending','late','heartbeat','identity'])
def test_manual_request_preconditions(kind):
    t=dict(sequence=1,requested_wall_s=1.,deadline_wall_s=3.);request=dict(task=t,forward_speed_mps=1.1)
    state=dict(task=t,phase='ExpectedComputeWait',operation=None,owner_wall_s=1.1);hb=dict(task=1,wall_s=1.1);now=1.15
    if kind=='pending':state['operation']={'kind':'tick'}
    if kind=='late':now=1.7
    if kind=='heartbeat':hb['wall_s']=0
    if kind=='identity':state['task']=dict(t,sequence=2)
    with pytest.raises(ValueError):validate_manual_request(request,state,hb,now)


def test_total_ledger_sequence_failure_and_no_reset():
    r=reserve_scenario([],'missing_camera','r1','sha');records=[r]
    with pytest.raises(ValueError):reserve_scenario(records,'invalid_output','r2','sha')
    records.append(dict(event='finished',passed=False))
    with pytest.raises(ValueError):reserve_scenario(records,'invalid_output','r2','sha')
    records[-1]['passed']=True
    records.append(reserve_scenario(records,'invalid_output','r2','sha'));records.append(dict(event='finished',passed=True))
    records.append(reserve_scenario(records,'manual_emergency_stop','r3','sha'));records.append(dict(event='finished',passed=True))
    assert sum(r.get('reserved_ticks',0) for r in records)==2400
    with pytest.raises(ValueError):reserve_scenario(records,'missing_camera','r4','sha')


def test_stop_observation_refuses_inflight_and_uncertain_even_with_brake_ack(tmp_path):
    r=Runtime(tmp_path,cfg());now=time.monotonic();r.terminal=dict(reason='MissingCamera',wall_s=now);r.guard.latch();r.pending={'kind':'tick','sequence':1}
    r.pump=lambda:None
    atomic(tmp_path/'fault_brake.json',dict(server_processed=True,handoff_uncertain=False))
    r.stop_observe();assert r.result['stop']['status']=='HandoffUncertain' and not r.counts


def test_suite_entry_is_exclusive_and_model_free():
    import ast
    source=Path('tools/m8j13_logic_fault_suite.py').read_text()
    assert "open('x',buffering=1)" in source and 'if not passed:break' in source
    for p in ('tools/m8j13_logic_fault_suite.py','simulator/m8j13_logic_faults.py'):
        t=ast.parse(Path(p).read_text())
        assert not any(isinstance(n,ast.Attribute) and n.attr in ('world.tick','control_pid','set_target_velocity','enable_constant_velocity') for n in ast.walk(t))


def test_real_compute_illegal_worker_result_cannot_reach_command_submission(tmp_path):
    for n in ('tasks','task_results'):(tmp_path/n).mkdir()
    r=Runtime(tmp_path,cfg('invalid_output'));r.channel=TaskChannel(IDENTITY);now=time.monotonic()
    r.current=snapshot(12);r.observation=dict(snapshot(),input_wall_s=now);r.state['identity']=list(IDENTITY)
    commands=[];r.issue=lambda source,control,*a:commands.append((source,dict(control)))
    r.call=lambda kind,*a,**kw:dict(snapshot=r.current)
    errors=[]
    def worker():
        try:run_worker(tmp_path,cfg('invalid_output'))
        except BaseException as e:errors.append(e)
    thread=threading.Thread(target=worker);thread.start()
    try:
        with pytest.raises(ValueError,match='^ControlOutOfRange$'):r.compute(dict(STOP,throttle=.5,brake=0),inject=True)
        assert commands==[('PlannedHoldBrake',STOP)] and not r.channel.consumed
    finally:(tmp_path/'test_worker_stop').touch();thread.join(3)
    assert not errors


def test_budget_and_stage_thresholds_remain_frozen():
    base=json.loads(Path('configs/m8j13_logic_init_recheck.json').read_text())
    for name in ('missing_camera','invalid_output','manual_emergency_stop'):
        c=cfg(name)
        for key in ('wall_deadlines_s','tick_caps','initialization_tick_wall_s','initialization_min_tick_interval_s','physics_dt_s','scene','test_control','capture_flush_ticks','speed_guard_mps','injection_speed_mps','speed_limit_mps'):
            assert c[key]==base[key]
        assert sum(c['tick_caps'].values())==800 and c['collect_return_reserve_s']==.1


def test_aggregate_audit_never_uses_runtime_passed_alone(tmp_path):
    from tools.m8j13_logic_fault_report import audit
    atomic(tmp_path/'frozen_config.json',cfg('invalid_output'))
    atomic(tmp_path/'result.json',dict(status='Passed_FiniteInjectedFaultOnly',model_forwards=0,model_controls=0,pid_executions=0,training=0))
    report=audit(tmp_path)
    assert report['status']=='Partial' and not report['checks']['injection_actual'] and not report['checks']['physical_hold']
    assert not report['checks']['sensor_stop'] and not report['checks']['brake_processed']
