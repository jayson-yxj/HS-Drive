"""Fixed clocks/fake execution through real owner entry; no CARLA or models."""
import json
from pathlib import Path
from types import SimpleNamespace as S
import pytest
from simulator.m8j13_logic_runtime import OperationDeadline,operation_stamp,supervision_reason,STOP
from tools.m8j13_logic_runtime import Runtime
from tools.m8j13_logic_io import measured_tick,cleanup_owner

CFG=json.loads(Path('configs/m8j13_logic_init_recheck.json').read_text())


class Clock:
    def __init__(self):self.now=100.
    def monotonic(self):return self.now
    def sleep(self,d):self.now+=d


def configured(tmp_path,monkeypatch):
    import tools.m8j13_logic_runtime as runtime
    clock=Clock();monkeypatch.setattr(runtime.time,'monotonic',clock.monotonic);monkeypatch.setattr(runtime.time,'sleep',clock.sleep)
    for name in ('io_requests','io_responses'):(tmp_path/name).mkdir()
    r=Runtime(tmp_path,CFG);r.guard.active=dict(source='InitialProtectionBrake',control=STOP,observation=None)
    return r,clock


def fake_io_response(r,clock,monkeypatch,duration):
    import tools.m8j13_logic_runtime as runtime
    original=runtime.load;captured=[]
    def load(path):
        if path.parent.name=='io_responses':
            request_path=r.root/'io_requests'/path.name
            if request_path.exists():
                req=json.loads(request_path.read_text());op=req['operation']
                if not captured:captured.append(req)
                if clock.now>=op['started']+duration:
                    snap=dict(frame=1,sim_s=.05,pose=[[1]],velocity=[0,0,0],speed=0.,speed_norm=0.)
                    return dict(success=True,sequence=req['sequence'],rpc_start_wall_s=op['started']+.01,operation=op,snapshot=snap)
        return original(path)
    monkeypatch.setattr(runtime,'load',load);return captured


@pytest.mark.parametrize('phase,duration,passed',[('initialization',.8,True),('initialization',2.01,False),('stationary',.8,False)])
def test_real_owner_uses_initialization_or_activity_deadline(tmp_path,monkeypatch,phase,duration,passed):
    r,c=configured(tmp_path,monkeypatch);r.state['armed']=True
    if phase!='initialization':r.state['phase']='BoundedCapture';r.state['phase_deadline']=c.now+.5
    requests=fake_io_response(r,c,monkeypatch,duration)
    if passed:
        r.tick(phase);assert r.current['frame']==1
    else:
        with pytest.raises(TimeoutError):r.tick(phase)
        r.latch('TickDeadline')
        assert r.terminal and r.guard.locked and r.advance_unknown
        assert r.counts[phase]==1
    stamp=requests[0]['operation'];assert stamp['deadline']==pytest.approx(100.+(2. if phase=='initialization' else .5))
    assert requests[0]['owner_submitted_wall_s']==stamp['started']
    # Ordinary liveness cannot change the exact shared operation deadline.
    state=dict(armed=True,owner_wall_s=stamp['deadline']+.01,run_deadline=300,phase_deadline=300,phase='Initialization',operation=stamp)
    assert supervision_reason(state,stamp['deadline']+.01)=='TickRPCDeadline'


def test_phase_switch_immediately_restores_half_second_and_does_not_reset_stage(tmp_path,monkeypatch):
    r,c=configured(tmp_path,monkeypatch);seen=[]
    def call(kind,args,timeout,allow_fault):
        seen.append(timeout);return dict(rpc_start_wall_s=c.now,snapshot=dict(frame=len(seen),sim_s=.05*len(seen),speed_norm=0.,speed=0.))
    r.call=call;r.tick('initialization');deadline=r.state['phase_deadline']
    r.phase('BoundedCapture',c.now+.5);r.tick('stationary')
    assert seen==[2.,.5] and deadline==130.


def test_queue_drain_rpc_snapshot_feedback_share_one_remaining_budget(monkeypatch):
    c=Clock();op=operation_stamp(1,'tick',c.now,2.,130.,340.);budget=OperationDeadline(op,c.monotonic)
    c.sleep(.4)  # IPC queue and input drain already consume original time.
    client=S(timeouts=[],set_timeout=lambda value:client.timeouts.append(value))
    calls=[]
    def rpc(seconds):calls.append(seconds);c.sleep(.8);return 'tick'
    assert budget.call(rpc,client=client,timeout_argument=True)=='tick'
    assert calls==pytest.approx([1.6])
    assert budget.remaining()==pytest.approx(.8)
    c.sleep(.5)  # Snapshot and feedback processing.
    assert budget.remaining()==pytest.approx(.3)
    c.sleep(.31)
    with pytest.raises(TimeoutError):budget.call(lambda:pytest.fail('must not issue next call'))
    assert op['deadline']==102.


def test_rpc_return_after_deadline_is_failed_even_if_function_succeeds():
    c=Clock();b=OperationDeadline(operation_stamp(1,'tick',c.now,.5,110,120),c.monotonic)
    def late():c.sleep(.8);return 'success'
    with pytest.raises(TimeoutError):b.call(late)


def test_effective_deadline_is_earliest_phase_run_or_original_observation():
    assert operation_stamp(1,'tick',10,2,10.8,12)['deadline']==10.8
    assert operation_stamp(1,'tick',10,2,20,10.7)['deadline']==10.7
    assert operation_stamp(1,'tick',10,2,20,30,10.6)['deadline']==10.6
    with pytest.raises(TimeoutError):operation_stamp(1,'tick',10,2,9,20)


def test_pacing_before_submission_and_slow_completion_does_not_catch_up(tmp_path,monkeypatch):
    r,c=configured(tmp_path,monkeypatch);r.last_init_rpc_start=c.now
    r.wait_initialization_pacing();assert c.now>=100.05 and r.pending is None and not r.counts

    for _ in range(80):c.sleep(.01);r.pump()  # Actual owner polls while IO is slow.
    before=c.now;r.last_init_rpc_start=100.05
    r.wait_initialization_pacing();assert c.now==before
    # A subsequent fast request must wait a fresh50ms from its actual start.
    r.last_init_rpc_start=c.now;r.wait_initialization_pacing();assert c.now>=before+.05
    assert max(r.owner_gaps)<.3


@pytest.mark.parametrize('success',[True,False])
def test_late_reply_updates_local_queue_only_and_retains_uncertain_server(tmp_path,monkeypatch,success):
    from tools.m8j13_common import atomic
    r,c=configured(tmp_path,monkeypatch);r.pending=dict(sequence=7,kind='tick',deadline=c.now-.1)
    r.guard.latch();r.guard.uncertain=True;r.advance_unknown=True;r.terminal=dict(reason='FirstFault',wall_s=c.now-.1)
    atomic(tmp_path/'io_responses'/'000007.json',dict(success=success,call_end_wall_s=c.now))
    assert r.reconcile_for_cleanup()
    assert r.pending is None and r.terminal['reason']=='FirstFault' and r.guard.locked
    assert r.result['late_io_cleanup_reconciliation']['server_advance_unknown']
    with pytest.raises(ValueError):r.guard.before_tick(.1,c.now)


def test_cleanup_records_each_action_and_skips_restore_for_unknown_advance(tmp_path):
    c=Clock();log=[]
    class Sensor:
        id=1
        def __init__(self):self.listening=True
        def is_listening(self):return self.listening
        def stop(self):self.listening=False;log.append('stop')
        def destroy(self):log.append('destroy');return True
    sensor=Sensor();world=S(set_weather=lambda w:log.append('weather'),apply_settings=lambda *a,**k:log.append('settings'))
    owner=S(world=world,sensors=[sensor],actors=[sensor],client=S(set_timeout=lambda x:None),original=S(),weather=S())
    b=OperationDeadline(operation_stamp(1,'cleanup',c.now,15,115,115),c.monotonic)
    report=cleanup_owner(owner,b,tmp_path,advance_unknown=True)
    assert log==['stop','destroy'] and report['sensor_stops'][0]['passed']
    assert report['actor_destruction'][0]['destroyed'] and not report['settings_restored']
    assert report['advance_state_unknown'] and (tmp_path/'cleanup_steps.json').exists()


def test_cleanup_deadline_prevents_further_calls_and_retains_partial_steps(tmp_path):
    c=Clock();calls=[]
    class Actor:
        id=1
        def destroy(self):calls.append('destroy');c.sleep(16);return True
    world=S(set_weather=lambda w:pytest.fail('expired cleanup must not issue settings'))
    owner=S(world=world,sensors=[],actors=[Actor()],client=S(set_timeout=lambda x:None),original=S(),weather=S())
    b=OperationDeadline(operation_stamp(1,'cleanup',c.now,15,115,115),c.monotonic)
    r=cleanup_owner(owner,b,tmp_path)
    assert calls==['destroy'] and not r['complete'] and r['errors']


def test_initialization_and_active_config_boundaries_and_budget():
    old=json.loads(Path('configs/m8j13_logic_runtime_test.json').read_text())
    assert CFG['wall_deadlines_s']==old['wall_deadlines_s']
    assert CFG['initialization_tick_wall_s']==2 and CFG['initialization_min_tick_interval_s']==.05
    assert CFG['tick_caps']['settings_startup']==CFG['tick_caps']['settings_restore']==30
    assert sum(CFG['tick_caps'].values())==800
    assert CFG['private_root']!=old['private_root'] and CFG['model_forward_attempts']==0


def test_actual_tick_adapter_passes_remaining_and_records_all_endpoints(monkeypatch):
    import tools.m8j13_logic_io as io
    c=Clock();monkeypatch.setattr(io.time,'monotonic',c.monotonic)
    monkeypatch.setattr(io,'measurement',lambda snap,ego:dict(frame=snap.frame,sim_s=.05,pose=[],velocity=[],speed_norm=0.))
    op=operation_stamp(1,'tick',100.,2.,130.,340.);b=OperationDeadline(op,c.monotonic);c.sleep(.2)
    seconds=[]
    def tick(seconds):c.sleep(.8);return 11
    def measured_rpc(**kw):seconds.append(kw['seconds']);return tick(**kw)
    def snap():c.sleep(.05);return S(frame=11)
    full=S(steer=0.,throttle=0.,brake=1.,hand_brake=False,reverse=False,manual_gear_shift=False,gear=0)
    ego=S(get_control=lambda:(c.sleep(.02) or full))
    owner=S(client=S(set_timeout=lambda x:None),world=S(tick=measured_rpc,get_snapshot=snap))
    reply={};snapshot,row=measured_tick(owner,ego,b,reply,10)
    assert seconds==pytest.approx([1.8]) and snapshot.frame==11
    assert reply['rpc_start_wall_s']==pytest.approx(100.2)
    assert reply['rpc_return_wall_s']==pytest.approx(101.)
    assert reply['snapshot_available_wall_s']==pytest.approx(101.05)
    assert reply['feedback_complete_wall_s']==pytest.approx(101.07)
    assert row['full_control']['brake']==1


def test_cleanup_success_uses_same_deadline_and_records_settings_readback(tmp_path):
    c=Clock();calls=[]
    class Sensor:
        id=1
        def is_listening(self):return bool(not calls)
        def stop(self):calls.append('stop')
        def destroy(self):calls.append('destroy');return True
    sensor=Sensor();settings=S(synchronous_mode=False,fixed_delta_seconds=None,no_rendering_mode=False,substepping=True,max_substep_delta_time=.01,max_substeps=10)
    def apply(value,seconds):calls.append(('restore',seconds));c.sleep(.2)
    world=S(set_weather=lambda w:calls.append('weather'),apply_settings=apply,get_snapshot=lambda:S(frame=42,find=lambda actor:None),get_settings=lambda:settings,wait_for_tick=lambda seconds:S(frame=43,find=lambda actor:None))
    owner=S(world=world,sensors=[sensor],actors=[sensor],client=S(set_timeout=lambda x:None),original=settings,weather=S())
    b=OperationDeadline(operation_stamp(1,'cleanup',100.,15.,115.,115.),c.monotonic)
    report=cleanup_owner(owner,b,tmp_path)
    assert report['complete'] and report['settings_restored'] and report['remaining']==[]
    assert calls[:3]==['stop','destroy','weather'] and calls[3]==('restore',15.)


def test_owner_timeout_partial_report_and_cleanup_budget_not_lost(tmp_path,monkeypatch):
    r,c=configured(tmp_path,monkeypatch);r.start=lambda *a,**kw:None;r.ready=lambda *a,**kw:None
    def fake_call(kind,*args,**kw):
        if kind=='setup':
            r.settings_charges['settings_startup']=30
            raise TimeoutError('SetupDeadline')
        if kind=='cleanup':
            r.settings_charges['settings_restore']=30
            return dict(cleanup=dict(complete=False,settings_restored=False,errors=['UnknownPriorAdvance']))
        raise AssertionError(kind)
    r.call=fake_call;r.latch=lambda reason:setattr(r,'terminal',dict(reason=reason,wall_s=c.now))
    r.run();result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']=='Partial' and result['charged_ticks']==60
    assert result['settings_charges']=={'settings_startup':30,'settings_restore':30}
    assert not result['cleanup']['complete'] and result['model_controls']==0
