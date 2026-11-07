"""No CARLA server or model. Fake executor + actual runtime/service boundaries."""
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as S
import pytest
from simulator.m8j13_logic_runtime import TargetCapture,TaskChannel,CommandGuard,STOP,canonical,frozen_equal,supervision_reason,reserve_start
from tools.m8j13_logic_services import batch_control
from tools.m8j13_logic_runtime import Runtime

CAMERAS=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
IDS={n:100+i for i,n in enumerate(CAMERAS)}
IDENTITY=('synthetic',42,'generation_1')
CFG=json.loads(Path('configs/m8j13_logic_runtime_test.json').read_text())


def snapshot(frame=10):
    return dict(frame=frame,sim_s=frame*.05,pose=((1.,0.),(0.,1.)),velocity=(0.,0.,0.),speed_norm=0.,input_wall_s=1.)


def image(name,frame=10):
    return dict(run=IDENTITY[0],episode=42,sensor_generation=IDENTITY[2],camera=name,actor_id=IDS[name],frame=frame,timestamp_s=frame*.05,transform=((1.,0.),(0.,1.)))


def target():
    t=TargetCapture(IDENTITY,IDS);t.register(10);t.start_tick(10,1.)
    return t


def test_callback_before_snapshot_and_tuple_json_equivalence():
    t=target()
    for name in CAMERAS:assert t.camera(image(name))=='PendingSnapshot'
    with pytest.raises(ValueError,match='SnapshotUnconfirmed'):t.validate(snapshot(12),1.1)
    t.confirm(snapshot())
    assert t.validate(snapshot(12),1.1)['valid']
    j=json.loads(json.dumps(snapshot()))
    assert frozen_equal(snapshot(),j)
    j['pose'][0][0]+=1e-12
    assert not frozen_equal(snapshot(),j)


@pytest.mark.parametrize('case',['missing','duplicate','generation','actor','late','pending','flush','timestamp'])
def test_target_failures_not_hidden(case):
    t=target()
    if case=='generation':
        r=image(CAMERAS[0]);r['sensor_generation']='old'
        with pytest.raises(ValueError,match='OldGeneration'):t.camera(r)
        return
    if case=='actor':
        r=image(CAMERAS[0]);r['actor_id']+=1
        with pytest.raises(ValueError,match='WrongSensorInstance'):t.camera(r)
        return
    for name in CAMERAS:
        if case=='missing' and name==CAMERAS[0]:continue
        r=image(name)
        if case=='timestamp' and name==CAMERAS[0]:r['timestamp_s']+=.05
        t.camera(r)
    if case=='duplicate':
        with pytest.raises(ValueError,match='DuplicateCamera'):t.camera(image(CAMERAS[0]))
        return
    t.confirm(snapshot())
    with pytest.raises(ValueError):t.validate(snapshot(13 if case=='flush' else 12),1.51 if case=='late' else 1.1,pending=case=='pending')


def test_non_target_dropped_explicitly_selected_keys_never_overwritten():
    t=target();assert t.camera(image(CAMERAS[0],9))=='NotSelected'
    for name in CAMERAS:t.camera(image(name))
    t.confirm(snapshot());assert t.validate(snapshot(12),1.2)['valid']
    t.register(14)
    with pytest.raises(ValueError,match='LateSelectedObservation'):t.camera(image(CAMERAS[0]))
    t.close();assert t.camera(image(CAMERAS[0],14))=='AfterClose'
    assert not t.groups


def request(channel):return channel.request(snapshot(),snapshot(12),1.2,'normal',STOP)
def response(t):return {k:t[k] for k in ('sequence','identity','frame','source','control')}


def test_task_timeout_not_renewed_by_worker_or_owner_heartbeat():
    channel=TaskChannel(IDENTITY);t=request(channel)
    state=dict(armed=True,owner_wall_s=3.21,run_deadline=240.,phase='ExpectedComputeWait',phase_deadline=t['deadline_wall_s'],task=t,worker_heartbeat=3.21)
    assert supervision_reason(state,3.21)=='TaskDeadline'
    with pytest.raises(ValueError,match='TaskDeadline'):channel.consume(response(t),snapshot(12),3.21)
    channel.locked=True
    with pytest.raises(ValueError,match='LateResultLocked'):channel.consume(response(t),snapshot(12),3.22)


def test_slow_result_frozen_identity_and_single_consumption():
    channel=TaskChannel(IDENTITY);t=request(channel)
    assert channel.consume(response(t),json.loads(json.dumps(snapshot(12))),1.9)==STOP
    with pytest.raises(ValueError,match='UnexpectedOrRepeated'):channel.consume(response(t),snapshot(12),1.91)


@pytest.mark.parametrize('case',['pending','advanced','source','identity'])
def test_current_frame_and_source_are_required(case):
    ch=TaskChannel(IDENTITY);t=request(ch);r=response(t)
    if case=='source':r['source']='ModelHeldCommand'
    if case=='identity':r['identity']=['old',42,'old']
    with pytest.raises(ValueError):ch.consume(r,snapshot(13 if case=='advanced' else 12),1.9,pending=case=='pending')


def test_owner_heartbeat_and_rpc_are_independent_deadlines():
    s=dict(armed=True,owner_wall_s=1.,run_deadline=240.,phase='BoundedCapture',phase_deadline=2.,operation=dict(kind='tick',deadline=1.1))
    assert supervision_reason(s,1.11)=='TickRPCDeadline'
    assert supervision_reason(s,1.31)=='OwnerUnresponsive'
    s['owner_wall_s']=3.
    assert supervision_reason(s,3.)=='PhaseDeadline:BoundedCapture'


def test_command_processing_does_not_mean_physics_executed(monkeypatch):
    calls=[]
    monkeypatch.setitem(sys.modules,'carla',S(command=S(ApplyVehicleControl=lambda actor,u:(actor,u)),VehicleControl=lambda **u:u))
    client=S(apply_batch_sync=lambda cmds,tick:(calls.append((cmds,tick)) or [S(actor_id=7,has_error=lambda:False,error='')]))
    assert batch_control(client,7,STOP)==dict(server_processed=True,physics_confirmed=False,due_tick_cue=False,actor_id=7)
    assert calls[0][1] is False
    bad=S(apply_batch_sync=lambda cmds,tick:[S(actor_id=7,has_error=lambda:True,error='missing')])
    with pytest.raises(RuntimeError,match='BatchProcessingError'):batch_control(bad,7,STOP)


def test_command_competition_requires_old_processing_return_then_new_brake():
    g=CommandGuard();normal=g.prepare('IndependentTestControl',dict(STOP,throttle=.5,brake=0.),1.,snapshot());g.latch()
    with pytest.raises(ValueError,match='ForbiddenOrLocked'):g.prepare('IndependentTestControl',STOP,1.1)
    with pytest.raises(ValueError,match='LateNormal'):g.confirm(normal,True)
    with pytest.raises(ValueError,match='HandoffUncertain'):g.before_tick(.55,1.2)
    brake=g.prepare('FaultBrake',STOP,1.2);g.confirm(brake,True)
    assert g.before_tick(.55,1.2)['source']=='FaultBrake'


def test_failed_normal_rpc_never_cleared_by_brake_guess():
    g=CommandGuard();normal=g.prepare('IndependentTestControl',STOP,1.)
    with pytest.raises(ValueError,match='HandoffUncertain'):g.confirm(normal,False)
    g.latch();brake=g.prepare('FaultBrake',STOP,1.1);g.confirm(brake,True)
    with pytest.raises(ValueError,match='HandoffUncertain'):g.before_tick(.05,1.2)


def test_hold_source_cannot_leak_into_motion_ticks():
    g=CommandGuard();hold=g.prepare('PlannedHoldBrake',STOP,1.);g.confirm(hold,True)
    with pytest.raises(ValueError,match='HoldCannot'):g.before_tick(.1,1.2)


def test_ledger_empty_and_no_second_start():
    l=reserve_start(None);assert len(l['runs'])==1 and l['runs'][0]['model_forwards']==0
    with pytest.raises(ValueError,match='StartBudget'):reserve_start(l)
    assert sum(CFG['tick_caps'].values())==800


def test_real_owner_orders_target_registration_before_target_tick(tmp_path):
    r=Runtime(tmp_path,CFG);r.current=snapshot(12);r.observation=snapshot(10);events=[]
    r.pump=lambda:None;r.phase=lambda name,deadline:events.append(('phase',name))
    r.register=lambda frame:(events.append(('register',frame)),setattr(r,'target',frame))
    def tick(phase):r.current=snapshot(r.current['frame']+1);events.append(('tick',r.current['frame']))
    r.tick=tick
    r.call=lambda *args,**kw:dict(observation=snapshot(14),frozen=snapshot(16),capture_wall_s=.15)
    r.event=lambda *args,**kw:None
    r.capture('stationary',first=False)
    assert events.index(('register',14))<events.index(('tick',14))
    assert [x for x in events if x[0]=='tick']==[('tick',13),('tick',14),('tick',15),('tick',16)]


def test_initialization_failure_runs_cleanup_and_records_partial(tmp_path,monkeypatch):
    r=Runtime(tmp_path,CFG);r.start=lambda *args,**kw:None;r.ready=lambda *args,**kw:None
    def call(kind,*args,**kwargs):
        if kind=='setup':raise RuntimeError('SyntheticSetupFailure')
        if kind=='cleanup':return dict(cleanup={'status':'NoWorldOwned'})
        raise AssertionError(kind)
    r.call=call;r.latch=lambda reason:setattr(r,'terminal',dict(reason=reason,wall_s=1.))
    r.run();result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']=='Partial' and result['cleanup']['status']=='NoWorldOwned'
    assert result['charged_ticks']==30 and result['model_controls']==0


def test_runtime_no_model_pid_physics_override_and_watchdog_no_tick():
    files=['tools/m8j13_logic_runtime.py','tools/m8j13_logic_services.py','simulator/m8j13_logic_runtime.py']
    banned={'set_target_velocity','enable_constant_velocity','set_simulate_physics','set_transform','apply_physics_control','control_pid'}
    for f in files:
        tree=ast.parse(Path(f).read_text())
        assert not any(isinstance(n,ast.Attribute) and n.attr in banned for n in ast.walk(tree))
        assert not any(isinstance(n,ast.Import) and any(a.name in ('torch','mmcv') for a in n.names) for n in ast.walk(tree))
    tree=ast.parse(Path(files[1]).read_text());wd=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='watchdog')
    assert not any(isinstance(n,ast.Attribute) and n.attr=='tick' for n in ast.walk(wd))


def test_actual_owner_poll_loop_stays_live_during_fake_io_delay(tmp_path):
    import threading,time
    for name in ('io_requests','io_responses'):(tmp_path/name).mkdir()
    r=Runtime(tmp_path,CFG)
    def fake_io():
        while not list((tmp_path/'io_requests').glob('*.json')):time.sleep(.001)
        p=next((tmp_path/'io_requests').glob('*.json'));q=json.loads(p.read_text());time.sleep(.08)
        from tools.m8j13_common import atomic
        atomic(tmp_path/'io_responses'/p.name,dict(sequence=q['sequence'],success=True))
    t=threading.Thread(target=fake_io);t.start()
    assert r.call('frozen',timeout=.2)['success'];t.join()
    assert len(r.owner_gaps)>10 and max(r.owner_gaps)<.3 and r.pending is None


def test_actual_network_isolated_worker_heartbeat_does_not_deliver_blocked_task(tmp_path):
    import subprocess,time,os
    from tools.m8j13_common import atomic
    for name in ('tasks','task_results'):(tmp_path/name).mkdir()
    atomic(tmp_path/'frozen_config.json',CFG)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()))
    proc=subprocess.Popen(['unshare','--net',sys.executable,'-m','tools.m8j13_logic_services','--role','test','--root',str(tmp_path)],env=env)
    def wait(path,timeout=3):
        until=time.monotonic()+timeout
        while not path.exists():
            assert proc.poll() is None
            if time.monotonic()>until:raise AssertionError('worker timeout')
            time.sleep(.005)
        return json.loads(path.read_text())
    try:
        ready=wait(tmp_path/'test_worker_ready.json');assert ready['network_namespace']!=os.readlink('/proc/self/ns/net')
        ch=TaskChannel(IDENTITY);now=time.monotonic();obs=dict(snapshot(),input_wall_s=now)
        t=ch.request(obs,snapshot(12),now,'normal',STOP);atomic(tmp_path/'tasks'/'000001.json',t)
        r=wait(tmp_path/'task_results'/'000001.json')
        assert r['worker_completed_wall_s']-r['worker_received_wall_s']>=.6
        ch.consume(r,snapshot(12),time.monotonic())
        now=time.monotonic();obs['input_wall_s']=now;t=ch.request(obs,snapshot(12),now,'blocked',STOP);atomic(tmp_path/'tasks'/'000002.json',t)
        while time.monotonic()<=t['deadline_wall_s']+.03:time.sleep(.01)
        hb=wait(tmp_path/'test_worker_heartbeat.json');assert time.monotonic()-hb['wall_s']<.3
        assert not (tmp_path/'task_results'/'000002.json').exists()
        s=dict(armed=True,owner_wall_s=time.monotonic(),run_deadline=now+240,phase='ExpectedComputeWait',phase_deadline=t['deadline_wall_s'])
        assert supervision_reason(s,time.monotonic())=='TaskDeadline'
        ch.locked=True;(tmp_path/'release_late_result').touch();r=wait(tmp_path/'task_results'/'000002.json')
        with pytest.raises(ValueError,match='LateResultLocked'):ch.consume(r,snapshot(12),time.monotonic())
    finally:
        (tmp_path/'test_worker_stop').touch();proc.wait(timeout=3)
    assert proc.returncode==0


def test_late_failed_tick_reply_allows_cleanup_only_never_physical_resumption(tmp_path):
    from tools.m8j13_common import atomic
    (tmp_path/'io_responses').mkdir()
    r=Runtime(tmp_path,CFG);r.pending=dict(sequence=5,kind='tick',deadline=1.)
    r.guard.latch();r.guard.uncertain=True;r.terminal=dict(reason='IODeadline:tick',wall_s=1.)
    atomic(tmp_path/'io_responses'/'000005.json',dict(success=False,error='500ms timeout',call_end_wall_s=1.003))
    assert r.reconcile_for_cleanup()
    assert r.pending is None and r.terminal['reason']=='IODeadline:tick'
    assert r.result['late_io_cleanup_reconciliation']['physics_resumption_allowed'] is False
    with pytest.raises(ValueError,match='HandoffUncertain'):r.guard.before_tick(.1,1.1)
    assert not r.counts


def test_missing_io_reply_remains_inflight_for_cleanup(tmp_path):
    (tmp_path/'io_responses').mkdir()
    r=Runtime(tmp_path,CFG);r.pending=dict(sequence=1,kind='tick',deadline=1.)
    assert not r.reconcile_for_cleanup(timeout=.005)
    assert r.pending is not None and not r.counts
