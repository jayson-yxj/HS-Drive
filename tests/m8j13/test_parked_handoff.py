"""Real owner/I/O admission with deterministic delays; no CARLA/model/PID."""
import copy
import json
from pathlib import Path
import pytest
from simulator.m8j13_logic_runtime import STOP, operation_stamp, OperationDeadline
from simulator.m8j13_parked_handoff import handoff_proof, verify_io_proof, uncertainty
from tools.m8j13_logic_shadow import ShadowRuntime
from tools.m8j13_logic_runtime import Runtime
from tests.m8j13.test_logic_deadlines import Clock

CFG=json.loads(Path('configs/m8j13_logic_shadow_recheck.json').read_text())


def setup_runtime(tmp_path,monkeypatch,restore=False,old=False):
    import tools.m8j13_logic_runtime as module
    clock=Clock();monkeypatch.setattr(module.time,'monotonic',clock.monotonic);monkeypatch.setattr(module.time,'sleep',clock.sleep)
    for n in ('io_requests','io_responses'):(tmp_path/n).mkdir()
    cfg=json.loads(Path('configs/m8j13_logic_shadow.json').read_text()) if old else CFG
    r=ShadowRuntime(tmp_path,cfg)
    s=dict(frame=12,sim_s=.6,pose=[[1]],velocity=[0,0,0],speed=0.,speed_norm=0.,full_control=dict(STOP),feedback_matches=True,identity=['test',1,'g'],ego_id=9,epoch=0)
    r.current=s;r.observation=dict(s,frame=10,sim_s=.5,input_wall_s=99.8)
    r.state.update(armed=True,identity=s['identity'],ego_id=9,phase='FiniteExecution' if restore else 'ConfirmedFrozenWorld',phase_started=100.,phase_deadline=101. if restore else 100.5,frozen_confirmation=copy.deepcopy(s))
    r.guard.active=dict(sequence=1,epoch=0,source='PlannedHoldBrake' if restore else 'InitialProtectionBrake',control=dict(STOP),observation=None)
    return r,clock


def delayed_response(r,clock,monkeypatch,duration):
    import tools.m8j13_logic_runtime as module
    original=module.load
    def load(path):
        req=r.root/'io_requests'/path.name
        if path.parent.name=='io_responses' and req.exists():
            request=json.loads(req.read_text());op=request['operation']
            if clock.now>=op['started']+duration:
                return dict(sequence=op['sequence'],operation=op,success=True,server_processed=True)
        return original(path)
    monkeypatch.setattr(module,'load',load)


@pytest.mark.parametrize('restore',[False,True])
def test_three_hundred_ms_uses_real_owner_extended_operation(tmp_path,monkeypatch,restore):
    r,c=setup_runtime(tmp_path,monkeypatch,restore);delayed_response(r,c,monkeypatch,.3)
    r.planned_issue('InitialProtectionBrake' if restore else 'PlannedHoldBrake')
    req=json.loads(next((tmp_path/'io_requests').glob('*.json')).read_text())
    assert req['operation']['operation_limit_s']==.5 and not r.guard.uncertain
    assert req['operation']['deadline']==pytest.approx(100.5)


@pytest.mark.parametrize('old,source',[(True,'PlannedHoldBrake'),(False,'FaultBrake'),(False,'InitialProtectionBrake')])
def test_ordinary_initial_fault_and_old_contract_remain_point_two(tmp_path,monkeypatch,old,source):
    r,c=setup_runtime(tmp_path,monkeypatch,old=old);delayed_response(r,c,monkeypatch,.3)
    with pytest.raises(TimeoutError):r.issue(source,STOP)
    req=json.loads(next((tmp_path/'io_requests').glob('*.json')).read_text())
    assert req['operation']['operation_limit_s']==.2 and r.guard.uncertain
    state=json.loads((tmp_path/'owner_state.json').read_text());assert uncertainty(state)[0]
    r.pending=None;r.pump();assert uncertainty(json.loads((tmp_path/'owner_state.json').read_text()))[0]


@pytest.mark.parametrize('limit_kind',['operation','phase','observation'])
def test_original_bound_truncates_and_never_renews(tmp_path,monkeypatch,limit_kind):
    r,c=setup_runtime(tmp_path,monkeypatch)
    if limit_kind=='phase':r.state['phase_deadline']=100.25
    if limit_kind=='observation':r.observation['input_wall_s']=96.25
    delayed_response(r,c,monkeypatch,.51 if limit_kind=='operation' else .3)
    with pytest.raises(TimeoutError):r.planned_issue('PlannedHoldBrake')
    assert r.guard.uncertain and r.failed_operation


@pytest.mark.parametrize('bad',['moving','feedback','frozen','identity','actor','epoch','pending','uncertain','locked','phase','active','source'])
def test_admission_cannot_borrow_exception(tmp_path,monkeypatch,bad):
    r,c=setup_runtime(tmp_path,monkeypatch);source='PlannedHoldBrake';kw={}
    if bad=='moving':r.current['speed_norm']=.11
    if bad=='feedback':r.current['full_control']['throttle']=.1
    if bad=='frozen':r.state['frozen_confirmation']['frame']=13
    if bad=='identity':r.current['identity']=['other',1,'g']
    if bad=='actor':r.current['ego_id']=10
    if bad=='epoch':r.current['epoch']=1
    if bad=='pending':kw['pending']={'kind':'tick'}
    if bad=='uncertain':kw['uncertain']=True
    if bad=='locked':kw['locked']=True
    if bad=='phase':r.state['phase']='Initialization'
    if bad=='active':r.guard.active['source']='IndependentTestControl'
    if bad=='source':source='FaultBrake'
    with pytest.raises(ValueError):handoff_proof(CFG,source,STOP,r.current,r.state,active=r.guard.active,**kw)


def test_io_independently_checks_live_world_and_previous_command(tmp_path,monkeypatch):
    r,c=setup_runtime(tmp_path,monkeypatch)
    proof=handoff_proof(CFG,'PlannedHoldBrake',STOP,r.current,r.state,active=r.guard.active)
    command=dict(source='PlannedHoldBrake',control=STOP,epoch=0,observation=r.observation)
    op=operation_stamp(1,'control',100,.5,100.5,340,103.8);r.state['operation']=op
    def verify(cur,active=None):verify_io_proof(CFG,proof,command,cur,r.state,op,r.state['identity'],9,last=r.current,active=active or r.guard.active)
    verify(copy.deepcopy(r.current))
    changed=copy.deepcopy(r.current);changed['frame']+=1
    with pytest.raises(ValueError):verify(changed)
    wrong=copy.deepcopy(r.guard.active);wrong['sequence']+=1
    with pytest.raises(ValueError):verify(r.current,wrong)


def test_full_phase_queue_rpc_freeze_publication_and_consume_share_budget():
    c=Clock();op=operation_stamp(1,'control',100,.5,100.5,340);b=OperationDeadline(op,c.monotonic)
    c.sleep(.08);b.call(lambda:c.sleep(.30));c.sleep(.08);b.remaining()
    c.sleep(.05)
    with pytest.raises(TimeoutError):b.remaining()
    assert op['deadline']==100.5


def test_io_failure_survives_empty_owner_queue():
    op={'sequence':7,'kind':'control'}
    assert uncertainty({'operation':None,'handoff_uncertain':False},{'operation':op})==(True,op)


def test_actual_new_compute_wiring_consumes_and_restores_with_two_certified_calls(tmp_path,monkeypatch):
    """Execute real owner methods and I/O credentials with a fake transport clock."""
    from simulator.m8j13_logic_shadow import ShadowChannel,CHECKS,ENDPOINTS,PROTOCOL
    import tools.m8j13_logic_runtime as module
    from tools.m8j13_common import atomic
    r,c=setup_runtime(tmp_path,monkeypatch)
    for name in ('shadow_tasks','shadow_completions'):(tmp_path/name).mkdir()
    r.channel=ShadowChannel(r.state['identity']);original=module.load;io_active=copy.deepcopy(r.guard.active);seen={};limits=[]
    def fake_load(path):
        nonlocal io_active
        if path.parent.name=='io_responses':
            reqpath=r.root/'io_requests'/path.name
            if reqpath.exists():
                q=json.loads(reqpath.read_text());op=q['operation'];args=q['args']
                if op['sequence'] not in seen:
                    c.sleep(.015)
                    if q['kind']=='control':
                        verify_io_proof(CFG,args['parked_handoff'],args['command'],r.current,r.state,op,r.state['identity'],9,last=r.current,active=io_active)
                        limits.append(op);c.sleep(.03);io_active=copy.deepcopy(args['command'])
                    seen[op['sequence']]=dict(sequence=op['sequence'],operation=op,success=True,server_processed=True,snapshot=copy.deepcopy(r.current))
                return seen[op['sequence']]
        taskpath=tmp_path/'shadow_tasks'/'000001.json'
        if taskpath.exists() and not (tmp_path/'shadow_completions'/'000001.json').exists():
            t=json.loads(taskpath.read_text())['task'];c.sleep(.35);times={k:c.monotonic() for k in ENDPOINTS}
            atomic(tmp_path/'shadow_completions'/'000001.json',dict(protocol=PROTOCOL,identity=t['identity'],sequence=t['sequence'],frame=t['frame'],input_sim_s=t['input_sim_s'],input_wall_s=t['input_wall_s'],task_deadline=t['deadline_wall_s'],validation_start=c.monotonic(),validation_deadline=c.monotonic()+.1,checks={k:True for k in CHECKS},times=times,model_forward=True))
        return original(path)
    monkeypatch.setattr(module,'load',fake_load)
    r.compute_shadow(dict(shadow_capture=dict(row={})))
    assert len(limits)==2 and all(x['operation_limit_s']==.5 for x in limits)
    assert r.channel.consumed=={1} and r.guard.active['source']=='InitialProtectionBrake'
    assert r.result['completions'][0]['old_realtime_reason']=='PlanWallExpired'
    assert r.current['frame']==12 and not r.guard.uncertain


def test_same_stage_reentry_keeps_first_start_and_absolute_deadline(tmp_path,monkeypatch):
    r,c=setup_runtime(tmp_path,monkeypatch);c.sleep(.1)
    r.phase('ConfirmedFrozenWorld',101.)
    assert r.state['phase_started']==100. and r.state['phase_deadline']==100.5


def test_owner_final_processing_cannot_return_late_success(tmp_path,monkeypatch):
    r,c=setup_runtime(tmp_path,monkeypatch);delayed_response(r,c,monkeypatch,.3)
    confirm=r.guard.confirm
    def slow_confirm(*args):
        confirm(*args);c.sleep(.21)
    monkeypatch.setattr(r.guard,'confirm',slow_confirm)
    with pytest.raises(TimeoutError,match='DeadlineAtReturn'):r.planned_issue('PlannedHoldBrake')
    assert r.guard.uncertain and r.failed_operation['kind']=='control'
    record=json.loads((tmp_path/'planned_handoff_returns.jsonl').read_text())
    assert not record['passed'] and record['owner_return_wall_s']>100.5


def test_watchdog_refreshes_owner_uncertainty_after_independent_brake(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace as S
    from tools import m8j13_logic_services as svc
    from tools.m8j13_common import atomic
    r,c=setup_runtime(tmp_path,monkeypatch)
    r.state['fault']='FirstControlFault';r.pump()
    client=S(set_timeout=lambda x:None,get_client_version=lambda:'0.9.16',get_server_version=lambda:'0.9.16')
    monkeypatch.setitem(sys.modules,'carla',S(Client=lambda *a:client,libcarla=S(__file__='fake-client')))
    def brake(client,actor,control,budget):
        assert budget.operation['deadline']-budget.operation['started']==pytest.approx(.2)
        state=json.loads((tmp_path/'owner_state.json').read_text())
        state.update(handoff_uncertain=True,failed_operation=dict(sequence=10,kind='control'))
        atomic(tmp_path/'owner_state.json',state);(tmp_path/'logic_watchdog_stop').touch()
        return dict(server_processed=True)
    monkeypatch.setattr(svc,'batch_control',brake);svc.watchdog(tmp_path,CFG)
    result=json.loads((tmp_path/'fault_brake.json').read_text())
    assert result['server_processed'] and result['handoff_uncertain'] and not result['ordinary_handoff_confirmed']
    assert result['failed_operation']['sequence']==10
    assert json.loads((tmp_path/'watchdog_fault.json').read_text())['reason']=='FirstControlFault'


def test_persistent_io_failure_atomic_publication_preserves_first(tmp_path,monkeypatch):
    from tools import m8j13_logic_services as svc
    real_atomic=svc.atomic;published=[]
    def observe_atomic(path,value):
        assert not path.exists()
        real_atomic(path,value)
        published.append(svc.load(path))
    monkeypatch.setattr(svc,'atomic',observe_atomic)
    monkeypatch.setattr(svc,'write',lambda *a:pytest.fail('shared marker must be atomic'))
    first=dict(operation=dict(sequence=10,kind='control'),args=dict(command=dict(source='PlannedHoldBrake',control=STOP)))
    svc.retain_handoff_uncertainty(tmp_path,first,'OriginalFailure')
    svc.retain_handoff_uncertainty(tmp_path,dict(first,operation=dict(sequence=11)),'LaterFailure')
    assert len(published)==1 and published[0]['operation']['sequence']==10
    assert published[0]['error']=='OriginalFailure' and not published[0]['remote_order_confirmed']
