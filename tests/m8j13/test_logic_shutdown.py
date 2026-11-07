"""Exercise production owner/supervisor finalizers with fake clocks and children.
No CARLA server, model imports/loads, real PID or GPU work.
"""
import copy,json,signal
from pathlib import Path
import pytest
from tools.m8j13_common import atomic
from tools.m8j13_logic_shadow import ShadowRuntime
from tools.m8j13_logic_shutdown import (first_fault,shutdown_notice,finish_processes,
    finalize_supervisor,final_model_record,signal_model_stop)
from tools.m8j13_logic_services import ServerAwareDeadline
from simulator.m8j13_logic_runtime import STOP,operation_stamp
from simulator.m8j13_logic_shadow import ShadowChannel

CFG=json.loads(Path('configs/m8j13_logic_shadow_recheck.json').read_text())
class Clock:
    def __init__(self):self.now=100.
    def monotonic(self):return self.now
    def sleep(self,n):self.now+=n
class Proc:
    def __init__(self,pid,c,exit_at=None,on_exit=None,code=0):
        self.pid=pid;self.c=c;self.exit_at=exit_at;self.returncode=None;self.on_exit=on_exit;self.code=code
    def poll(self):
        if self.returncode is None and self.exit_at is not None and self.c.now>=self.exit_at:
            self.returncode=self.code
            if self.on_exit:self.on_exit()
        return self.returncode
    def wait(self,timeout):
        import subprocess
        if self.poll() is None:raise subprocess.TimeoutExpired('fake',timeout)
        return self.returncode
    def kill(self):self.returncode=-9


def owner(tmp_path,monkeypatch):
    import tools.m8j13_logic_runtime as module
    c=Clock();monkeypatch.setattr(module.time,'monotonic',c.monotonic);monkeypatch.setattr(module.time,'sleep',c.sleep)
    for n in ['io_requests','io_responses']:(tmp_path/n).mkdir()
    r=ShadowRuntime(tmp_path,CFG);r.current=dict(frame=35,sim_s=1.75,pose=[[1]],velocity=[0,0,-.98],speed_norm=.98,full_control=STOP)
    r.state.update(armed=True,ego_id=9,phase='Initialization',phase_deadline=130.,active_target=None)
    r.channel=ShadowChannel(['test',1,'g']);r.settings_charges['settings_startup']=30
    return r,c


@pytest.mark.parametrize('inflight',[None,'tick','control'])
def test_actual_owner_dead_server_latches_once_and_saves_without_rpcs(tmp_path,monkeypatch,inflight):
    r,c=owner(tmp_path,monkeypatch)
    if inflight:
        r.pending=operation_stamp(5,inflight,100.,2.,130.,340.)
        if inflight=='tick':r.counts['initialization']=3
    atomic(tmp_path/'server_exited.json',dict(wall_s=100.,confirmed=True,wrapper_returncode=139))
    r.pump();first=json.loads((tmp_path/'first_fault.json').read_text());epoch=r.guard.epoch
    c.sleep(3);r.pump();r.latch('LaterTimeout')
    assert r.guard.epoch==epoch and json.loads((tmp_path/'first_fault.json').read_text())==first
    r.finish_owner();result=json.loads((tmp_path/'result.json').read_text())
    assert result['fault']['reason']=='ServerExited' and result['stop']['ticks']==0
    assert result['cleanup']['settings_restored'] is None
    assert result['server_advance_unknown']==(inflight=='tick')
    assert r.guard.uncertain==(inflight is not None)
    assert list((tmp_path/'io_requests').glob('*'))==[]
    assert result['model_controls']==0 and (tmp_path/'model_stop').exists()


def test_late_reply_is_cleanup_only_even_after_server_death(tmp_path,monkeypatch):
    r,c=owner(tmp_path,monkeypatch);r.pending=operation_stamp(5,'tick',100.,.5,130.,340.)
    atomic(tmp_path/'server_exited.json',dict(confirmed=True));r.pump();c.sleep(1)
    atomic(tmp_path/'io_responses'/'000005.json',dict(success=True,call_end_wall_s=101.))
    r.reconcile_for_cleanup()
    assert r.pending is None and r.guard.locked and r.guard.uncertain and r.advance_unknown
    with pytest.raises(RuntimeError,match='ServerExited'):r.call('tick',{},allow_fault=True)
    assert not list((tmp_path/'io_requests').glob('*'))


def test_death_between_nested_rpcs_does_not_issue_next(tmp_path):
    c=Clock();b=ServerAwareDeadline(operation_stamp(1,'control',100.,.2,130.,340.),c.monotonic,tmp_path);calls=[]
    def first():calls.append('first');atomic(tmp_path/'server_exited.json',dict(confirmed=True))
    with pytest.raises(RuntimeError,match='ServerExited'):b.call(first)
    with pytest.raises(RuntimeError,match='ServerExited'):b.call(lambda:calls.append('second'))
    assert calls==['first']


def test_supervisor_actual_finally_allows_owner_model_to_flush(tmp_path):
    c=Clock();server=Proc(3,c,100.,code=139);server.poll()
    ownerp=Proc(1,c,101.,lambda:atomic(tmp_path/'result.json',dict(charged_ticks=33,server_advance_unknown=True)))
    model=Proc(2,c,102.,lambda:atomic(tmp_path/'model_final.json',dict(attempts=0,successful_forwards=0,pid_calls=0)))
    procs=dict(owner=ownerp,model=model,server=server);sent=[];ledger=[]
    def groups(pid):return [] if procs[{1:'owner',2:'model',3:'server'}[pid]].poll() is not None else [pid]
    report=finalize_supervisor(tmp_path,CFG,procs,dict(error='ServerExited'),lambda *a,**kw:ledger.append(kw),groups,lambda p:False,[2300],clock=c.monotonic,sleep=c.sleep,send_group=lambda *x:sent.append(x))
    assert not sent and report['process_cleanup'] and report['model_final']['attempts']==0
    assert report['result']['charged_ticks']==33 and ledger[0]['charged_ticks']==800
    assert not report.get('missing_child_evidence') and c.now<103
    assert json.loads((tmp_path/'server_exited.json').read_text())['engine_signal'] is None


def test_hung_children_share_deadlines_even_when_one_signal_fails(tmp_path):
    c=Clock();procs={k:Proc(i,c) for i,k in enumerate(['owner','model','server'],1)};sent=[];ledger=[]
    def send(pid,sig):
        sent.append((pid,int(sig),c.now))
        if pid==1 and sig==signal.SIGTERM:raise PermissionError('fake TERM failure')
        if sig==signal.SIGKILL:next(p for p in procs.values() if p.pid==pid).returncode=-9
    groups=lambda pid:[] if next(p for p in procs.values() if p.pid==pid).poll() is not None else [pid]
    r=finalize_supervisor(tmp_path,CFG,procs,dict(error='SyntheticFailure'),lambda *a,**kw:ledger.append(kw),groups,lambda p:False,[2300],clock=c.monotonic,sleep=c.sleep,send_group=send)
    assert c.now<=138. and all(p.poll()==-9 for p in procs.values())
    assert r['process_cleanup'] and r['shutdown']['errors'] and ledger
    assert set(r['missing_child_evidence'])=={'owner','model'}
    assert not (tmp_path/'result.json').exists() and not (tmp_path/'model_final.json').exists()
    assert min(t for p,s,t in sent if p==2)>=130.
    assert 117.<=max(t for p,s,t in sent if p==1)<118.


def test_repeat_shutdown_and_earlier_worker_stop_do_not_renew(tmp_path):
    c=Clock();signal_model_stop(tmp_path,c.now);c.sleep(5)
    n=shutdown_notice(tmp_path,'Failure',clock=c.monotonic);c.sleep(8)
    assert shutdown_notice(tmp_path,'Later',clock=c.monotonic)==n
    assert n['model_deadline_wall_s']==130. and n['owner_deadline_wall_s']==120.
    assert json.loads((tmp_path/'first_fault.json').read_text())['reason']=='Failure'


def test_worker_owned_final_counters_survive_digest_error(tmp_path):
    def broken(x):raise RuntimeError('digest unavailable')
    r=final_model_record(tmp_path,dict(attempts=2,successful_forwards=1,pid_calls=1,completed=1),object(),'base',broken,lambda m:{'cached':True})
    assert r['attempts']==2 and r['fixed_state_unchanged'] is None and r['finalization_errors']
    assert json.loads((tmp_path/'model_final.json').read_text())==r


def test_owner_cleanup_exception_does_not_skip_other_child_or_result(tmp_path,monkeypatch):
    r,c=owner(tmp_path,monkeypatch);r.terminal=dict(reason='NormalShadowCompletion',wall_s=100.)
    r.stop_observe=lambda:r.result.update(stop=dict(status='SyntheticHoldOnly'))
    def bad(*args,**kwargs):raise RuntimeError('cleanup step failed')
    r.call=bad;r.processes={'watchdog':Proc(1,c,100.),'io':Proc(2,c,100.)}
    r.finish_owner();out=json.loads((tmp_path/'result.json').read_text())
    assert out['finalization_errors'][0]['step']=='resource_cleanup'
    assert out['remaining_processes']==[] and (tmp_path/'io_stop').exists()


def test_first_fault_exclusive_publication(tmp_path):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(lambda x:first_fault(tmp_path,x,x,now=100.),['owner','watchdog']))
    assert values[0]==values[1]
    assert not list(tmp_path.glob('.first_fault*'))


def test_real_stub_processes_flush_before_forcing_own_groups(tmp_path):
    """Actual subprocess lifecycle; stubs contain no simulator/model/PID code."""
    import subprocess,sys,time,os
    from tools.m8j12_capture_supervisor import group_members
    source="""import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);role=sys.argv[2]
while not (p/('shutdown_request.json' if role=='owner' else 'model_stop')).exists():time.sleep(.01)
(p/('result.json' if role=='owner' else 'model_final.json')).write_text(json.dumps({'stub':True,'charged_ticks':0}))
"""
    procs={role:subprocess.Popen([sys.executable,'-c',source,str(tmp_path),role],start_new_session=True) for role in ['owner','model']}
    try:
        report=finalize_supervisor(tmp_path,CFG,procs,{},lambda *a,**kw:None,group_members,lambda p:False,[],clock=time.monotonic)
        assert all(v['normal_exit'] for v in report['shutdown']['processes'].values())
        assert report['result']['stub'] and report['model_final']['stub']
        assert report['process_cleanup']
    finally:
        for p in procs.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=2)


def test_owner_run_uses_finalizer_after_server_exit_during_setup(tmp_path,monkeypatch):
    r,c=owner(tmp_path,monkeypatch);r.start=lambda *a,**kw:None;r.ready=lambda *a,**kw:{}
    def call(kind,*a,**kw):
        assert kind=='setup'
        atomic(tmp_path/'server_exited.json',dict(confirmed=True,wall_s=c.now))
        r.pump();raise RuntimeError('ServerExited_InFlightResultUnverified')
    r.call=call;r.run()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['first_fault']['reason']=='ServerExited'
    assert result['cleanup']['settings_restored'] is None and not result['remaining_processes']


def test_watchdog_start_after_confirmed_death_has_no_client_or_rpc(tmp_path,monkeypatch):
    import sys
    from tools.m8j13_logic_services import watchdog
    monkeypatch.setitem(sys.modules,'carla',None)
    atomic(tmp_path/'server_exited.json',dict(confirmed=True,wall_s=100.))
    atomic(tmp_path/'owner_state.json',dict(phase='Initialization',epoch=0,operation={'kind':'tick'}))
    watchdog(tmp_path,CFG)
    assert json.loads((tmp_path/'fault_brake.json').read_text())['status']=='CommandNotDelivered_ServerExited'
    assert json.loads((tmp_path/'first_fault.json').read_text())['reason']=='ServerExited'


def test_supervisor_terminal_summary_is_attempted_if_port_check_raises(tmp_path):
    c=Clock();ledger=[]
    def broken_port(p):raise OSError('unavailable')
    report=finalize_supervisor(tmp_path,CFG,{},dict(error='Synthetic'),lambda *a,**kw:ledger.append(kw),lambda p:[],broken_port,[2300],clock=c.monotonic,sleep=c.sleep)
    assert (tmp_path/'supervisor.json').exists() and ledger and not report['process_cleanup']
    assert report['ports_after']['2300'] is None and report['finalization_errors']
