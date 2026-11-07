"""Logical shadow dispatch and actual owner/worker helper wiring; no model/server."""
import copy,json,time,threading,ast
from pathlib import Path
import numpy as np
import pytest
from simulator.m8j13_logic_shadow import PROTOCOL,CHECKS,ENDPOINTS,ShadowChannel,require_notification,require_parked_execution,memory_valid
from simulator.m8j13_logic_runtime import STOP,supervision_reason
from simulator.m8j13_safety import PlanStamp,freshness
from tools.m8j13_logic_shadow import ShadowRuntime
from tools.m8j13_logic_shadow_diagnostics import transported_pid
from tools.m8j13_common import atomic


def cfg():return json.loads(Path('configs/m8j13_logic_shadow.json').read_text())
def snap(frame=12):return dict(frame=frame,sim_s=frame*.05,pose=np.eye(4).tolist(),velocity=[0,0,0],speed=0.,speed_norm=0.,input_wall_s=1.)
def prepare():
    ch=ShadowChannel(['test',42,'g']);obs=snap(10);t=ch.request(obs,snap(),1.15,99.)
    return ch,t

def reply(t,done=1.8):
    times={k:done for k in ENDPOINTS};times['worker_received']=1.16
    return dict(protocol=PROTOCOL,identity=t['identity'],sequence=t['sequence'],frame=t['frame'],input_sim_s=t['input_sim_s'],input_wall_s=t['input_wall_s'],task_deadline=t['deadline_wall_s'],validation_start=done,validation_deadline=min(done+.1,t['deadline_wall_s'],t['input_wall_s']+4,t['run_deadline']),checks={k:True for k in CHECKS},times=times,model_forward=True)


def test_explicit_logic_accepts_old_realtime_expiry_without_altering_old_rule():
    ch,t=prepare();r=reply(t);ch.consume(r,snap(),1.85);ch.finalize_return(r,1.851)
    assert ch.consumed=={1}
    s=PlanStamp('test',10,.5,1.,1.2,1.8,'frozen');assert freshness(s,'test',12,.6,1.85)=='PlanWallExpired'


@pytest.mark.parametrize('change,reason',[('missing','Schema'),('extra','Schema'),('false','Checks'),('identity','Identity'),('sequence','Sequence'),('renew','Resigned'),('validation','Deadline'),('future','Clock')])
def test_completion_schema_identity_and_absolute_deadline(change,reason):
    ch,t=prepare();r=reply(t)
    if change=='missing':del r['checks']['memory']
    if change=='extra':r['throttle']=0
    if change=='false':r['checks']['finite']=False
    if change=='identity':r['identity'][2]='old' # task uses same list; restore task expected separately
    if change=='identity':ch.identity=['test',42,'g'];ch.task['identity']=['test',42,'g']
    if change=='sequence':r['sequence']=2
    if change=='renew':r['task_deadline']+=1
    if change=='validation':r['validation_deadline']+=.1
    if change=='future':r['times']['publish_start']=2.
    with pytest.raises(ValueError,match=reason):ch.consume(r,snap(),1.85)


def test_actual_return_checked_and_single_consumption_lock():
    ch,t=prepare();r=reply(t);ch.consume(r,snap(),1.89)
    with pytest.raises(ValueError,match='AtReturn'):ch.finalize_return(r,1.91)
    assert not ch.consumed
    ch.finalize_return(r,1.895)
    with pytest.raises(ValueError,match='Repeated'):ch.consume(r,snap(),1.896)
    ch.locked=True
    with pytest.raises(ValueError,match='Locked'):ch.consume(r,snap(),1.897)


@pytest.mark.parametrize('what',['pending','frame','velocity','sim'])
def test_live_frozen_confirmation_is_required(what):
    ch,t=prepare();r=reply(t);s=snap()
    if what=='frame':s['frame']+=1
    if what=='velocity':s['velocity'][0]=.001
    if what=='sim':s['sim_s']+=.31
    with pytest.raises(ValueError):ch.consume(r,s,1.85,pending=what=='pending')


def test_24_sequential_budget_and_real_point_two_interval():
    ch=ShadowChannel(['test',42,'g'])
    for i in range(24):
        t=ch.request(snap(10+i*4),snap(12+i*4),1.15,99);r=reply(t)
        ch.consume(r,snap(12+i*4),1.85);ch.finalize_return(r,1.851)
    with pytest.raises(ValueError,match='Budget'):ch.request(snap(106),snap(108),1.15,99)
    ch,_=prepare();ch.consumed.add(1)
    with pytest.raises(ValueError,match='Cadence'):ch.request(snap(15),snap(17),1.15,99)


@pytest.mark.parametrize('source,control,tick', [('IndependentTestControl',STOP,False),('InitialProtectionBrake',dict(STOP,throttle=.1),False),('PlannedHoldBrake',STOP,True),('ModelOutput',STOP,False)])
def test_execution_side_brake_enforced(source,control,tick):
    with pytest.raises(ValueError):require_parked_execution(cfg(),dict(source=source,control=control),tick)
    require_parked_execution({},dict(source=source,control=control),tick)  # old mode unchanged


def test_real_transport_uses_current_pose_speed_and_one_pid_without_mutating_native():
    path=np.column_stack([np.zeros(15),np.arange(1,16.)]);prediction=dict(lat_reg_final=path,lon_reg_final=np.ones(6)*6,traj_final=np.zeros((6,2)))
    row=dict(ego_world_matrix=np.eye(4).tolist(),target_world_xy=[20,0]);frozen=snap();frozen['pose'][0][3]=2.;frozen['speed']=.7;calls=[]
    class C:
        def command(self,out,speed,target):
            calls.append((out['lat_reg_final'].copy(),speed,target.copy()));return dict(raw_pid=dict(steer=0,throttle=1,brake=0),native_agent_control=dict(steer=0,throttle=1,brake=0))
    invoke,meta=transported_pid(prediction,row,frozen,C());assert not calls
    result,a,b=invoke();assert len(calls)==1 and a is None and b is None
    np.testing.assert_array_equal(calls[0][0][:,1],path[:,1]-2);assert calls[0][1]==.7
    np.testing.assert_array_equal(calls[0][2],[0,18]);np.testing.assert_array_equal(path,prediction['lat_reg_final'])


def test_memory_history_is_exactly_previous_input():
    empty=dict(cached=False,history_timestamp=None);after=dict(cached=True,history_timestamp=[.5])
    memory_valid(empty,after,None,.5);memory_valid(after,dict(cached=True,history_timestamp=[.7]),.5,.7)
    with pytest.raises(ValueError):memory_valid(after,after,None,.5)
    with pytest.raises(ValueError):memory_valid(after,after,.4,.7)


def test_real_owner_completion_no_control_payload_or_model_actuation(tmp_path):
    for n in ('shadow_tasks','shadow_completions'):(tmp_path/n).mkdir()
    r=ShadowRuntime(tmp_path,cfg());r.channel=ShadowChannel(['test',42,'g']);r.current=snap();r.observation=snap(10);r.observation['input_wall_s']=time.monotonic();r.state['run_deadline']=time.monotonic()+10
    commands=[];r.issue=lambda source,control,*a:commands.append((source,dict(control)));r.call=lambda *a,**kw:dict(snapshot=r.current);r.pump=lambda:None
    def model():
        path=tmp_path/'shadow_tasks'/'000001.json'
        while not path.exists():time.sleep(.001)
        t=json.loads(path.read_text())['task'];now=time.monotonic();v=reply(t,now);v['times']['worker_received']=now;atomic(tmp_path/'shadow_completions'/path.name,v)
    thread=threading.Thread(target=model);thread.start();r.compute_shadow(dict(shadow_capture=dict(row={})));thread.join(2)
    assert commands==[('PlannedHoldBrake',STOP),('InitialProtectionBrake',STOP)]
    d=r.result['completions'][0];assert d['consume_return_wall_s']>=d['consume_reference_wall_s']>=d['received_wall_s']
    assert r.channel.consumed=={1}


def test_error_latch_can_publish_even_when_model_failed(tmp_path):
    r=ShadowRuntime(tmp_path,cfg());r.state['armed']=True;r.state['fault']='ShadowWorker:bad'
    atomic(tmp_path/'model_error.json',dict(error='bad'));r.pump()
    assert supervision_reason(r.state,time.monotonic())=='ShadowWorker:bad'


def test_normal_end_and_fault_end_remain_distinct_and_worker_has_no_carla():
    source=Path('tools/m8j13_logic_shadow_worker.py').read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('tick','apply_control','set_target_velocity','optimizer','step') for n in ast.walk(tree))
    src=Path('tools/m8j13_logic_shadow.py').read_text();assert "self.latch('NormalShadowCompletion')" in src
    assert 'shadow_completions' in src and "self.root/'diagnostics'" not in src
    assert sum(cfg()['tick_caps'].values())==800 and cfg()['tick_caps']['stationary']==95
