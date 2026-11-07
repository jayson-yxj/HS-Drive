"""Explicit one-start no-model synchronous_logic_v1 runtime test owner.

Owner event loop is the ONLY authority for ticks. Blocking CARLA calls run in
one IO subprocess, while an independent phase supervisor can submit braking.
No real model/PID is imported or invoked. All synthetic outputs are labelled.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from tools.m8j13_common import config,atomic,write,append,digest
from tools.m8j13_logic_services import load
from tools.m8j13_logic_shutdown import first_fault
from simulator.m8j13_logic_faults import scenario
from simulator.m8j13_logic_runtime import STOP,TaskChannel,CommandGuard,reserve_start,frozen_equal,operation_stamp,initialization_due

CFG=Path('configs/m8j13_logic_runtime_test.json')


class Runtime:
    def __init__(self,root,cfg):
        self.root=root;self.cfg=cfg;self.scenario=scenario(cfg);self.injection=None;self.sensor_actors=None;self.processes={};self.io_sequence=0;self.pending=None;self.current=None;self.counts=Counter()
        self.guard=CommandGuard();self.channel=None;self.task=None;self.result=dict(status='Partial',model_forwards=0,model_controls=0,training=0,stationary_cycles=[],moving_cycles=[])
        self.state=dict(armed=False,phase='Initialization',phase_deadline=time.monotonic()+30,owner_wall_s=time.monotonic(),run_deadline=time.monotonic()+240,epoch=0,ego_id=None,operation=None,fault=None,task=None)
        self.terminal=None;self.last_owner=None;self.owner_gaps=[];self.target=None;self.observation=None;self.started_motion=False;self.last_init_rpc_start=None;self.settings_charges=Counter();self.advance_unknown=False;self.cleanup_deadline=None;self.failed_operation=None;self.server_dead=False;self.finishing=False
    def event(self,kind,**kw):append(self.root/'owner_events.jsonl',dict(event=kind,wall_s=time.monotonic(),**kw))
    def phase(self,name,deadline):
        if name not in ('Cleanup','FaultLocked'):deadline=min(deadline,self.state['run_deadline'])
        self.state.update(phase=name,phase_started=time.monotonic(),phase_deadline=deadline);self.event('phase',phase=name,deadline=deadline);self.pump()
    def pump(self):
        now=time.monotonic()
        death=load(self.root/'server_exited.json');stop=load(self.root/'shutdown_request.json')
        if death or stop:
            self.server_dead=bool(death or stop.get('server_confirmed_exited'))
            if self.terminal is None:
                fault=first_fault(self.root,'ServerExited' if death else stop['reason'],'owner',dict(self.state,operation=self.pending),self.current,now)
                self.terminal=fault;self.guard.latch()
                if self.channel:self.channel.locked=True
            if self.pending and self.pending['kind']=='tick':self.advance_unknown=True;self.guard.uncertain=True
            if self.pending and self.pending['kind']=='control':self.guard.uncertain=True
            self.state.update(fault=self.terminal['reason'],server_confirmed_exited=self.server_dead,server_advance_unknown=self.advance_unknown)
            # Stop is latched, not repeatedly thrown: finalization must still run.
        if self.last_owner is not None:self.owner_gaps.append(now-self.last_owner)
        self.last_owner=now;self.state.update(owner_wall_s=now,operation=self.pending,epoch=self.guard.epoch,locked=self.guard.locked,handoff_uncertain=self.guard.uncertain,failed_operation=self.failed_operation)
        atomic(self.root/'owner_state.json',self.state)
        fault=load(self.root/'watchdog_fault.json')
        if fault and self.terminal is None:
            self.terminal=fault;self.guard.latch()
            if self.channel:self.channel.locked=True
            self.event('fault_observed',fault=fault)
        if self.finishing or self.server_dead:return
        for role,proc in self.processes.items():
            if role=='io' and self.pending and self.pending['kind']=='cleanup' and (self.root/'io_responses'/('%06d.json'%self.pending['sequence'])).exists():continue
            if proc.poll() is not None and not (self.root/{'test':'test_worker_stop','watchdog':'logic_watchdog_stop','io':'io_stop','manual':'manual_stop'}[role]).exists():raise RuntimeError('ServiceExited:'+role)
    def start(self,role,cpus,isolated=False):
        argv=['taskset','-c',cpus]+(['unshare','--net'] if isolated else [])+[self.cfg['python'],'-m','tools.m8j13_logic_services','--role',role,'--root',str(self.root)]
        with (self.root/(role+'.log')).open('x') as f:self.processes[role]=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT)
    def ready(self,path,timeout=10):
        until=time.monotonic()+timeout
        while not path.exists():
            self.pump()
            if self.terminal:raise RuntimeError(self.terminal['reason'])
            if time.monotonic()>until:raise TimeoutError('ServiceReadiness:'+path.name)
            time.sleep(.005)
        return load(path)
    def call(self,kind,args=None,timeout=None,allow_fault=False):
        if self.pending:raise RuntimeError('IOAlreadyInFlight')
        self.pump()
        if self.server_dead:raise RuntimeError('ServerExited_NoFurtherRPC')
        if self.terminal and not allow_fault:raise RuntimeError(self.terminal['reason'])
        timeout=timeout or (.5 if kind=='tick' else .2)
        self.io_sequence+=1;now=time.monotonic()
        cleanup=kind=='cleanup'
        phase_deadline=self.cleanup_deadline if cleanup else self.state['phase_deadline']
        run_deadline=self.cleanup_deadline if cleanup else self.state['run_deadline']
        # Bounded terminal braking has its own10s, following a run deadline.
        if allow_fault and kind=='tick':run_deadline=phase_deadline
        obs=(args or {}).get('command',{}).get('observation')
        if obs is None:obs=(args or {}).get('observation')
        extra=None if obs is None else obs['input_wall_s']+4.
        self.pending=operation_stamp(self.io_sequence,kind,now,timeout,phase_deadline,run_deadline,extra)
        if kind=='setup':self.settings_charges['settings_startup']=self.cfg['tick_caps'].get('settings_startup',30)
        if kind=='cleanup':self.settings_charges['settings_restore']=self.cfg['tick_caps'].get('settings_restore',30)
        if kind=='tick':
            phase=args['phase'];cap=self.cfg['tick_caps'][phase]
            if self.counts[phase]>=cap or sum(self.counts.values())+sum(self.settings_charges.values())>=self.cfg.get('max_active_ticks',800):raise RuntimeError('TickBudgetExhausted')
            self.counts[phase]+=1
        self.pump();name='%06d.json'%self.io_sequence
        atomic(self.root/'io_requests'/name,dict(sequence=self.io_sequence,kind=kind,args=args or {},operation=dict(self.pending),owner_submitted_wall_s=now))
        while True:
            self.pump()
            if self.server_dead:raise RuntimeError('ServerExited_InFlightResultUnverified')
            reply=load(self.root/'io_responses'/name)
            if reply:
                end=time.monotonic();pending=self.pending
                if reply['sequence']!=pending['sequence'] or reply['operation']!=pending:raise RuntimeError('IOOperationIdentityMismatch')
                if kind=='cleanup':(self.root/'io_stop').touch()
                if kind=='control' and (not reply['success'] or end>pending['deadline']):self.mark_handoff_uncertain(pending)
                self.pending=None;self.pump()
                # Include owner identity/state processing in the shared operation,
                # rather than treating file arrival as complete consumption.
                end=time.monotonic()
                if kind=='control' and end>pending['deadline']:self.mark_handoff_uncertain(pending)
                append(self.root/'io_consumption.jsonl',dict(operation=pending,reply=reply,owner_consumed_wall_s=end))
                if kind=='tick' and (not reply['success'] or end>pending['deadline']):
                    self.guard.uncertain=True;self.advance_unknown=True
                if end>pending['deadline']:raise TimeoutError('IODeadline:'+kind)
                if not reply['success']:raise RuntimeError(reply.get('rejection_reason',reply['error']))
                if self.terminal and not allow_fault:raise RuntimeError(self.terminal['reason'])
                return reply
            if time.monotonic()>self.pending['deadline']:
                # No repeated request, no cancellation fiction: keep in-flight identity.
                if kind=='tick':self.guard.uncertain=True;self.advance_unknown=True
                if kind=='control':self.mark_handoff_uncertain(self.pending)
                raise TimeoutError('IODeadline:'+kind)
            time.sleep(.002)
    def mark_handoff_uncertain(self,operation):
        self.guard.uncertain=True
        if self.failed_operation is None:self.failed_operation=dict(operation)
        self.state.update(handoff_uncertain=True,failed_operation=self.failed_operation)
        atomic(self.root/'owner_state.json',self.state)
    def issue(self,source,control,observation=None):
        cmd=self.guard.prepare(source,control,time.monotonic(),observation)
        try:r=self.call('control',dict(command=cmd),timeout=.2,allow_fault=source=='FaultBrake')
        except BaseException:self.guard.uncertain=True;raise
        self.guard.confirm(cmd,r.get('server_processed',False));self.event('server_processed',command=cmd)
        return cmd
    def wait_initialization_pacing(self):
        due=initialization_due(self.last_init_rpc_start,self.cfg.get('initialization_min_tick_interval_s',.05))
        if due is None:return
        start=time.monotonic()
        while time.monotonic()<due:
            self.pump()
            if self.terminal:raise RuntimeError(self.terminal['reason'])
            if time.monotonic()>=min(self.state['phase_deadline'],self.state['run_deadline']):raise TimeoutError('InitializationDeadlineDuringPacing')
            time.sleep(min(.005,max(0.,due-time.monotonic())))
        self.event('initialization_pacing',wait_start_wall_s=start,wait_end_wall_s=time.monotonic(),previous_rpc_start_wall_s=self.last_init_rpc_start)
    def tick(self,phase):
        if phase=='initialization':self.wait_initialization_pacing()
        if self.current is None:
            cmd=self.guard.active
            if cmd is None:raise RuntimeError('NoInitialControl')
        else:cmd=self.guard.before_tick(self.current['sim_s']+.05,time.monotonic())
        r=self.call('tick',dict(command=cmd,phase=phase),timeout=self.cfg.get('initialization_tick_wall_s',.5) if phase=='initialization' else .5,allow_fault=phase in ('fault_stop','exit'))
        if phase=='initialization':self.last_init_rpc_start=r['rpc_start_wall_s']
        self.current=r['snapshot'];self.event('snapshot_confirmed',frame=self.current['frame'],sim_s=self.current['sim_s'],source=cmd['source'])
        speed=self.current['speed_norm']
        if phase=='stationary' and speed>.1:raise RuntimeError('StationaryVehicleMoved')
        if phase=='start' and (speed>=1.7 or self.current['speed']<-.2):raise RuntimeError('IndependentSpeedGuard')
        return self.current
    def register(self,frame):
        args=dict(frame=frame)
        if self.injection and self.scenario=='missing_camera' and not self.injection.get('armed'):
            args['injection']=dict(identity=list(self.channel.identity),camera='CAM_FRONT',actor_id=self.sensor_actors['CAM_FRONT'],frame=frame)
            self.injection['armed']=True;self.injection['key']=args['injection']
            self.event('injection_registered',injection=self.injection)
        self.call('register',args,timeout=.1);self.target=frame
    def capture(self,phase,first=False):
        # Target is registered before issuing its tick, including every F+4.
        if first:
            self.register(self.current['frame']+1)
            self.phase('BoundedCapture',time.monotonic()+.5)
            self.tick(phase)
        else:
            if self.state['phase']!='FiniteExecution':raise RuntimeError('ExecutionPhaseNotEntered')
            self.tick(phase)
            self.register(self.current['frame']+1)
            self.phase('BoundedCapture',time.monotonic()+.5)
            self.tick(phase)
        capture_start=self.current['input_wall_s']
        self.state['phase_deadline']=min(self.state['phase_deadline'],capture_start+.5)
        self.tick(phase);self.tick(phase)
        # The collection helper keeps the original target deadline itself.
        r=self.call('collect',timeout=max(.001,capture_start+.5-time.monotonic()))
        self.observation=r['observation'];self.current=r['frozen']
        self.event('input_validated',target=self.target,capture_wall_s=r['capture_wall_s'],sim_age_s=self.current['sim_s']-self.observation['sim_s'])
        return r
    def compute(self,control,blocked=False,inject=False):
        self.phase('ConfirmedFrozenWorld',time.monotonic()+.2)
        self.issue('PlannedHoldBrake',STOP)
        # No tick may run under this source. Real snapshot must remain unchanged.
        self.call('frozen',dict(expected=self.current),timeout=.1)
        requested=time.monotonic();task=self.channel.request(self.observation,self.current,requested,'blocked' if blocked else 'normal',control)
        if inject and self.scenario=='invalid_output':task['injection']='invalid_output'
        self.task=task;self.state['task']={k:task[k] for k in ('sequence','frame','deadline_wall_s','requested_wall_s','mode')}
        self.phase('ExpectedComputeWait',task['deadline_wall_s'])
        atomic(self.root/'tasks'/('%06d.json'%task['sequence']),task)
        self.event('task_requested',task=self.state['task'],frozen_frame=self.current['frame'],speed=self.current['speed_norm'])
        if inject and self.scenario=='manual_emergency_stop':
            atomic(self.root/'manual_request.json',dict(identity=list(self.channel.identity),task=self.state['task'],forward_speed_mps=self.current['speed']))
        while True:
            self.pump()
            if self.terminal:
                if blocked and self.terminal['reason']=='TaskDeadline':
                    self.result['task_timeout']=dict(task=self.state['task'],fault=self.terminal,worker_heartbeat=load(self.root/'test_worker_heartbeat.json'),speed=self.current['speed_norm'],frame=self.current['frame'])
                    (self.root/'release_late_result').touch()
                    return None
                raise RuntimeError(self.terminal['reason'])
            response=load(self.root/'task_results'/('%06d.json'%task['sequence']))
            if response:
                self.phase('ValidateOutput',time.monotonic()+.1)
                confirmed=self.call('frozen',dict(expected=self.current),timeout=.1)['snapshot']
                accepted=self.channel.consume(response,confirmed,time.monotonic(),pending=self.pending is not None)
                self.phase('FiniteExecution',time.monotonic()+1.)
                self.issue('IndependentTestControl',accepted,self.observation)
                row=dict(task_sequence=task['sequence'],frame=task['frame'],wait_wall_s=time.monotonic()-requested,frame_unchanged=True,sim_unchanged=True,state_unchanged=True,source='IndependentTestControl',capture_wall_s=requested-self.observation['input_wall_s'])
                self.event('task_accepted',**row);return row
            # Watchdog is the primary detector; owner also has a finite allowance
            # to observe its decision rather than spin indefinitely.
            if time.monotonic()>task['deadline_wall_s']+.3:raise TimeoutError('IndependentTaskDetectorMissing')
            time.sleep(.005)
    def latch(self,reason):
        if self.terminal:return
        first_fault(self.root,reason,'owner',dict(self.state,operation=self.pending),self.current)
        self.state['fault']=reason;self.event('owner_fault_requested',reason=reason)
        until=time.monotonic()+.3
        while self.terminal is None and time.monotonic()<until:self.pump();time.sleep(.005)
        if self.terminal is None:
            self.guard.latch();self.terminal=dict(reason=reason,wall_s=time.monotonic(),independent_detection=False)
            if self.channel:self.channel.locked=True
    def stop_observe(self):
        self.pump()
        if self.server_dead:
            self.result['stop']=dict(status='ResponseUnverified_ServerExited',ticks=0,hold_sim_s=0.,physical_stop_verified=False)
            return
        if not self.terminal:return
        self.state['fault']=self.terminal['reason'];self.phase('FaultLocked',self.terminal['wall_s']+10.)
        until=self.terminal['wall_s']+10.;ack=None
        while time.monotonic()<min(until,self.terminal['wall_s']+.5):
            self.pump();ack=load(self.root/'fault_brake.json')
            if ack:break
            time.sleep(.005)
        if not ack or not ack.get('server_processed') or ack.get('handoff_uncertain'):
            self.result['stop']=dict(status='ResponseUnverified',brake=ack);return
        if self.pending or self.guard.uncertain:
            self.result['stop']=dict(status='HandoffUncertain',brake=ack);return
        # Confirmed ordinary operations are drained. Independent brake response is
        # processing evidence; physics evidence begins only at the next tick.
        self.guard.pending=None;self.guard.active=dict(sequence=-1,epoch=self.guard.epoch,source='FaultBrake',control=STOP,observation=None,request_wall_s=ack['submission_wall_s'])
        before=self.current;stopped=None;first=None;rows=[]
        while time.monotonic()<until and self.counts['fault_stop']<120:
            self.pump();m=self.tick('fault_stop');rows.append(m)
            if first is None:first=m
            if m['speed_norm']<=.1:
                if stopped is None:stopped=m
            else:stopped=None
            self.check_late()
            if stopped and m['sim_s']-stopped['sim_s']>=1.-1e-6:break
        hold=0 if stopped is None or not rows else rows[-1]['sim_s']-stopped['sim_s']
        self.result['stop']=dict(status='PhysicalStopObserved' if hold>=1.-1e-6 else 'ResponseObserved_HoldIncomplete',brake=ack,hold_sim_s=hold,
            speed_before_mps=None if before is None else before['speed_norm'],first_feedback_speed_mps=None if first is None else first['speed_norm'],
            terminal_speed_mps=None if not rows else rows[-1]['speed_norm'],stop_sim_s=None if stopped is None or before is None else stopped['sim_s']-before['sim_s'],
            stop_distance_m=None if stopped is None or before is None else math.dist(stopped['position'],before['position']),ticks=len(rows),
            position_scope='Last confirmed frozen snapshot to first stopped snapshot; timeout wall pause excluded',all_feedback_matched=all(x['feedback_matches'] for x in rows))
    def check_late(self):
        if self.task is None or self.result.get('late_result'):return
        response=load(self.root/'task_results'/('%06d.json'%self.task['sequence']))
        if response and (response.get('late') or (self.scenario=='manual_emergency_stop' and self.terminal)):
            try:self.channel.consume(response,self.current,time.monotonic())
            except ValueError as exc:
                self.result['late_result']=dict(rejected=True,reason=str(exc));self.event('late_result_rejected',reason=str(exc));return
            raise RuntimeError('LateResultAccepted')
    def reconcile_for_cleanup(self, timeout=.2):
        """Retire a returned IO call after deadline only for teardown, never ticks.

        A late reply cannot restore command validity or prove a timed-out tick
        did not run. Keep the first fault and guard uncertainty unchanged.
        """
        if self.pending is None:return True
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            reply=load(self.root/'io_responses'/('%06d.json'%self.pending['sequence']))
            if reply is not None:
                pending=self.pending;self.pending=None
                if pending['kind']=='tick':self.advance_unknown=True;self.guard.uncertain=True
                if pending['kind']=='control':self.mark_handoff_uncertain(pending)
                self.event('late_io_retired_for_cleanup_only',operation=pending,success=reply.get('success'),error=reply.get('error'),call_end_wall_s=reply.get('call_end_wall_s'))
                self.result['late_io_cleanup_reconciliation']=dict(kind=pending['kind'],success=reply.get('success'),physics_resumption_allowed=False,server_advance_unknown=self.advance_unknown or pending['kind']=='tick')
                return True
            self.pump();time.sleep(.005)
        return False
    def run(self):
        try:
            cpus=config()['process_cpus'];self.start('io',cpus['capture']);self.ready(self.root/'io_ready.json')
            self.phase('Initialization',time.monotonic()+30.)
            initialization_deadline=self.state['phase_deadline']
            setup=self.call('setup',timeout=30.)
            if setup['geometry']['status']!='Passed':raise RuntimeError('NavigationGeometryRejected')
            self.state.update(ego_id=setup['ego_id']);self.sensor_actors=setup['sensor_actors'] if self.scenario else setup.get('sensor_actors')
            self.channel=TaskChannel(setup['identity']);self.state['identity']=setup['identity'];self.start('watchdog',cpus['watchdog']);self.ready(self.root/'logic_watchdog_ready.json')
            self.start('test',cpus['model'],isolated=True);ready=self.ready(self.root/'test_worker_ready.json')
            if ready['network_namespace']==os.readlink('/proc/self/ns/net'):raise RuntimeError('TestWorkerNotIsolated')
            if self.scenario=='manual_emergency_stop':
                self.start('manual',cpus['watchdog']);self.ready(self.root/'manual_ready.json')
            self.state['armed']=True;self.phase('Initialization',initialization_deadline)
            self.issue('InitialProtectionBrake',STOP)
            stable=0
            for _ in range(60):
                m=self.tick('initialization');stable=stable+1 if m['speed_norm']<=.1 else 0
                if stable>=21:break
            if stable<21:raise RuntimeError('InitializationNotStable')
            self.result['initialization']=dict(ticks=self.counts['initialization'],stable_ticks=stable)
            for cycle in range(6):
                self.capture('stationary',first=cycle==0)
                self.result['stationary_cycles'].append(self.compute(STOP))
            self.result['stage_A']='Passed_SixFixedCycles'
            # No reset/reload: the last stationary result carries the next target.
            while self.counts['start']<200:
                if 200-self.counts['start']<4:raise RuntimeError('PreconditionsNotMet')
                self.capture('start');self.started_motion=True
                control=dict(STOP,**self.cfg['test_control'])
                eligible=1.<=self.current['speed']<1.7 and self.current['speed_norm']<1.7 if self.scenario else 1.<=self.current['speed_norm']<2.
                if eligible:
                    if self.scenario:
                        if self.injection is not None:raise RuntimeError('RepeatedInjection')
                        self.injection=dict(scenario=self.scenario,boundary=self.current,wall_s=time.monotonic(),armed=False)
                        self.result['injection']=self.injection
                        self.event('injection_precondition',scenario=self.scenario,forward_speed_mps=self.current['speed'],speed_norm=self.current['speed_norm'],frame=self.current['frame'])
                        if self.scenario=='missing_camera':
                            # Fresh finite command required before next target's two flush ticks.
                            self.result['moving_cycles'].append(self.compute(control))
                            self.capture('start')
                            raise RuntimeError('MissingCameraInjectionNotRejected')
                        self.compute(control,inject=True)
                        raise RuntimeError('InjectedOutputNotRejected')
                    self.compute(control,blocked=True);break
                self.result['moving_cycles'].append(self.compute(control))
            if self.terminal is None:raise RuntimeError('PreconditionsNotMet')
        except BaseException as exc:
            self.result.update(error=repr(exc),traceback=traceback.format_exc())
            try:self.latch(str(exc))
            except Exception as secondary:self.result['latch_error']=repr(secondary)
        finally:
            try:
                if self.state.get('ego_id') and self.terminal:self.stop_observe()
            except BaseException as exc:self.result['stop_error']=repr(exc)
            try:
                if self.scenario=='manual_emergency_stop' and self.task and self.terminal:
                    while time.monotonic()<self.task['deadline_wall_s'] and not self.result.get('late_result'):
                        self.pump();self.check_late();time.sleep(.005)
                self.check_late()
            except Exception as exc:self.result['late_check_error']=repr(exc)
            self.cleanup_deadline=time.monotonic()+self.cfg['wall_deadlines_s']['cleanup']
            try:self.reconcile_for_cleanup(timeout=min(.2,max(0.,self.cleanup_deadline-time.monotonic())))
            except Exception as exc:self.result['cleanup_reconcile_error']=repr(exc)
            self.state['armed']=False;atomic(self.root/'owner_state.json',self.state)
            for role,name in [('test','test_worker_stop'),('watchdog','logic_watchdog_stop'),('manual','manual_stop')]:
                (self.root/name).touch()
                proc=self.processes.get(role)
                if proc:
                    try:proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
            try:
                if self.pending is None:self.result['cleanup']=self.call('cleanup',dict(advance_unknown=self.advance_unknown),timeout=max(.001,self.cleanup_deadline-time.monotonic()),allow_fault=True).get('cleanup')
                else:self.result['cleanup_error']='IOStillInFlight_ResourceCleanupUnverified'
            except BaseException as exc:self.result['cleanup_error']=repr(exc)
            (self.root/'io_stop').touch()
            for role,proc in self.processes.items():
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
            self.result.update(fault=self.terminal,phase_ticks=dict(self.counts),charged_ticks=sum(self.counts.values())+sum(self.settings_charges.values()),settings_charges=dict(self.settings_charges),server_advance_unknown=self.advance_unknown,owner_heartbeat_max_gap_s=max(self.owner_gaps,default=0),remaining_processes=[r for r,p in self.processes.items() if p.poll() is None])
            if self.result.get('stage_A') and self.result.get('task_timeout') and self.result.get('stop',{}).get('status')=='PhysicalStopObserved' and self.result.get('late_result',{}).get('rejected') and self.result.get('cleanup',{}).get('complete',False) and not self.result['cleanup'].get('remaining') and not self.result.get('cleanup_error'):
                self.result['status']='Passed_FiniteNoModelTaskTimeoutTestOnly'
            if self.scenario:
                self.result['scenario']=self.scenario
                self.result['status']='Partial_PendingIndependentScenarioAudit'
                self.result['pid_executions']=0
            write(self.root/'result.json',self.result)


def supervise(config_path=CFG,suite_reservation=None):
    import carla
    from tools.m8j12_capture_supervisor import listening,group_members
    from tools.m8j13_supervisor import terminate
    cfg=json.loads(config_path.read_text());name=scenario(cfg)
    if name and (suite_reservation is None or suite_reservation['scenario']!=name or suite_reservation['config_sha256']!=digest(config_path)):raise ValueError('SuiteReservationRequired')
    c=config();base=Path(cfg['private_root']);base.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=cfg['python_deps']+':'+str(Path.cwd()))
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=reserve_start(load(base/'budget.json'))
        if any(listening(p) for p in range(c['port'],c['port']+3)):raise RuntimeError('PortsOccupied')
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('FreezeRequiresCleanWorktree')
        root=base/cfg.get('run_id','run');root.mkdir()
        for folder_name in ('io_requests','io_responses','tasks','task_results'):(root/folder_name).mkdir()
        write(root/'frozen_config.json',cfg)
        write(root/'request.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config_sha256=digest(config_path),config_path=str(config_path),config=cfg,
            purpose='Six fixed no-model parked cycles then one low-speed '+(name or 'blocked_task')+' stop',budget=ledger['runs'][0],stop_conditions=['First system fault','TwoTickCaptureFailure','UncertainHandoff','SpeedGuard','TaskDeadline terminal','FixedBudget']))
        atomic(base/'budget.json',ledger);server=None;owner=None;report={}
        try:
            argv=['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),cfg['server_root'],str(c['port']),'Epic']
            with (root/'server.log').open('x') as log:server=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            until=time.monotonic()+150
            while time.monotonic()<until:
                if server.poll() is not None:raise RuntimeError('ServerEarlyExit')
                client=carla.Client('127.0.0.1',c['port']);client.set_timeout(2.)
                try:
                    versions=[client.get_client_version(),client.get_server_version()]
                    if versions!=['0.9.16']*2:raise ValueError('ClientServerVersionMismatch')
                    report['client']=dict(versions=versions,path=carla.libcarla.__file__,sha256=digest(carla.libcarla.__file__),port=c['port']);break
                except RuntimeError:time.sleep(1.)
            else:raise TimeoutError('ServerReadiness')
            with (root/'owner.log').open('x') as log:owner=subprocess.Popen(['taskset','-c',c['process_cpus']['capture'],cfg['python'],'-m','tools.m8j13_logic_runtime','--owner-root',str(root)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            until=time.monotonic()+360
            while owner.poll() is None:
                if server.poll() is not None:atomic(root/'server_exited.json',dict(wall_s=time.monotonic()));raise RuntimeError('ServerExited')
                if time.monotonic()>until:raise TimeoutError('OwnerWholeLifecycleDeadline')
                time.sleep(.2)
            report['owner_exit_code']=owner.returncode;report['result']=load(root/'result.json')
        except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
        finally:
            terminate(owner);terminate(server)
            until=time.monotonic()+8
            while time.monotonic()<until and any(listening(p) for p in range(c['port'],c['port']+3)):time.sleep(.1)
            report['ports_after']={str(p):listening(p) for p in range(c['port'],c['port']+3)}
            report['groups_after']=[group_members(p.pid) for p in (server,owner) if p]
            report['cleanup_complete']=not any(report['ports_after'].values()) and not any(report['groups_after'])
            ledger['runs'][0]['charged_ticks']=(report.get('result') or {}).get('charged_ticks',800)
            atomic(base/'budget.json',ledger);write(root/'supervisor.json',report)
            print(json.dumps(dict(owner_exit_code=report.get('owner_exit_code'),error=report.get('error'),cleanup_complete=report.get('cleanup_complete'),root=str(root)),indent=2),flush=True)
        return root


def main():
    p=argparse.ArgumentParser();p.add_argument('--owner-root',type=Path);p.add_argument('--config',type=Path,default=CFG);a=p.parse_args();os.umask(0o077)
    if a.owner_root:Runtime(a.owner_root,load(a.owner_root/'frozen_config.json')).run()
    else:supervise(a.config)

if __name__=='__main__':main()
