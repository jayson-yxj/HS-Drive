"""Explicit one-start parked logical shadow; immutable diagnostic-only IPC."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from tools.m8j13_logic_runtime import Runtime
from tools.m8j13_logic_services import load
from tools.m8j13_common import config,atomic,write,append,digest,assets
from simulator.m8j13_logic_runtime import STOP
from simulator.m8j13_logic_shadow import PROTOCOL,ShadowChannel,require_notification
from simulator.m8j13_logic_shadow_budget import validate_shadow_budget
from simulator.m8j13_parked_handoff import enabled as parked_handoff_enabled,handoff_proof
from simulator.m8j13_safety import PlanStamp,freshness

CFG=Path('configs/m8j13_logic_shadow.json')


class ShadowRuntime(Runtime):
    def __init__(self,root,cfg):
        super().__init__(root,cfg)
        if cfg.get('shadow_protocol')!=PROTOCOL:raise ValueError('ExplicitShadowOnly')
        validate_shadow_budget(cfg)
        self.result.update(completions=[],model_forwards=None)
    def phase(self,name,deadline):
        started=None
        if parked_handoff_enabled(self.cfg) and name==self.state['phase'] and name in ('ConfirmedFrozenWorld','FiniteExecution'):
            deadline=min(deadline,self.state['phase_deadline']);started=self.state['phase_started']
        super().phase(name,deadline)
        if started is not None:self.state['phase_started']=started;self.pump()
    def pump(self):
        super().pump()
        if self.terminal is None and self.state['armed'] and self.state['phase'] not in ('FaultLocked','Cleanup') and not self.state.get('fault'):
            err=load(self.root/'model_error.json')
            if err:raise RuntimeError('ShadowWorker:'+err['error'])
            if (self.root/'model_exited.json').exists():raise RuntimeError('ShadowWorkerExited')
    def planned_issue(self,source):
        proof=handoff_proof(self.cfg,source,STOP,self.current,self.state,pending=self.pending,guard_pending=self.guard.pending,uncertain=self.guard.uncertain,locked=self.guard.locked,active=self.guard.active)
        cmd=self.guard.prepare(source,STOP,time.monotonic(),self.observation)
        try:
            r=self.call('control',dict(command=cmd,parked_handoff=proof),timeout=.5)
            self.guard.confirm(cmd,r.get('server_processed',False))
            self.event('planned_handoff_confirmed',command_sequence=cmd['sequence'],source=source,operation_sequence=self.io_sequence,phase_started=proof['phase_started'],phase_deadline=proof['phase_deadline'])
            returned=time.monotonic();operation=r['operation']
            append(self.root/'planned_handoff_returns.jsonl',dict(operation_sequence=operation['sequence'],source=source,phase_started=proof['phase_started'],phase_deadline=proof['phase_deadline'],owner_return_wall_s=returned,passed=returned<=operation['deadline']))
            if returned>operation['deadline']:
                self.mark_handoff_uncertain(operation)
                raise TimeoutError('PlannedHandoffDeadlineAtReturn')
        except BaseException:
            self.guard.uncertain=True
            if self.failed_operation is None:
                self.failed_operation=dict(sequence=self.io_sequence,kind='control',source=source)
            self.pump();raise
    def compute_shadow(self,capture):
        revision=parked_handoff_enabled(self.cfg)
        if revision:
            self.state['task']=None;self.task=None
            self.state['active_target']=dict(frame=self.observation['frame'],identity=list(self.channel.identity),next_sequence=self.channel.sequence+1)
        phase_start=time.monotonic()
        self.phase('ConfirmedFrozenWorld',phase_start+(.5 if revision else .2))
        if revision:
            # collect has checked target+2 world, now establish a fresh frozen proof
            # before asking for either parked-only exception. Same .5s phase.
            current=self.call('frozen',dict(expected=self.current,observation=self.observation),timeout=.5)['snapshot']
            self.current.update(current);self.state['frozen_confirmation']=current
            self.planned_issue('PlannedHoldBrake')
        else:self.issue('PlannedHoldBrake',STOP)
        current=self.call('frozen',dict(expected=self.current,**({'observation':self.observation} if revision else {})),timeout=.5 if revision else .1)['snapshot']
        frozen_confirmed=time.monotonic();self.current.update(current)
        if revision:
            if frozen_confirmed>self.state['phase_deadline']:raise TimeoutError('ConfirmedFrozenPhaseDeadline')
            self.state['frozen_confirmation']=current
        requested=time.monotonic();task=self.channel.request(self.observation,self.current,requested,self.state['run_deadline'])
        self.task=task;self.state['task']={k:task[k] for k in ('sequence','frame','deadline_wall_s','requested_wall_s')}
        # Publish within the original ConfirmedFrozenWorld budget on revised path.
        frozen_phase_deadline=self.state['phase_deadline']
        if not revision:self.phase('ExpectedComputeWait',task['deadline_wall_s'])
        name='%06d.json'%task['sequence'];atomic(self.root/'shadow_tasks'/name,dict(task=task,row=capture['shadow_capture']['row']))
        submit_end=time.monotonic()
        if revision:
            if submit_end>frozen_phase_deadline:raise TimeoutError('ConfirmedFrozenPublicationDeadline')
            self.phase('ExpectedComputeWait',task['deadline_wall_s'])
        self.event('shadow_task_submitted',task=task,frozen_confirmed_wall_s=frozen_confirmed,submit_end_wall_s=submit_end,confirmed_phase_deadline=frozen_phase_deadline)
        while True:
            self.pump()
            if self.terminal:raise RuntimeError(self.terminal['reason'])
            reply=load(self.root/'shadow_completions'/name)
            if reply:
                received=time.monotonic();require_notification(reply)
                # Worker forward completion is the single validation start; no new .1s here.
                self.phase('ValidateOutput',reply['validation_deadline'])
                confirmed=self.call('frozen',dict(expected=self.current),timeout=.1)['snapshot']
                reference=time.monotonic();value=self.channel.consume(reply,confirmed,reference,pending=self.pending is not None)
                returned=time.monotonic();self.channel.finalize_return(reply,returned)
                stamp=PlanStamp(self.channel.identity[0],task['frame'],task['input_sim_s'],task['input_wall_s'],reply['times']['forward_start'],reply['times']['checks_done'],'frozen')
                old=freshness(stamp,self.channel.identity[0],confirmed['frame'],confirmed['sim_s'],returned)
                record=dict(sequence=task['sequence'],frame=task['frame'],input_wall_s=task['input_wall_s'],received_wall_s=received,consume_reference_wall_s=reference,consume_return_wall_s=returned,sim_age_s=value['sim_age_s'],old_realtime_reason=old,task_deadline=task['deadline_wall_s'],validation_deadline=reply['validation_deadline'],frozen_confirmed=True,model_controls=0)
                append(self.root/'shadow_consumed.jsonl',record);self.result['completions'].append(record)
                self.phase('FiniteExecution',time.monotonic()+1.)
                # Completion only allows continuing the fixed parked experiment.
                if revision:
                    self.current.update(confirmed);self.state['frozen_confirmation']=confirmed
                    self.planned_issue('InitialProtectionBrake')
                else:self.issue('InitialProtectionBrake',STOP,self.observation)
                return
            if time.monotonic()>task['deadline_wall_s']+.3:raise TimeoutError('IndependentTaskDetectorMissing')
            time.sleep(.002)
    def run(self):
        try:
            cpus=config()['process_cpus'];self.start('io',cpus['capture']);self.ready(self.root/'io_ready.json')
            self.phase('Initialization',time.monotonic()+30.);init_deadline=self.state['phase_deadline']
            setup=self.call('setup',timeout=30.)
            if setup['geometry']['status']!='Passed':raise RuntimeError('NavigationGeometryRejected')
            self.state.update(ego_id=setup['ego_id'],identity=setup['identity']);self.channel=ShadowChannel(setup['identity'])
            self.start('watchdog',cpus['watchdog']);self.ready(self.root/'logic_watchdog_ready.json')
            self.state['armed']=True;self.phase('Initialization',init_deadline);self.issue('InitialProtectionBrake',STOP)
            stable=0
            for _ in range(60):
                m=self.tick('initialization');stable=stable+1 if m['speed_norm']<=.1 else 0
                if stable>=21:break
            if stable<21:raise RuntimeError('InitializationNotStable')
            self.result['initialization']=dict(ticks=self.counts['initialization'],stable_ticks=stable)
            origin=self.current['frame']+1
            write(self.root/'target_schedule.json',dict(first_frame=origin,frames=[origin+4*i for i in range(24)],planned_ticks=95))
            for i in range(24):
                if i>=self.cfg['live_pid_remaining']:raise RuntimeError('PIDBudgetReservedByPreflight')
                if parked_handoff_enabled(self.cfg):
                    self.state['task']=None;self.task=None;self.state['active_target']=dict(frame=origin+4*i,identity=list(self.channel.identity),next_sequence=i+1)
                capture=self.capture('stationary',first=i==0)
                if self.observation['frame']!=origin+4*i:raise RuntimeError('FixedTargetMismatch')
                self.compute_shadow(capture)
            self.result['normal_completed']=True;self.latch('NormalShadowCompletion')
        except BaseException as exc:
            self.result.update(error=repr(exc),traceback=traceback.format_exc())
            try:
                # Prevent a worker error from preventing fault publication/polling.
                self.latch(str(exc))
            except Exception as secondary:self.result['latch_error']=repr(secondary)
        finally:
            try:
                if self.state.get('ego_id') and self.terminal:self.stop_observe()
            except BaseException as exc:self.result['stop_error']=repr(exc)
            (self.root/'model_stop').touch()
            self.cleanup_deadline=time.monotonic()+15.
            try:self.reconcile_for_cleanup(timeout=.2)
            except Exception as exc:self.result['cleanup_reconcile_error']=repr(exc)
            self.state['armed']=False;atomic(self.root/'owner_state.json',self.state)
            (self.root/'logic_watchdog_stop').touch()
            wd=self.processes.get('watchdog')
            if wd:
                try:wd.wait(timeout=3)
                except subprocess.TimeoutExpired:wd.kill();wd.wait(timeout=3)
            try:
                if self.pending is None:self.result['cleanup']=self.call('cleanup',dict(advance_unknown=self.advance_unknown),timeout=max(.001,self.cleanup_deadline-time.monotonic()),allow_fault=True).get('cleanup')
                else:self.result['cleanup_error']='IOStillInFlight_ResourceCleanupUnverified'
            except BaseException as exc:self.result['cleanup_error']=repr(exc)
            (self.root/'io_stop').touch()
            for proc in self.processes.values():
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
            self.result.update(status='RecordedLogicalShadow_PendingAudit',fault=self.terminal,phase_ticks=dict(self.counts),charged_ticks=sum(self.counts.values())+sum(self.settings_charges.values()),settings_charges=dict(self.settings_charges),server_advance_unknown=self.advance_unknown,owner_heartbeat_max_gap_s=max(self.owner_gaps,default=0),remaining_processes=[r for r,p in self.processes.items() if p.poll() is None],model_controls=0,training=0)
            write(self.root/'result.json',self.result)


def supervise(config_path=CFG):
    import carla
    from tools.m8j12_capture_supervisor import listening,group_members
    from tools.m8j13_supervisor import terminate
    cfg=load(config_path);validate_shadow_budget(cfg);c=config();base=Path(cfg['private_root']);base.mkdir(parents=True,exist_ok=True)
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (base/'budget.jsonl').open('x',buffering=1) as ledger:
            def reserve(event,**kw):ledger.write(json.dumps(dict(event=event,wall_s=time.monotonic(),**kw))+'\n');ledger.flush();os.fsync(ledger.fileno())
            if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('FreezeRequiresCleanWorktree')
            if any(listening(p) for p in range(c['port'],c['port']+3)):raise RuntimeError('PortsOccupied')
            root=base/cfg['run_id'];root.mkdir()
            for n in ('io_requests','io_responses','shadow_tasks','shadow_completions','native','diagnostics'):(root/n).mkdir()
            write(root/'frozen_config.json',cfg);write(root/'request.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config_sha256=digest(config_path),config=cfg,purpose='24 sequential parked logical model shadow tasks; no model execution'))
            model=server=owner=None;report={};env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=cfg['python_deps']+':'+str(Path.cwd()))
            try:
                write(root/'assets_before.json',assets())
                reserve('model_start',max_attempts=24,max_pid_calls=24,model_controls=0,training=0)
                with (root/'model.log').open('x') as log:model=subprocess.Popen(['taskset','-c',c['process_cpus']['model'],'bash','tools/m8j12_env.sh','-m','tools.m8j13_logic_shadow_worker','--root',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                until=time.monotonic()+120.
                while not (root/'model_ready.json').exists():
                    if model.poll() is not None or (root/'model_error.json').exists():raise RuntimeError('ModelLoadFailed')
                    if time.monotonic()>until:raise TimeoutError('ModelLoadDeadline')
                    time.sleep(.02)
                if time.monotonic()>until:raise TimeoutError('ModelLoadDeadlineAtConsumption')
                ready=load(root/'model_ready.json')
                if ready['network_namespace']==os.readlink('/proc/self/ns/net') or not all(ready[k] for k in ('parameters_frozen','eval','fp32')):raise RuntimeError('ModelIsolationOrFreezeFailed')
                reserve('carla_start',reserved_ticks=800)
                with (root/'server.log').open('x') as log:server=subprocess.Popen(['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),cfg['server_root'],str(c['port']),'Epic'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                until=time.monotonic()+150.
                while time.monotonic()<until:
                    if server.poll() is not None:raise RuntimeError('ServerEarlyExit')
                    client=carla.Client('127.0.0.1',c['port']);client.set_timeout(2.)
                    try:
                        versions=[client.get_client_version(),client.get_server_version()]
                        if versions!=['0.9.16']*2:raise ValueError('ClientServerVersionMismatch')
                        report['client']=dict(versions=versions,path=carla.libcarla.__file__,sha256=digest(carla.libcarla.__file__));break
                    except RuntimeError:time.sleep(1.)
                else:raise TimeoutError('ServerStartupDeadline')
                with (root/'owner.log').open('x') as log:owner=subprocess.Popen(['taskset','-c',c['process_cpus']['capture'],cfg['python'],'-m','tools.m8j13_logic_shadow','--owner-root',str(root)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                until=time.monotonic()+360.
                while owner.poll() is None:
                    if server.poll() is not None:atomic(root/'server_exited.json',dict(wall_s=time.monotonic()));raise RuntimeError('ServerExited')
                    if model.poll() is not None and not (root/'model_stop').exists():atomic(root/'model_exited.json',dict(wall_s=time.monotonic()))
                    if time.monotonic()>until:raise TimeoutError('OwnerLifecycleDeadline')
                    time.sleep(.05)
                report['owner_exit_code']=owner.returncode;report['result']=load(root/'result.json')
                (root/'model_stop').touch();model.wait(timeout=30.)
                report['model_final']=load(root/'model_final.json');report['model_exit_code']=model.returncode
                write(root/'assets_after.json',assets());report['assets_unchanged']=load(root/'assets_before.json')==load(root/'assets_after.json')
            except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
            finally:
                (root/'model_stop').touch()
                terminate(owner);terminate(model);terminate(server)
                until=time.monotonic()+8.
                while time.monotonic()<until and any(listening(p) for p in range(c['port'],c['port']+3)):time.sleep(.1)
                report['ports_after']={str(p):listening(p) for p in range(c['port'],c['port']+3)};report['groups_after']=[group_members(p.pid) for p in (server,owner,model) if p]
                report['process_cleanup']=not any(report['ports_after'].values()) and not any(report['groups_after'])
                report['transport']='Immutable files retained as private evidence; no shared memory or leased slots';write(root/'supervisor.json',report)
                reserve('finished',charged_ticks=(report.get('result') or {}).get('charged_ticks',800 if server else 0),model_final=load(root/'model_final.json'),process_cleanup=report['process_cleanup'])
                print(json.dumps(dict(root=str(root),error=report.get('error'),process_cleanup=report['process_cleanup'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--owner-root',type=Path);p.add_argument('--config',type=Path,default=CFG);a=p.parse_args();os.umask(0o077)
    if a.owner_root:ShadowRuntime(a.owner_root,load(a.owner_root/'frozen_config.json')).run()
    else:supervise(a.config)

if __name__=='__main__':main()
