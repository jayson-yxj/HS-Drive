"""One explicitly authorized no-model initialization diagnostic. No worker/PID.

Reuses production I/O, watchdog, owner deadline and bounded finalization paths.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import time
import traceback
from datetime import datetime,timezone
from tools.m8j13_common import atomic,write,append,digest,config
from tools.m8j13_logic_runtime import Runtime
from tools.m8j13_logic_shadow import ShadowRuntime
from tools.m8j13_logic_shutdown import shutdown_notice,finalize_supervisor
from tools.m8j13_logic_services import load
from simulator.m8j13_logic_runtime import STOP

CFG=Path('configs/m8j13_init_diagnostic.json')


def validate(cfg):
    required=dict(mode='ExplicitNoModelInitializationDiagnostic',max_starts=1,max_active_ticks=180,
                  max_model_starts=0,max_forward_attempts=0,max_pid_calls=0,model_actuation_allowed=False,
                  training=False,initialization_tick_wall_s=2.,initialization_min_tick_interval_s=.05,
                  stable_ticks=21,physics_dt_s=.05)
    for k,v in required.items():
        if type(cfg.get(k))!=type(v) or cfg[k]!=v:raise ValueError('DiagnosticConfiguration:'+k)
    expected=dict(settings_startup=30,initialization=60,stationary=0,start=0,fault_stop=21,exit=0,settings_restore=30,failure_cleanup_reserve=39)
    if cfg['tick_caps']!=expected or sum(cfg['tick_caps'].values())!=180:raise ValueError('DiagnosticTickBudget')
    if cfg.get('fault_injection') or cfg.get('parked_handoff_revision'):raise ValueError('ForbiddenDiagnosticVariant')
    return True


def anchor():
    before=time.monotonic();utc=datetime.now(timezone.utc).isoformat();after=time.monotonic()
    return dict(monotonic_before_s=before,utc=utc,monotonic_after_s=after,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def process_identity(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=p.joinpath('stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        env=dict(x.split('=',1) for x in p.joinpath('environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
        return dict(pid=pid,ppid=int(fields[1]),pgrp=int(fields[2]),start_ticks=int(fields[19]),
                    executable=os.readlink(p/'exe'),argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),
                    cwd=os.readlink(p/'cwd'),uid=p.stat().st_uid,
                    environment={k:env.get(k) for k in ['HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','DISPLAY','VK_ICD_FILENAMES']})
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def capture_processes(root,server,seen):
    from tools.m8j12_capture_supervisor import group_members
    for row in group_members(server.pid):
        rec=process_identity(row['pid'])
        if rec and (rec['pid'],rec['start_ticks']) not in seen:
            seen.add((rec['pid'],rec['start_ticks']));append(root/'process_identity.jsonl',dict(anchor=anchor(),**rec))


def crash_inventory(cfg):
    home=Path(pwd.getpwnam('ubuntu').pw_dir)
    roots=[home/'.config/Epic/CarlaUE4/Saved',home/'.local/share/CarlaUE4/Saved',Path(cfg['server_root'])/'CarlaUE4/Saved',Path('/var/crash'),Path('/var/lib/systemd/coredump')]
    files={}
    for base in roots:
        if not base.exists():continue
        for p in base.rglob('*'):
            if not p.is_file() or not (p.suffix in ('.log','.dmp','.mdmp','.core') or p.name in ('Diagnostics.txt','CrashContext.runtime-xml') or p.name.startswith('core.')):continue
            if len(files)>=1000:raise RuntimeError('CrashInventoryCapacity')
            stat=p.stat();files[str(p)]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest(p) if stat.st_size<=16*1024*1024 else None)
    return dict(home=str(home),roots=[str(p) for p in roots],files=files)


def preserve_crashes(root,cfg,before):
    after=crash_inventory(cfg);write(root/'crash_inventory_after.json',after);changes=[];total=0
    for name,row in after['files'].items():
        if before['files'].get(name)==row:continue
        record=dict(path=name,metadata=row,association='Candidate only: check embedded PID/build/time, mtime insufficient')
        total+=row['bytes']
        if total<=128*1024*1024:
            folder=root/'crash_artifacts'/str(len(changes));folder.mkdir(parents=True)
            shutil.copyfile(name,folder/Path(name).name);record['private_copy']=str(folder/Path(name).name)
        else:record['copy_skipped']='128MiB evidence cap; original retained'
        changes.append(record)
    write(root/'crash_candidates.json',changes)
    return len(changes)


class InitializationDiagnostic(Runtime):
    finish_owner=ShadowRuntime.finish_owner
    def __init__(self,root,cfg):
        validate(cfg);super().__init__(root,cfg)
        self.state['run_deadline']=min(self.state['run_deadline'],load(root/'request.json')['runtime_deadline_wall_s'])
    def run(self):
        try:
            cpus=config()['process_cpus'];self.start('io',cpus['capture']);self.ready(self.root/'io_ready.json')
            self.phase('Initialization',time.monotonic()+30.);deadline=self.state['phase_deadline']
            setup=self.call('setup',timeout=30.)
            if setup['geometry']['status']!='Passed':raise RuntimeError('NavigationGeometryRejected')
            self.state.update(ego_id=setup['ego_id'],identity=setup['identity'])
            self.start('watchdog',cpus['watchdog']);self.ready(self.root/'logic_watchdog_ready.json')
            self.state['armed']=True;self.phase('Initialization',deadline);self.issue('InitialProtectionBrake',STOP)
            stable=0
            for _ in range(60):
                row=self.tick('initialization');stable=stable+1 if row['speed_norm']<=.1 else 0
                if stable>=21:break
            if stable<21:raise RuntimeError('InitializationNotStable')
            self.result['initialization']=dict(passed=True,ticks=self.counts['initialization'],stable_ticks=stable)
            self.result['normal_completed']=True;self.latch('NormalInitializationCompletion')
        except BaseException as exc:
            self.result.update(error=repr(exc),traceback=traceback.format_exc())
            try:self.latch(str(exc))
            except Exception as secondary:self.result['latch_error']=repr(secondary)
        finally:
            self.finish_owner()


def supervise(path):
    import carla
    from tools.m8j12_capture_supervisor import listening,group_members
    cfg=load(path);validate(cfg);c=config();base=Path(cfg['private_root']);base.mkdir(parents=True,exist_ok=True)
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (base/'budget.jsonl').open('x',buffering=1) as ledger:
            def reserve(event,**kw):
                ledger.write(json.dumps(dict(event=event,wall_s=time.monotonic(),**kw))+'\n');ledger.flush();os.fsync(ledger.fileno())
            if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('FreezeRequiresCleanWorktree')
            if any(listening(p) for p in range(c['port'],c['port']+3)):raise RuntimeError('PortsOccupied')
            root=base/cfg['run_id'];root.mkdir()
            for n in ['io_requests','io_responses']:(root/n).mkdir()
            started=time.monotonic();total_deadline=started+240.;runtime_deadline=total_deadline-38.
            write(root/'frozen_config.json',cfg);write(root/'request.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config_sha256=digest(path),start_anchor=anchor(),total_deadline_wall_s=total_deadline,runtime_deadline_wall_s=runtime_deadline,purpose='No-model one initialization; max180 charged ticks; no retry'))
            before=crash_inventory(cfg);write(root/'crash_inventory_before.json',before)
            write(root/'provenance.json',dict(client_path=carla.libcarla.__file__,client_sha256=digest(carla.libcarla.__file__),binary_sha256=digest(Path(cfg['server_root'])/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'),startup_script_sha256=digest('scripts/start_carla.sh'),boot_id=anchor()['boot_id'],process_cpus=c['process_cpus'],core_pattern=Path('/proc/sys/kernel/core_pattern').read_text().strip()))
            server=owner=None;report={};seen=set();environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=cfg['python_deps']+':'+str(Path.cwd()))
            try:
                reserve('carla_start',reserved_ticks=180,model_loads=0,forwards=0,pid=0,training=0)
                argv=['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),cfg['server_root'],str(c['port']),'Epic']
                with (root/'server.log').open('x') as log:server=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                write(root/'server_launch.json',dict(anchor=anchor(),wrapper_pid=server.pid,argv=argv))
                until=min(time.monotonic()+150.,runtime_deadline)
                while time.monotonic()<until:
                    capture_processes(root,server,seen)
                    if server.poll() is not None:shutdown_notice(root,'ServerExited',server);raise RuntimeError('ServerExitedBeforeReady')
                    client=carla.Client('127.0.0.1',c['port']);client.set_timeout(min(.25,max(.001,until-time.monotonic())))
                    try:
                        versions=[client.get_client_version(),client.get_server_version()]
                        if versions!=['0.9.16']*2:raise ValueError('VersionMismatch')
                        report['versions']=versions;break
                    except RuntimeError:time.sleep(.05)
                else:raise TimeoutError('ServerStartupDeadline')
                capture_processes(root,server,seen)
                with (root/'owner.log').open('x') as log:owner=subprocess.Popen(['taskset','-c',c['process_cpus']['capture'],cfg['python'],'-m','tools.m8j13_init_diagnostic','--owner-root',str(root)],stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                write(root/'owner_launch.json',dict(anchor=anchor(),pid=owner.pid))
                while owner.poll() is None:
                    if server.poll() is not None:shutdown_notice(root,'ServerExited',server);raise RuntimeError('ServerExited')
                    if time.monotonic()>=runtime_deadline:raise TimeoutError('DiagnosticLifecycleDeadline')
                    time.sleep(.05)
                report['owner_exit_code']=owner.returncode
            except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
            finally:
                finalize_supervisor(root,cfg,dict(owner=owner,server=server),report,reserve,group_members,listening,range(c['port'],c['port']+3))
                report['end_anchor']=anchor();report['whole_lifecycle_within240s']=time.monotonic()<=total_deadline
                report['model']='NotStarted';report['model_loads']=report['forwards']=report['pid']=report['model_controls']=report['training']=0
                try:report['new_crash_candidates']=preserve_crashes(root,cfg,before)
                except Exception as exc:report['crash_archive_error']=repr(exc)
                atomic(root/'supervisor.json',report)
                print(json.dumps(dict(root=str(root),error=report.get('error'),cleanup=report.get('process_cleanup'))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CFG);p.add_argument('--owner-root',type=Path);a=p.parse_args();os.umask(0o077)
    if a.owner_root:InitializationDiagnostic(a.owner_root,load(a.owner_root/'frozen_config.json')).run()
    else:supervise(a.config)


def audit_run(root):
    """Independent raw-event checks, not the owner's declared status."""
    from tools.m8j13_logic_init_report import lines,distribution
    from simulator.m8j13_logic_runtime import canonical
    from tools.m8j13_logic_shadow_report import termination_evidence
    result=load(root/'result.json') or {};sup=load(root/'supervisor.json') or {};clean=load(root/'cleanup_steps.json') or {}
    snaps=lines(root,'snapshots.jsonl');ops=lines(root,'io_consumption.jsonl');ticks=lines(root,'tick_requests.jsonl');ledger=lines(root.parent,'budget.jsonl')
    requests=[load(p) for p in sorted((root/'io_requests').glob('*.json'))];init=[s for s in snaps if s['phase']=='initialization']
    fault=load(root/'first_fault.json') or load(root/'watchdog_fault.json') or {};brake=load(root/'fault_brake.json') or {}
    uncertain=result.get('server_advance_unknown',False) or result.get('handoff_uncertain',False) or result.get('inflight_operation')
    terminal=termination_evidence(snaps,fault,brake,uncertain)
    terminal['normal']=fault.get('reason')=='NormalInitializationCompletion'
    known=[s for s in init[-21:] if s['speed_norm']<=.1 and s['feedback_matches'] and canonical(s['full_control'])==canonical(STOP)]
    consumed={o['operation']['sequence'] for o in ops};missing=[q['sequence'] for q in requests if q['sequence'] not in consumed]
    ontime=all(o['reply']['success'] and o['owner_consumed_wall_s']<=o['operation']['deadline'] for o in ops)
    control=[q for q in requests if q['kind'] in ('control','tick')]
    isolation=all(canonical(q['args']['command']['control'])==canonical(STOP) and q['args']['command']['source'] in ('InitialProtectionBrake','FaultBrake') for q in control)
    resource=(len(clean.get('sensor_stops',[]))==8 and all(x['passed'] for x in clean['sensor_stops']) and len(clean.get('actor_destruction',[]))==9 and all(x['destroyed'] for x in clean['actor_destruction']) and clean.get('settings_restored') is True and clean.get('remaining')==[])
    charged=next((r.get('charged_ticks') for r in reversed(ledger) if r['event']=='finished'),None)
    starts=sum(r['event']=='carla_start' for r in ledger)
    durations=[(o['owner_consumed_wall_s']-o['operation']['started'])*1000 for o in ops if o['operation']['kind']=='tick']
    timing=dict(tick_operation_ms=distribution(durations),initialization_exceeded_old500ms=sum(x>500 for x in durations[:len(init)]))
    passed=(len(known)==21 and ontime and not missing and isolation and resource and sup.get('process_cleanup') and terminal['terminal_hold_verified'] and fault.get('reason')=='NormalInitializationCompletion' and starts==1 and charged is not None and charged<=180 and not lines(root,'collisions.jsonl'))
    return dict(status='Passed_SingleNoModelInitialization' if passed else 'Partial',J13_A='Partial',run_commit=(load(root/'request.json') or {}).get('commit'),run_id=root.name,
                initialization=dict(confirmed_ticks=len(init),last21_stable=len(known)==21),terminal_hold=terminal,timing=timing,
                first_fault=fault.get('reason'),unconsumed_operations=missing,all_operations_timely=ontime,
                isolation=dict(brake_only=isolation,model_loads=0,forwards=0,pid=0,model_controls=0,training=0),
                cleanup=dict(resources_verified=resource,sensor_stops=sum(x.get('passed',False) for x in clean.get('sensor_stops',[])),actors_destroyed=sum(x.get('destroyed',False) for x in clean.get('actor_destruction',[])),settings_restored=clean.get('settings_restored'),remaining=clean.get('remaining'),process_groups_ports=sup.get('process_cleanup'),owner_final_present=bool(result)),
                budget=dict(starts=starts,max_starts=1,tick_attempts=len(ticks),confirmed_snapshots=len(snaps),charged_ticks=charged,max_ticks=180,phase_ticks=result.get('phase_ticks'),settings_charges=result.get('settings_charges')),
                whole_lifecycle_within240s=sup.get('whole_lifecycle_within240s'),crash_candidates=sup.get('new_crash_candidates'),S0_review_ready=False,
                limits=['One run cannot establish crash cause or reliability','No resident model; cannot isolate model-load causality','No500ms planned handoff or24target evidence','No moving driving proof'])


if __name__=='__main__':main()
