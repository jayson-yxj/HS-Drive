"""Bounded shutdown for the parked logical runner; no simulator/model imports.

Only the supervisor's own process groups are signalled. Runtime deadlines are
unchanged. These helpers also run under fake clocks/processes in contract tests.
"""
import json
import os
import signal
import tempfile
import time
from tools.m8j13_common import atomic


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def publish_once(path,value):
    if path.exists():return read(path)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(path.parent))
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
        try:os.link(name,path)
        except FileExistsError:pass
    finally:os.unlink(name)
    return read(path)


def signal_model_stop(root,now):
    return publish_once(root/'model_stop',dict(wall_s=now,deadline_wall_s=now+30.))


def first_fault(root, reason, source, state=None, snapshot=None, now=None, **extra):
    """Publish once, atomically even when owner and watchdog race."""
    path=root/'first_fault.json'
    if path.exists():return read(path)
    state=state or {};op=state.get('operation')
    value=dict(reason=reason,source=source,wall_s=time.monotonic() if now is None else now,
               clock='monotonic',phase=state.get('phase'),active_target=state.get('active_target'),
               task=state.get('task'),operation=op,last_confirmed_snapshot=snapshot,
               failed_operation=state.get('failed_operation'),epoch=state.get('epoch',0)+1,
               handoff_uncertain=bool(state.get('handoff_uncertain') or (op and op['kind']=='control')),
               server_advance_unknown=bool(state.get('server_advance_unknown') or (op and op['kind']=='tick')),**extra)
    # A unique temp inode + exclusive link cannot expose a partial first record
    # or let a later writer replace the winner. No lock is held over an RPC.
    return publish_once(path,value)


def last_snapshot(root):
    path=root/'snapshots.jsonl'
    if not path.exists():return None
    # Bounded runs contain at most800 records. A torn last line is not a snapshot.
    for line in reversed(path.read_text().splitlines()):
        try:return json.loads(line)
        except ValueError:continue
    return None


def shutdown_notice(root, reason, server=None, clock=time.monotonic):
    existing=read(root/'shutdown_request.json')
    if existing:return existing
    now=clock();state=read(root/'owner_state.json') or {}
    dead=server is not None and server.poll() is not None
    fault=first_fault(root,reason,'supervisor',state,last_snapshot(root),now)
    model_stop=signal_model_stop(root,now)
    notice=dict(reason=reason,first_fault=fault,started_wall_s=now,clock='monotonic',
                owner_deadline_wall_s=now+15.,model_deadline_wall_s=min(now+30.,model_stop['deadline_wall_s']),
                final_deadline_wall_s=now+38.,server_confirmed_exited=dead)
    if dead:
        atomic(root/'server_exited.json',dict(wall_s=now,confirmed=True,wrapper_pid=server.pid,
               wrapper_returncode=server.returncode,engine_signal=None,
               engine_signal_source='Not inferred from wrapper returncode'))
    atomic(root/'shutdown_request.json',notice)
    return notice


def finish_processes(root, processes, notice, group_members, *, clock=time.monotonic,
                     sleep=time.sleep, send_group=os.killpg):
    """Concurrent grace/escalation within ONE shared final deadline.

    Owner grace ends at t+15 (resource deadline); model at t+30. TERM only
    follows the applicable grace deadline, KILL2s later. All escalation and
    port/group checks share the former8s termination allowance at t+38;
    it is not renewed per process/signal. Server stays for owner cleanup.
    """
    begin=notice['started_wall_s'];end=notice['final_deadline_wall_s'];rows={};errors=[]
    def fail(role,step,exc):errors.append(dict(role=role,step=step,error=repr(exc),wall_s=clock()))
    for role,p in processes.items():
        if p is None:continue
        rows[role]=dict(pid=p.pid,deadline_wall_s=notice['model_deadline_wall_s'] if role=='model' else notice['owner_deadline_wall_s'],signals=[],returncode=None,remaining_group=None)
    while True:
        now=clock();owner=processes.get('owner');owner_done=owner is None
        if owner:
            try:owner_done=owner.poll() is not None
            except Exception as exc:fail('owner','poll',exc)
        complete=True
        for role,row in rows.items():
            p=processes[role]
            try:row['returncode']=p.poll()
            except Exception as exc:fail(role,'poll',exc)
            try:members=group_members(p.pid);row['remaining_group']=members
            except Exception as exc:fail(role,'group_members',exc);members=None
            live=row['returncode'] is None or bool(members) or members is None
            if not live:continue
            complete=False;deadline=row['deadline_wall_s']
            # Server has no application flush file; keep it for live-owner cleanup.
            term_due=(owner_done or now>=notice['owner_deadline_wall_s']) if role=='server' else now>=deadline
            term=next((s['wall_s'] for s in row['signals'] if s['signal']==int(signal.SIGTERM)),None)
            kill_due=term is not None and now>=term+2.
            sig=signal.SIGKILL if kill_due else signal.SIGTERM if term_due else None
            if sig is not None and int(sig) not in [s['signal'] for s in row['signals']]:
                row['signals'].append(dict(signal=int(sig),wall_s=now))
                try:send_group(p.pid,sig)
                except ProcessLookupError:pass
                except Exception as exc:fail(role,'signal',exc)
        if complete or now>=end:break
        sleep(min(.02,max(0.,end-clock())))
    for role,row in rows.items():
        row['normal_exit']=row['returncode']==0 and not row['signals']
        row['forced_exit']=bool(row['signals'])
    return dict(started_wall_s=begin,finished_wall_s=clock(),absolute_deadline_wall_s=end,
                processes=rows,errors=errors,groups_clear=all(r['remaining_group']==[] for r in rows.values()))


def final_model_record(root, counts, model, baseline, state_digest, memory, clock=time.monotonic):
    """Worker-owned counters survive a failed final digest or memory read."""
    result=dict(counts,model_controls=0,training=0,optimizer_steps=0,wall_s=clock(),
                fixed_state_unchanged=None,final_memory=None,finalization_errors=[])
    if model is not None:
        try:
            if baseline is not None:result['fixed_state_unchanged']=state_digest(model)==baseline
        except Exception as exc:result['finalization_errors'].append(dict(step='state_digest',error=repr(exc)))
        try:result['final_memory']=memory(model)
        except Exception as exc:result['finalization_errors'].append(dict(step='memory',error=repr(exc)))
    atomic(root/'model_final.json',result)
    return result


def finalize_supervisor(root,cfg,processes,report,reserve,group_members,listening,ports,
                        *,clock=time.monotonic,sleep=time.sleep,send_group=os.killpg):
    """Real supervisor finally path; failure in one step cannot skip the others."""
    errors=[]
    def attempt(step,fn):
        try:return fn()
        except BaseException as exc:
            errors.append(dict(step=step,error=repr(exc),wall_s=clock()));return None
    notice=attempt('stop_notice',lambda:shutdown_notice(root,report.get('error') or 'NormalSupervisorShutdown',processes.get('server'),clock))
    if notice is None:
        now=clock();notice=dict(started_wall_s=now,owner_deadline_wall_s=now+15,model_deadline_wall_s=now+30,final_deadline_wall_s=now+38)
    summary=attempt('process_finalization',lambda:finish_processes(root,processes,notice,group_members,clock=clock,sleep=sleep,send_group=send_group))
    report['shutdown']=summary;report['shutdown_notice']=notice
    for role,name in (('owner','result.json'),('model','model_final.json')):
        if role=='model' and cfg.get('max_model_starts')==0 and role not in processes:
            report['model_status']='NotStarted';continue
        value=attempt('read_'+name,lambda name=name:read(root/name))
        report['result' if role=='owner' else 'model_final']=value
        report[role+'_exit_code']=(summary or {}).get('processes',{}).get(role,{}).get('returncode')
        if value is None:
            report.setdefault('missing_child_evidence',{})[role]=dict(file=name,reason='NotPublishedBeforeSharedShutdownDeadline',
                last_state=attempt('last_owner_state',lambda:read(root/'owner_state.json')) if role=='owner' else attempt('last_worker_state',lambda:read(root/'model_progress.json')),
                source='supervisor fallback only; no child final file fabricated')
    report['ports_after']={str(p):attempt('port_'+str(p),lambda p=p:listening(p)) for p in ports}
    report['groups_after']=[]
    for role,proc in processes.items():
        if proc:report['groups_after'].append(attempt(role+'_group',lambda proc=proc:group_members(proc.pid)))
    report['process_cleanup']=(summary is not None and all(r['returncode'] is not None for r in summary['processes'].values())
        and all(v is False for v in report['ports_after'].values()) and all(v==[] for v in report['groups_after']))
    report['finalization_errors']=errors
    result=report.get('result') or {}
    # An unknown submitted tick remains reserved even when owner wrote its
    # enumerated count; this cannot release a previous conservative charge.
    uncertain=result.get('server_advance_unknown') or result.get('inflight_operation') or not result
    charge=cfg['max_active_ticks'] if processes.get('server') and uncertain else result.get('charged_ticks',0)
    report['budget_basis']=dict(charged_ticks=charge,enumerated_ticks=result.get('charged_ticks'),conservative_reservation=bool(uncertain and processes.get('server')))
    report['transport']='Immutable private evidence; missing child files remain missing'
    attempt('supervisor_publish',lambda:atomic(root/'supervisor.json',report))
    attempt('ledger_finished',lambda:reserve('finished',charged_ticks=charge,model_final=report.get('model_final'),process_cleanup=report['process_cleanup']))
    if errors:attempt('finalization_errors',lambda:atomic(root/'supervisor_finalization_errors.json',dict(errors=errors)))
    return report
