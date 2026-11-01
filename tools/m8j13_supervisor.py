"""Own J13 CARLA, bounded child phases and only their process groups."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from tools.m8j12_capture_supervisor import listening,group_members
from tools.m8j13_common import config,write,digest,provenance,atomic


def terminate(proc):
    if proc is None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=8)
    if group_members(proc.pid):
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass


def main():
    import carla
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['safety','campaign'],required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
    os.umask(0o077);c=config();root=Path(c['private_root'])/(a.phase+'_'+a.attempt);root.mkdir(parents=True)
    if a.phase=='campaign':
        from simulator.m8j13_gate import require_a_gate
        from tools.m8j13_common import sources
        require_a_gate('reports/m8j13_a_gate.json',digest('configs/m8j13_protocol.json'),sources())
        raise RuntimeError('J13BNotImplemented: A1 did not pass; this delivery has no model actuation loop')
    ports=range(c['port'],c['port']+3)
    if any(listening(p) for p in ports):raise RuntimeError('DedicatedPortsOccupied')
    report=dict(phase=a.phase,attempt=a.attempt,provenance=provenance(),ready_attempts=[]);server=None;worker=None
    argv=['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),'/root/gpufree-data/simulators/CARLA_0.9.15',str(c['port']),'Epic']
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    report['process_cpus']=c['process_cpus'];report['cpu_quota']=Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    try:
        with (root/'server.log').open('x') as log:
            server=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
        report['server_pid']=server.pid;deadline=time.monotonic()+c['server_startup_timeout_s']
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('ServerExitedBeforeReady')
            client=carla.Client('127.0.0.1',c['port']);client.set_timeout(3)
            try:
                version=client.get_server_version();world=client.get_world();cmap=world.get_map()
                identity=dict(client=client.get_client_version(),server=version,map=cmap.name,episode=world.id,frame=world.get_snapshot().frame,
                    xodr_sha256=__import__('hashlib').sha256(cmap.to_opendrive().encode()).hexdigest())
                write(root/'initial_identity.json',identity)
                assert version==identity['client']=='0.9.15'
                break
            except RuntimeError as exc:report['ready_attempts'].append(str(exc));time.sleep(1)
        else:raise RuntimeError('ServerReadinessTimeout')
        python=str(Path(c['runtime_root'])/'capture-env/bin/python')
        module='tools.m8j13_safety_campaign' if a.phase=='safety' else 'tools.m8j13_drive'
        cmd=['taskset','-c',c['process_cpus']['capture'],python,'-m',module,'--root',str(root)]
        with (root/'worker.log').open('x') as log:
            worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
            deadline=time.monotonic()+(1100 if a.phase=='safety' else 1500)
            while worker.poll() is None:
                if server.poll() is not None:
                    atomic(root/'server_exited.json',dict(wall_s=time.monotonic(),exit_code=server.returncode))
                    raise RuntimeError('ServerExitedDuringRun')
                if time.monotonic()>deadline:raise RuntimeError('PhaseWallBudgetExceeded')
                time.sleep(.2)
        report['worker_exit_code']=worker.returncode
        if worker.returncode:raise RuntimeError('PhaseFailed:'+str(worker.returncode))
        report['status']='Passed'
    except BaseException as exc:report.update(status='Partial',error=repr(exc))
    finally:
        terminate(worker);terminate(server);time.sleep(1)
        report.update(ports_after={str(p):listening(p) for p in ports},remaining_server_group=[] if server is None else group_members(server.pid),remaining_worker_group=[] if worker is None else group_members(worker.pid))
        report['cleanup_complete']=not any(report['ports_after'].values()) and not report['remaining_server_group'] and not report['remaining_worker_group']
        if not report['cleanup_complete']:report['status']='Partial'
        report['logs']={p.name:digest(p) for p in root.glob('*.log')}
        write(root/'supervisor.json',report);write(Path('reports/m8j13_sources')/(a.phase+'_'+a.attempt+'_supervisor.json'),report)
        print(json.dumps(report,indent=2),flush=True)
    if report['status']!='Passed':raise SystemExit(1)


if __name__=='__main__':main()
