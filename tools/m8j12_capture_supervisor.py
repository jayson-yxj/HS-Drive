"""Own one bounded CARLA process group and terminate it before offline inference."""
import hashlib,json,os,signal,socket,subprocess,time
from pathlib import Path
from datetime import datetime,timezone


def listening(port):
    with socket.socket() as s:
        s.settimeout(.2)
        return s.connect_ex(('127.0.0.1',port))==0


def group_members(pgid):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if os.getpgid(int(p.name))==pgid:
                result.append(dict(pid=int(p.name),argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),cwd=os.readlink(p/'cwd')))
        except (ProcessLookupError,FileNotFoundError,PermissionError):pass
    return result


def main():
    import carla
    os.umask(0o077)
    c=json.loads(Path('configs/m8j12_protocol.json').read_text());root=Path(c['private_root']);port=2300
    if any(listening(p) for p in (port,port+1,port+2)):raise RuntimeError('DedicatedPortsOccupied')
    if not Path('reports/m8j12_sources/model_load.json').exists():raise RuntimeError('ModelLoadNotVerified')
    argv=['runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),'/root/gpufree-data/simulators/CARLA_0.9.15',str(port),'Epic']
    log=root/'carla_stdout.log';report=dict(utc=datetime.now(timezone.utc).isoformat(),argv=argv,cwd=str(Path.cwd()),port=port,authorization='Current user M8-J12 attachment explicitly permits two bounded captures; no fabricated message ID',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),ready_attempts=[],startup_identity=None)
    proc=None;worker=None;error=None
    try:
        with log.open('x') as f:
            proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            report['pid']=proc.pid;deadline=time.monotonic()+150
            while time.monotonic()<deadline:
                if proc.poll() is not None:raise RuntimeError('ServerExitedBeforeReady')
                client=carla.Client('127.0.0.1',port);client.set_timeout(4.)
                try:
                    version=client.get_server_version();world=client.get_world();cmap=world.get_map();snap=world.get_snapshot()
                    identity=dict(map=cmap.name,world_episode=world.id,frame=snap.frame,timestamp_s=snap.timestamp.elapsed_seconds,
                        client_version=client.get_client_version(),server_version=version,xodr_sha256=hashlib.sha256(cmap.to_opendrive().encode()).hexdigest(),processes=group_members(proc.pid))
                    # Persist identity before validation or requesting the fixed Town03.
                    (root/'startup_identity.json').write_text(json.dumps(identity,indent=2)+'\n');report['startup_identity']=identity
                    if version!='0.9.15' or client.get_client_version()!='0.9.15':raise ValueError('VersionMismatch')
                    break
                except RuntimeError as exc:
                    report['ready_attempts'].append(dict(time_utc=datetime.now(timezone.utc).isoformat(),error=str(exc)))
                    time.sleep(1.)
            else:raise RuntimeError('BoundedCARLAReadinessFailed')
            cmd=[str(Path('.venv/bin/python').resolve()),'-m','tools.m8j12_capture','--port',str(port)]
            with (root/'capture.log').open('x') as capturelog:
                worker=subprocess.Popen(cmd,stdout=capturelog,stderr=subprocess.STDOUT,start_new_session=True)
                try:rc=worker.wait(timeout=600)
                except subprocess.TimeoutExpired:raise RuntimeError('CaptureWallTimeout')
            report['capture_exit_code']=rc
            if rc:raise RuntimeError('CaptureFailed:'+str(rc))
    except BaseException as exc:
        error=repr(exc)
    finally:
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid,signal.SIGTERM)
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
        if proc is not None:
            report['owned_before_cleanup']=group_members(proc.pid)
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            time.sleep(2)
            remaining=group_members(proc.pid)
            if remaining:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                time.sleep(1)
            report['server_exit_code']=proc.returncode;report['owned_after_cleanup']=group_members(proc.pid)
        report['ports_after']={str(p):listening(p) for p in (port,port+1,port+2)}
        report['error']=error;report['cleanup_complete']=not report.get('owned_after_cleanup') and not any(report['ports_after'].values())
        report['status']='Passed' if not error and report['cleanup_complete'] else 'Partial'
        report['server_log_sha256']=hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None
        (root/'capture_supervisor.json').write_text(json.dumps(report,indent=2)+'\n')
        public={k:v for k,v in report.items() if k not in ('startup_identity','owned_before_cleanup')}
        Path('reports/m8j12_sources/capture_supervisor.json').write_text(json.dumps(public,indent=2)+'\n')
    if report['status']!='Passed':raise RuntimeError(str(error or 'CleanupIncomplete'))


if __name__=='__main__':main()
