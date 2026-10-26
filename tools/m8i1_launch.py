"""Single-use server/process supervisor, hard timeout, own process groups only."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from tools.m8i1_common import ROOT, PROTOCOL, HISTORY, sha, asset, write, log_json, provenance


def stop_group(process):
    if process is None:
        return dict(started=False)
    pid=process.pid
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        try: os.killpg(pid,sig)
        except ProcessLookupError: break
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: continue
        # Shell may have exited while a child survives in the same group.
        try: os.killpg(pid,0)
        except ProcessLookupError: break
    try: os.killpg(pid,0); alive=True
    except ProcessLookupError: alive=False
    return dict(pid=pid,returncode=process.poll(),process_group_absent=not alive)


def main():
    cfg=json.loads(PROTOCOL.read_text())
    # Refuse second run even if the first session failed before spawning.
    ROOT.mkdir(exist_ok=False)
    start=time.monotonic(); server=None; runner=None; fail=None
    write(ROOT/'launch_once.json',dict(**provenance('ActualServerLaunchIntent'),monotonic_start=start))
    for port in [cfg['port'],cfg['port']+1,cfg['port']+2,cfg['tm_port']]:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0: raise RuntimeError('PortOccupied_NoSessionStarted')
    frozen=json.loads(Path('reports/m8i1_preflight_freeze.json').read_text())
    for row in frozen['files']:
        if sha(row['path'])!=row['sha256']: raise RuntimeError('FrozenFileChanged:'+row['path'])
    log=ROOT/'server.log'
    command=['runuser','-u','ubuntu','--','env','-u','DISPLAY',
             'VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json',
             '/root/gpufree-data/simulators/CARLA_0.9.15/CarlaUE4.sh','/Game/Carla/Maps/Town02',
             '-RenderOffScreen','-quality-level=Epic','-nosound','-carla-rpc-port='+str(cfg['port']),
             '-stdout','-FullStdOutLogOutput']
    write(ROOT/'execution_commands.json',dict(server=command,runner=[sys.executable,'-m','tools.m8i1_run_session','--start',str(start)]))
    try:
        with log.open('xb') as stream:
            server=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        import carla
        connected=False
        deadline=min(start+180,start+cfg['wall_seconds'])
        while time.monotonic()<deadline:
            if server.poll() is not None: raise RuntimeError('OwnServerExitedDuringStartup')
            try:
                c=carla.Client('127.0.0.1',cfg['port']);c.set_timeout(2)
                version=c.get_server_version();connected=True;break
            except RuntimeError: time.sleep(1)
        if not connected: raise RuntimeError('StartupTimeout_NoRetry')
        with (ROOT/'session_stdout.log').open('xb') as stream:
            runner=subprocess.Popen([sys.executable,'-m','tools.m8i1_run_session','--start',str(start)],
                         stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while runner.poll() is None:
            used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if used>cfg['max_bytes'] or time.monotonic()-start>cfg['wall_seconds']:
                raise RuntimeError('SupervisorResourceBudgetExceeded')
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,timeout=5)
            log_json(ROOT/'resource_samples.jsonl',dict(elapsed_s=time.monotonic()-start,disk_bytes=used,
                         gpu_memory_MiB_utilization_percent=gpu.stdout.strip(),returncode=gpu.returncode))
            time.sleep(2)
        if runner.returncode: fail='SessionProcessExit:'+str(runner.returncode)
    except Exception as exc:
        fail=type(exc).__name__+': '+str(exc)
    finally:
        # SIGTERM handler lets runner execute finally/cleanup before force kill.
        run_stop=stop_group(runner); server_stop=stop_group(server)
        write(ROOT/'supervisor_result.json',dict(**provenance('ActualBoundedServerSupervisor'),
              failure=fail,runner=run_stop,server=server_stop,elapsed_s=time.monotonic()-start,
              disk_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
              startup_frames='Server asynchronous boot before exclusive acquisition; not traffic session; no actors spawned then',
              no_server_restart=True))
    print('supervisor complete',fail,flush=True)


if __name__=='__main__': main()
