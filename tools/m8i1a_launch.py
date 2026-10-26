"""Separately authorized single-process startup; never starts traffic or ticks."""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import time
import uuid
from tools.m8i1a_common import REPO, OLD_ROOT, PROTOCOL, config, persist, provenance, sha
from tools.m8i1a_identity import reconcile


def group_members(pgid):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if os.getpgid(int(p.name)) != pgid:
                continue
            found.append(dict(pid=int(p.name), pgid=pgid, cwd=os.readlink(p / 'cwd'),
                              cmdline=(p / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace'),
                              stat=(p / 'stat').read_text()))
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue
    return found


def os_probe(groups, ports):
    living = {}
    for pgid in groups:
        try:
            os.killpg(pgid, 0)
            living[str(pgid)] = True
        except ProcessLookupError:
            living[str(pgid)] = False
    listening = {}
    for port in ports:
        with socket.socket() as s:
            s.settimeout(.15)
            listening[str(port)] = s.connect_ex(('127.0.0.1', port)) == 0
    return dict(group_alive=living, listening_ports=listening,
                ok=not any(living.values()) and not any(listening.values()))


def stop_owned(processes, ports, seconds):
    """Keep transient failures; final independent OS probe governs cleanup."""
    groups = [p.pid for p in processes if p is not None]
    trace = []
    start = time.monotonic()
    for sig, wait in [(signal.SIGTERM, min(5, seconds / 2)), (signal.SIGKILL, max(0, seconds - 5))]:
        for pgid in groups:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
        end = min(start + seconds, time.monotonic() + wait)
        while True:
            for p in processes:
                if p is not None:
                    p.poll()  # Reap direct children before group/port test.
            probe = os_probe(groups, ports)
            trace.append(dict(elapsed_s=time.monotonic() - start, signal=int(sig), **probe))
            if probe['ok'] or time.monotonic() >= end:
                break
            time.sleep(.1)
        if probe['ok']:
            break
    final = os_probe(groups, ports)
    return dict(ok=final['ok'], probes=trace, final_OS_probe=final,
                scope='Own process groups only; no Actor/Sensor/TM creation or World changes')


def authorize(receipt_path, cfg):
    """A receipt records a separate human approval; this program never creates one."""
    receipt = json.loads(Path(receipt_path).read_text())
    if not (receipt.get('approved') is True and receipt.get('scope') == 'M8-I1a_OneReadOnlyStartup'
            and receipt.get('user_approval_reference') and receipt.get('max_launches') == 1
            and receipt.get('protocol_sha256') == sha(PROTOCOL)):
        raise ValueError('AuthorizationMissingOrMismatch')
    freeze = json.loads((REPO / 'reports/m8i1a_prelaunch_freeze.json').read_text())
    for item in freeze['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('PrelaunchSourceOrResourceChanged:' + item['path'])
    tests = json.loads((REPO / 'reports/m8i1a_failure_injection_tests.json').read_text())
    if tests['status'] != 'OfflineInstrumentationPassed':
        raise ValueError('OfflineGateNotPassed')
    return receipt


def load_optional(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', required=True,
                        help='Separately obtained user approval receipt; no default authorization')
    args = parser.parse_args()
    cfg = config()
    approval = authorize(args.authorization, cfg)  # No CARLA import, socket or launch before approval.
    root = Path(cfg['runtime_root'])
    root.mkdir(exist_ok=False)  # Permanent single-attempt marker, even on preflight failure.
    run_id = str(uuid.uuid4())
    start = time.monotonic()
    server = runner = None
    error = None
    selector = selectors.DefaultSelector()
    streams = []
    commands = json.loads((OLD_ROOT / 'execution_commands.json').read_text())
    context = dict(**provenance('ActualReadOnlyCARLAStartup'), run_id=run_id,
                   runtime_authorized=True, authorization=approval, server_argv=commands['server'],
                   requested_server_cwd=str(REPO), supervisor_cwd=os.getcwd(),
                   install=cfg['install'], protocol_sha256=sha(PROTOCOL))
    persist(root / 'launch_intent.json', context)
    def attach(process, name):
        log = (root / name).open('xb')
        streams.append(log)
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ, log)
    try:
        initial = os_probe([], cfg['ports_to_check'])
        persist(root / 'prelaunch_ports.json', initial)
        if not initial['ok']:
            raise RuntimeError('PortOccupied')
        try:
            server = subprocess.Popen(commands['server'], cwd=REPO, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            raise RuntimeError('ServerLaunchFailed') from exc
        attach(server, 'server.log')
        initial_process = dict(pid=server.pid, members=group_members(server.pid))
        persist(root / 'server_process_initial.json', initial_process)
        context.update(server_pid=server.pid, initial_process_identity=initial_process,
                       process_identity_updates=str(root / 'server_process_binary.json'))
        persist(root / 'context.json', context)
        runner_cmd = [sys.executable, '-m', 'tools.m8i1a_run_session', '--context',
                      str(root / 'context.json'), '--output', str(root)]
        persist(root / 'execution_commands.json', dict(server=commands['server'], server_cwd=str(REPO),
                                                      runner=runner_cmd, runner_cwd=str(REPO)))
        runner = subprocess.Popen(runner_cmd, cwd=REPO, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        attach(runner, 'runner.log')
        saved_binary_identity = False
        while True:
            for key, _ in selector.select(.1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                    remaining = cfg['max_log_bytes'] - used - 1048576  # Reserve result/cleanup space.
                    if remaining < len(chunk):
                        if remaining > 0:
                            key.data.write(chunk[:remaining]); key.data.flush()
                        raise RuntimeError('ResourceBudgetExceeded')
                    key.data.write(chunk); key.data.flush()
                else:
                    selector.unregister(key.fileobj)
            if not saved_binary_identity:
                members = group_members(server.pid)
                if any('CarlaUE4-Linux-Shipping' in p['cmdline'] for p in members):
                    persist(root / 'server_process_binary.json', dict(members=members))
                    saved_binary_identity = True
            if time.monotonic() - start >= cfg['work_seconds']:
                raise RuntimeError('ResourceBudgetExceeded')
            if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > cfg['max_log_bytes'] - 1048576:
                raise RuntimeError('ResourceBudgetExceeded')
            if server.poll() is not None:
                raise RuntimeError('ServerExited')
            if runner.poll() is not None:
                break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        cleanup = stop_owned([runner, server], cfg['ports_to_check'], cfg['cleanup_seconds'])
        # Drain already buffered log bytes after process termination, within the same cap.
        for key in list(selector.get_map().values()):
            while True:
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                room = max(0, cfg['max_log_bytes'] - used - 1048576)
                key.data.write(chunk[:room]); key.data.flush()
                if len(chunk) > room:
                    error = 'ResourceBudgetExceeded'
                    break
        selector.close()
        for stream in streams:
            stream.flush(); os.fsync(stream.fileno()); stream.close()
        persist(root / 'cleanup.json', cleanup)
        result = reconcile(None if runner is None else runner.poll(),
                           load_optional(root / 'session_result.json'),
                           load_optional(root / 'runner_cleanup.json'), cleanup, run_id, error)
        result.update(wall_elapsed_s=time.monotonic() - start, authorization=approval,
                      kind='ActualReadOnlyStartupSupervisor', server_launches=int(server is not None),
                      server_returncode=None if server is None else server.poll())
        if result['wall_elapsed_s'] > cfg['wall_seconds']:
            result['failures'].append('ResourceBudgetExceeded')
            result['status'] = 'StartupDiagnosticNoGo'; result['supervisor_returncode'] = 1
        persist(root / 'supervisor_result.json', result)
    return result['supervisor_returncode']


if __name__ == '__main__':
    raise SystemExit(main())
