"""Bounded read-only local forensics. No CARLA import, socket or server launch."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
BASE = '5c14c347de2b183d59d7bd3f604ec45d0b602da6'
OUT = Path('/root/gpufree-data/hs-drive-experiments/m8i1b-offline-forensics')
EXP = Path('/root/gpufree-data/hs-drive-experiments')
INSTALL = Path('/root/gpufree-data/simulators/CARLA_0.9.15')
I1 = EXP / 'm8i1-traffic-development'
I1A = EXP / 'm8i1a-startup-identity'
APPROVAL = EXP / 'm8i1a-approved-execution-evidence'
H1 = EXP / 'm8h1-native-template-smoke'
WINDOWS = {
    'M8-H1': ('2026-10-25T15:58:00+00:00', '2026-10-25T16:20:00+00:00'),
    'M8-I1': ('2026-10-26T03:21:30+00:00', '2026-10-26T03:22:30+00:00'),
    'M8-I1a': ('2026-10-26T04:38:00+00:00', '2026-10-26T04:40:40+00:00'),
}


def utc(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def asset(path):
    p = Path(path); s = p.stat()
    return dict(path=str(p), bytes=s.st_size, sha256=sha(p),
                mtime_ns=s.st_mtime_ns, mtime_utc=utc(s.st_mtime),
                ctime_utc=utc(s.st_ctime), uid=s.st_uid, gid=s.st_gid,
                mode=oct(s.st_mode & 0o7777))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def report(name, **fields):
    write(REPO / f'reports/m8i1b_{name}.json', dict(
        kind='ActualReadOnlyOfflineForensics', base_commit=BASE, collector=asset(__file__),
        server_launches=0, CARLA_connections=0, actor_creations=0, sensor_creations=0,
        TM_instances=0, owner_ticks=0, model_forwards=0, training_steps=0, visual_control=False,
        **fields))


def command(name, args):
    r = subprocess.run(args, capture_output=True, timeout=25, env={**os.environ, 'TZ': 'UTC'})
    for suffix, raw in [('stdout', r.stdout), ('stderr', r.stderr)]:
        with (OUT / f'{name}.{suffix}.txt').open('xb') as f:
            f.write(raw)
    record = dict(command=args, returncode=r.returncode, captured_utc=utc(time.time()),
                  stdout=asset(OUT / f'{name}.stdout.txt'), stderr=asset(OUT / f'{name}.stderr.txt'))
    write(OUT / f'{name}.command.json', record)
    return record, r.stdout.decode(errors='replace')


def within(timestamp, window):
    lo, hi = (dt.datetime.fromisoformat(v).timestamp() for v in window)
    return lo <= timestamp <= hi


def main():
    OUT.mkdir(exist_ok=False)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == BASE
    manifests = ['m8h1_release_manifest.json', 'm8i1_release_manifest.json',
                 'm8i1a_release_manifest.json', 'm8i1a_execution_manifest.json', 'm8i1a_prelaunch_freeze.json']
    frozen = {}; seals = []
    for name in manifests:
        p = REPO / 'reports' / name; m = read(p)
        for row in m['files']:
            assert sha(row['path']) == row['sha256'], row['path']
            frozen[row['path']] = dict(path=row['path'], sha256=row['sha256'], bytes=Path(row['path']).stat().st_size)
        seals.append(dict(manifest=asset(p), entries=len(m['files'])))
    write(OUT / 'historical_freeze.json', dict(manifests=seals, files=list(frozen.values())))

    # Search only relevant install/user/log locations, not unrelated credentials/data.
    userhome = Path(pwd.getpwnam('ubuntu').pw_dir)
    roots = [INSTALL / 'CarlaUE4/Saved', INSTALL / 'Engine/Saved', INSTALL / 'Saved',
             userhome / '.config/Epic/CarlaUE4/Saved', userhome / '.local/share/CarlaUE4',
             userhome / '.local/share/Epic', userhome / 'CarlaUE4/Saved',
             Path('/root/.config/Epic/CarlaUE4/Saved'), Path('/root/.local/share/CarlaUE4'),
             Path('/var/lib/systemd/coredump'), Path('/var/crash')]
    scopes = []; candidates = []
    for root in roots:
        entry = dict(path=str(root), exists=root.exists(), errors=[])
        paths = []
        if root.exists():
            for directory, dirs, files in os.walk(root, onerror=lambda e: entry['errors'].append(str(e))):
                for name in files:
                    p = Path(directory) / name
                    if (p.suffix.lower() in ['.log', '.txt', '.runtime-xml', '.dmp', '.core']
                            or name == 'CrashReportClient.ini' or '.log.' in name):
                        paths.append(p)
        entry['candidate_count'] = len(paths); scopes.append(entry)
        for p in paths:
            row = asset(p)
            row['windows_by_mtime_only'] = [k for k, w in WINDOWS.items() if within(p.stat().st_mtime, w)]
            if p.name == 'CrashContext.runtime-xml':
                tree = ET.parse(p).getroot()
                fields = {k: tree.findtext('.//' + k) for k in ['ProcessId', 'SecondsSinceStart', 'CrashType',
                          'ErrorMessage', 'CommandLine', 'BaseDir', 'EngineVersion']}
                row.update(kind='UECrashRecord', fields=fields, attribution='ExcludedOtherRun',
                           reason='PID799952, port2000, Low quality and 2026-10-19 date differ from all three audited windows')
            elif p.name == 'CrashReportClient.ini':
                row.update(kind='CrashReporterConfiguration_NotCrashEvent', text=p.read_text(),
                           attribution='TemporalConfigurationCorrelationOnly_NoPIDOrMapOrError')
            else:
                row.update(kind='ExistingUEAuxiliaryArtifact', attribution='NoTargetRunMatch',
                           reason='No matching PID and execution window')
            candidates.append(row)
    saved_stdout = []
    for stage, root in [('M8-I1', I1), ('M8-I1a', I1A)]:
        for p in sorted(root.glob('*.log')):
            saved_stdout.append(dict(**asset(p), stage=stage, kind='OwnedProcessRedirectedStdout',
                confidence='High_RunDirectoryAndExecutedPopenLink',
                contains_loaded_map=False, line_timestamps=False, text=p.read_text(errors='replace')))
    old_logs = []
    for p in sorted((REPO / 'logs').glob('*.log')):
        if any(k in p.name.lower() for k in ['carla', 'server', 'vulkan']):
            old_logs.append(dict(**asset(p), attribution='ExcludedOtherRun_OldMtime',
                                within_any_target_window=any(within(p.stat().st_mtime, w) for w in WINDOWS.values())))
    write(OUT / 'log_search.json', dict(scopes=scopes, UE_candidates=candidates,
          historical_workspace_logs=old_logs, own_redirected_stdout=saved_stdout))

    journal_record, _ = command('journal_boots', ['journalctl', '--list-boots', '--no-pager'])
    journal_windows = []
    for stage, (a, b) in WINDOWS.items():
        r, _ = command('journal_' + stage.lower(), ['journalctl', '-k', '--since', a, '--until', b, '--no-pager'])
        journal_windows.append(r)
    dmesg_record, kernel = command('kernel_ring_utc', ['dmesg', '--time-format', 'iso'])
    time_anchor = dict(wall_utc=utc(time.time()), clock_monotonic_s=time.monotonic(),
                       clock_boottime_s=time.clock_gettime(time.CLOCK_BOOTTIME),
                       clk_tck=os.sysconf('SC_CLK_TCK'), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                       warning='Current wall/boot conversion is approximate, not a historical NTP-offset measurement')
    write(OUT / 'clock_anchor.json', time_anchor)
    kernel_windows = {}
    pattern = re.compile(r'Xid|NVRM|Out of memory|oom-kill|Killed process|segfault|Carla|denied|permission|GPU|Vulkan', re.I)
    for stage, window in WINDOWS.items():
        rows = []
        for line in kernel.splitlines():
            try:
                t = dt.datetime.fromisoformat(line.split()[0].replace(',', '.')).timestamp()
            except (ValueError, IndexError):
                continue
            if within(t, window): rows.append(line)
        matched = [line for line in rows if pattern.search(line)]
        write(OUT / ('kernel_' + stage.lower() + '_window.json'), dict(lines=rows, targeted_error_lines=matched))
        kernel_windows[stage] = dict(total_lines=len(rows), targeted_error_lines=matched,
            limitations='Host ring buffer/container PID namespaces and wall-clock conversion; no hit is not proof of no system error')
    system_locations = ['/var/log/kern.log', '/var/log/syslog', '/var/log/messages', '/var/log/audit',
                        '/var/log/journal', '/run/log/journal']
    system_files = []
    for name in ['Xorg.20.log', 'xorg.log', 'de.log', 'dbus.log', 'selkies.log']:
        p = Path('/var/log') / name
        if p.exists():
            text = p.read_text(errors='replace')
            system_files.append(dict(**asset(p), relevant_lines=[l for l in text.splitlines() if pattern.search(l)][:40],
                                     attribution='NoRunMatchedError_NoServerPIDAssociation'))
    windows_detail = dict(utc_windows=WINDOWS, window_basis={
        'M8-H1':'Conservative audit window around 15:59:04 reporter-config mtime and 16:00 preflight/16:04 actor outputs; exact launch time not recorded',
        'M8-I1':'launch_once/execution_commands mtime 03:21:44 through supervisor_result 03:22:07; expanded bounds',
        'M8-I1a':'launch_intent 04:38:19 through supervisor_result 04:39:53 and post-shutdown evidence 04:40:28; expanded bounds'},
        mtime_is_not_intrinsic_log_timestamp=True)
    report('runtime_log_inventory', status='NoRunMatchedUELog',
           status_scope='No separate UE file log with map-loading/engine-event timeline matched to audited runs; redirected stdout retained separately',
           historical_seals=seals, execution_windows=windows_detail, search_scopes=scopes,
           UE_candidate_logs=candidates, owned_process_stdout=saved_stdout,
           excluded_old_workspace_logs=old_logs, kernel_ring=dmesg_record,
           kernel_windows=kernel_windows, journal_commands=[journal_record, *journal_windows],
           system_locations=[dict(path=p, exists=Path(p).exists()) for p in system_locations],
           OS_log_files=system_files, supported_environment_issue=False,
           PID_attribution='Recorded container PIDs; host kernel PID equivalence was not captured',
           no_missing_log_regenerated=True)
    print(json.dumps(dict(log_status='NoRunMatchedUELog', UE_candidates=len(candidates),
          historical_unique_files=len(frozen), kernel_windows=kernel_windows, output=str(OUT)), indent=2))


if __name__ == '__main__':
    main()
