"""Read-only verification of the authorized actual failure, preserving offline seals."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from tools.m8i1a_common import REPO, sha
from tools.m8i1a_runtime_closeout import RUNTIME, EVIDENCE, EXECUTED


def read(path):
    return json.loads(Path(path).read_text())


def check_files(manifest):
    for row in manifest['files']:
        p = Path(row['path'])
        assert p.is_file() and sha(p) == row['sha256'] and p.stat().st_size == row['bytes'], str(p)
    return len(manifest['files'])


def main():
    manifest_path = REPO / 'reports/m8i1a_execution_manifest.json'
    manifest = read(manifest_path)
    count = check_files(manifest)
    for row in manifest['files']:
        p = Path(row['path'])
        if p.suffix == '.json': read(p)
        elif p.suffix == '.py': ast.parse(p.read_text())
        elif p.suffix == '.md':
            for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
                if '://' not in link and not link.startswith('#'):
                    assert (p.parent / link).exists(), (p, link)
    historical = {}
    for name in ['m8i1', 'm8i1a']:
        historical[name] = check_files(read(REPO / f'reports/{name}_release_manifest.json'))
    frozen = check_files(read(REPO / 'reports/m8i1a_prelaunch_freeze.json'))
    context = read(RUNTIME / 'context.json')
    assert context['git_head'] == EXECUTED
    for path, digest in context['source_sha256'].items():
        blob = subprocess.check_output(['git', 'show', EXECUTED + ':' + path], cwd=REPO)
        assert hashlib.sha256(blob).hexdigest() == digest == sha(REPO / path)
    approval = read(EVIDENCE / 'authorization.json')
    assert approval['approved'] and approval['max_launches'] == 1
    assert approval['user_message_id'] is None
    assert approval['user_message_text'].startswith('批准了')
    assert '不得编造批准消息ID' in approval['user_message_text']
    session = read(RUNTIME / 'session_result.json')
    supervisor = read(RUNTIME / 'supervisor_result.json')
    assert supervisor['authorization'] == approval
    assert supervisor['server_launches'] == 1
    assert session['failure'] == 'ClientConnectionFailed' and session['identity_records'] == []
    assert session['expected_runner_returncode'] == supervisor['runner_returncode'] == 10
    assert supervisor['supervisor_returncode'] == 1
    assert read(EVIDENCE / 'launcher_exit_receipt.json')['actual_supervisor_process_returncode'] == 1
    assert supervisor['propagation'] == 'StatusPropagationGo' and not supervisor['contradictions']
    assert supervisor['status'] == 'StartupDiagnosticNoGo'
    assert supervisor['wall_elapsed_s'] < 180
    assert len(list(RUNTIME.glob('connection_*.json'))) == 31
    assert not list(RUNTIME.glob('identity_*.json')) and not list(RUNTIME.rglob('*.png'))
    assert sum(p.stat().st_size for p in RUNTIME.rglob('*') if p.is_file()) < 67108864
    assert read(RUNTIME / 'cleanup.json')['ok']
    post = read(EVIDENCE / 'post_shutdown_OS_probe.json')
    assert post['probe']['ok'] and not post['matching_CARLA_processes']
    acc = read(REPO / 'reports/m8i1a_execution_acceptance.json')
    assert acc['status'] == 'StartupDiagnosticNoGo' and acc['cleanup'] == 'CleanupGo'
    assert acc['approval_consumed'] and not acc['further_startup_authorized']
    assert not acc['TM_pilot_proposal_eligible'] and not acc['TM_pilot_authorized']
    assert all(acc[k] == 0 for k in ['actor_creations', 'sensor_creations', 'TM_instances', 'owner_ticks', 'model_forwards', 'training_steps'])
    changed = subprocess.check_output(['git', 'diff', EXECUTED, '--name-status', '--diff-filter=MDTR'], cwd=REPO, text=True)
    assert not changed.strip(), changed
    print(json.dumps(dict(kind='OfflineVerificationOfActualAuthorizedAttempt', status='Passed',
        executed_commit=EXECUTED, verified_new_files=count, historical_files=historical,
        frozen_inputs=frozen, new_live_connections=0, actual_attempts=1,
        actual_startup='StartupDiagnosticNoGo', actual_identity='StartupIdentityUnknown',
        propagation='StatusPropagationGo', cleanup='CleanupGo', runner_rc=10, supervisor_rc=1,
        approval_consumed=True, further_startup_authorized=False,
        manifest_sha256=sha(manifest_path)), indent=2))


if __name__ == '__main__':
    main()
