"""Offline verification only. No CARLA imports, connection, authorization or launch."""
import ast
import json
from pathlib import Path
import re
import subprocess
from tools.m8i1a_common import REPO, BASE, config, sha


def main():
    manifest = json.loads((REPO / 'reports/m8i1a_release_manifest.json').read_text())
    paths = []
    for row in manifest['files']:
        p = Path(row['path'])
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], str(p)
        paths.append(p)
        if p.suffix == '.json': json.loads(p.read_text())
        if p.suffix == '.py': ast.parse(p.read_text())
        if p.suffix == '.md':
            for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
                if '://' not in link and not link.startswith('#'):
                    assert (p.parent / link).exists(), (p, link)
    required = ['summary.md', 'acceptance.json', 'static_launch_audit.json', 'map_identity.json',
                'startup_readiness.json', 'exit_status_contract.json', 'failure_injection_tests.json',
                'runtime_probe.json', 'cleanup.json', 'root_cause_assessment.md']
    assert all(REPO / ('reports/m8i1a_' + x) in paths for x in required)
    assert all(REPO / ('docs/M8I1A_' + x + '.md') in paths for x in ['IMPLEMENTATION', 'ARCHITECTURE_DECISION'])
    freeze = json.loads((REPO / 'reports/m8i1a_prelaunch_freeze.json').read_text())
    for row in freeze['files']:
        assert sha(row['path']) == row['sha256'], row['path']
    tests = json.loads((REPO / 'reports/m8i1a_failure_injection_tests.json').read_text())
    assert tests['status'] == 'OfflineInstrumentationPassed' and tests['test_count'] == 25
    assert all(t['passed'] for t in tests['tests']) and len(tests['subprocess_cases']) == 10
    assert tests['actual_server_launches'] == tests['actual_CARLA_connections'] == 0
    assert all(c['real_server_launches'] == 0 for c in tests['subprocess_cases'])
    identity = json.loads((REPO / 'reports/m8i1a_map_identity.json').read_text())
    assert identity['actual_map'] is identity['world_episode'] is identity['actual_xodr_sha256'] is None
    runtime = json.loads((REPO / 'reports/m8i1a_runtime_probe.json').read_text())
    assert runtime['status'] == 'NotRun' and not runtime['authorization_received']
    assert not Path(config()['runtime_root']).exists(), 'A new authorized execution requires its own review, not rewriting this offline seal'
    acc = json.loads((REPO / 'reports/m8i1a_acceptance.json').read_text())
    assert acc['status'] == 'OfflineInstrumentationPassed' and acc['actual_startup'] == 'NotRun'
    assert not acc['next_TM_pilot_proposal_eligible'] and not acc['runtime_authorized']
    changed = subprocess.check_output(['git', 'diff', BASE, '--name-status', '--diff-filter=MDTR'], cwd=REPO, text=True)
    assert not changed.strip(), changed
    old = subprocess.run(['python3', '-m', 'tools.m8i1_verify'], cwd=REPO, capture_output=True, text=True, check=True)
    old = json.loads(old.stdout)
    print(json.dumps(dict(kind='ActualOfflineReleaseVerification', status='Passed',
        new_sealed_files=len(paths), prelaunch_frozen_files=len(freeze['files']), required_deliverables=12,
        offline_tests=25, mock_subprocess_cases=10, runtime='NotRun',
        old_m8i1_files=old['sealed_files'], historical_entries=old['historical_entries'],
        base_tracked_modifications=0, runtime_authorized=False,
        release_manifest_sha256=sha(REPO / 'reports/m8i1a_release_manifest.json')), indent=2))


if __name__ == '__main__':
    main()
