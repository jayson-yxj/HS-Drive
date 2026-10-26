"""Read-only artifact and failure-gate verification. No CARLA connection/model."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from tools.m8i1_common import ROOT, BASE, sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    manifest_path = Path('reports/m8i1_release_manifest.json')
    manifest = read(manifest_path)
    verified = set()
    for row in manifest['files']:
        p = Path(row['path'])
        assert p.is_file() and p.stat().st_size == row['bytes'], str(p)
        assert sha(p) == row['sha256'], str(p)
        if p.suffix == '.json':
            read(p)
        elif p.suffix == '.jsonl':
            for line in p.read_text().splitlines():
                json.loads(line)
        elif p.suffix == '.py':
            ast.parse(p.read_text())
        elif p.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
                if '://' not in target and not target.startswith('#'):
                    assert (p.parent / target.split('#')[0]).exists(), (str(p), target)
        verified.add(str(p))
    required = ['summary.md', 'acceptance.json', 'session_and_tick_audit.json',
                'spawn_lifecycle.json', 'traffic_actor_snapshots.json', 'sensor_sync.json',
                'rgb_input_isolation.json', 'gt_sidecar_contract.json', 'render_identity.json',
                'vehicle_walker_20_40m.json', 'background_unknown_audit.json',
                'natural_scene_review.json', 'historical_overlap.json',
                'resource_and_cleanup.json', 'training_readiness.json']
    assert all('reports/m8i1_' + name in verified for name in required)
    assert all('docs/M8I1_' + name + '.md' in verified
               for name in ['IMPLEMENTATION', 'ARCHITECTURE_DECISION'])
    result = read(ROOT / 'session_result.json')
    assert result['failure'] == 'RuntimeError: MapMismatch_NoReload'
    assert result['ticks'] == 0 and result['saved_frames'] == []
    assert result['vehicle_ids'] == result['walker_ids'] == [] and result['ego_id'] is None
    assert result['execution_commit'] == manifest['executed_commit']
    for path, digest in result['source_sha256'].items():
        blob = subprocess.check_output(['git', 'show', result['execution_commit'] + ':' + path])
        assert hashlib.sha256(blob).hexdigest() == digest, path
        if not path.startswith('tests/'):
            assert sha(path) == digest, 'Executed runtime unexpectedly changed: ' + path
    frozen = read('reports/m8i1_preflight_freeze.json')
    test_delta = []
    for row in frozen['files']:
        if sha(row['path']) != row['sha256']:
            assert row['path'] == 'tests/test_m8i1_contract.py'
            test_delta.append(row['path'])
    assert test_delta == ['tests/test_m8i1_contract.py']
    acc = read('reports/m8i1_acceptance.json')
    assert acc['status'] == 'PilotNoGo' and acc['training_readiness'] == 'TrainingReadinessNoGo'
    assert acc['session_attempts'] == 1 and acc['retries'] == 0
    assert acc['actual_successful_science_keyframes'] == 0 and not acc['relaunch_authorized']
    assert not any(ROOT.rglob('*.png'))
    for directory in ['RGBInferencePacket', 'EvaluatorSidecar']:
        assert not list((ROOT / directory).iterdir())
    iso = read('reports/m8i1_rgb_input_isolation.json')
    assert iso['real_pilot_packet_status'] == 'NotRun_NoImages'
    assert len(iso['forbidden_cases']) == 6
    assert all(r['rejected'] and not r['GT_path_opened'] for r in iso['forbidden_cases'])
    assert iso['unprivileged_RGB_probe']['returncode'] == 0
    assert iso['unprivileged_sidecar_probe']['error'] == 'PermissionError'
    post = read(ROOT / 'post_shutdown_process_probe.json')
    assert post['all_ports_closed'] and post['own_process_groups_absent']
    assert post['matching_processes'] == []
    history = read('reports/m8i1_historical_overlap.json')
    assert history['records'] == 16562 and history['new_Final'] == 0
    assert len(history['near_footprint_regressions']) == 5
    assert all(row['passed'] for row in history['near_footprint_regressions'])
    for p in Path('reports').glob('m8i1_*.json'):
        if p.name == manifest_path.name:
            continue
        row = read(p)
        assert row['model_forwards'] == row['training_steps'] == 0
        assert not row['visual_control'] and not row['FinalEligible']
    changes = subprocess.check_output(
        ['git', 'diff', '--name-status', BASE, '--diff-filter=MDTR'], text=True).strip()
    assert not changes, changes
    old = subprocess.run(['python3', '-m', 'tools.m8i0_verify'],
                         capture_output=True, text=True, check=True)
    old_result = json.loads(old.stdout)
    print(json.dumps(dict(
        kind='ActualOfflineArtifactVerification_NotSimulation', status='Passed',
        manifest_sha256=sha(manifest_path), sealed_files=len(verified),
        required_deliverables=17, executed_source_hashes=len(result['source_sha256']),
        preflight_files=len(frozen['files']), declared_postrun_test_extension=test_delta,
        preserved_m8i0_files=old_result['review_files'],
        historical_entries=old_result['historical_manifest_entries'],
        base_tracked_modifications=0, failed_attempt_status_consistent=True,
        live_CARLA_connections=0, model_forwards=0, training_steps=0,
        actual_data_collection='NotRun', next_session_authorized=False,
        command='python3 -m tools.m8i1_verify'), indent=2))


if __name__ == '__main__':
    main()
