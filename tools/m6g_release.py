"""Verify historical bytes and seal M6-G; never modifies input assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tools.m6g_analyze import read, write, asset


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(expected):
    bad = [p for p, h in expected.items() if not Path(p).is_file() or digest(p) != h]
    if bad:
        raise RuntimeError('Preservation mismatch: ' + str(bad))
    return len(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    base = read('reports/m6g_execution_logs/base_freeze.json')
    checks = dict(base_tracked=verify(base['files']),
                  inherited_external_assets=verify(read('runs/m6b_logs/freeze.json')['assets']),
                  official_0916_binary_files=verify(read('reports/m6f_execution_logs/server_binary_manifest.json')['files']),
                  M6F_release_assets=verify({a['path']: a['sha256'] for a in read('reports/m6f_release_manifest.json')['files']}))
    if not args.seal:
        write('reports/m6g_execution_logs/preservation.json', dict(
            kind='ActualServerSHA256Verification', status='Passed', checks=checks,
            base=base['base'], changed=[], scope='Inventoried assets only; no claim of unfrozen external files'))
    else:
        files = []
        for root in ['reports', 'docs', 'tools', 'diagnostics', 'tests', 'configs/m6g']:
            for p in Path(root).rglob('*'):
                if p.is_file() and '__pycache__' not in str(p) and 'm6g' in str(p).lower() and p.name != 'm6g_release_manifest.json':
                    files.append(asset(p))
        write('reports/m6g_release_manifest.json', dict(
            kind='ActualServerAppendOnlyReleaseInventory', base=base['base'], checks=checks,
            artifact_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            files=sorted(files, key=lambda a: a['path']),
            self_reference='Manifest excludes itself; final seal commit contains manifest. Individual execution commits are in reports.',
            canonical_acquisition='reports/m6f_execution_logs/trial_manifest.json',
            historical_server_binary='reports/m6f_execution_logs/server_binary_manifest.json',
            new_build_SHA=None, new_build_status='NotRun'))
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
