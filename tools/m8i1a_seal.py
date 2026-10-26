"""Freeze offline deliverables and separately reviewable future launch inputs."""
from pathlib import Path
import json
from tools.m8i1a_common import REPO, PROTOCOL, asset, persist, provenance


def main():
    root = Path('/root/gpufree-data/hs-drive-experiments/m8i1a-offline-evidence')
    sources = [PROTOCOL, *REPO.glob('tools/m8i1a_*.py'), *REPO.glob('tests/*m8i1a*.py')]
    audit = json.loads((REPO / 'reports/m8i1a_static_launch_audit.json').read_text())
    resources = [Path(x['path']) for x in audit['resource_files']]
    resources += [Path(audit['launcher']['path']), Path(audit['old_start_script']['path'])]
    resources += [Path(x['path']) for x in audit['inspected_configuration']]
    resources += [REPO / 'reports/m8i1a_failure_injection_tests.json']
    pre = REPO / 'reports/m8i1a_prelaunch_freeze.json'
    persist(pre, dict(**provenance('OfflinePrelaunchFreeze_NotAuthorization'),
                      files=[asset(p) for p in sorted(set(sources + resources))],
                      runtime_authorized=False, planned_server_launches=1))
    paths = set(sources + [pre])
    paths.update(REPO.glob('reports/m8i1a_*.json'))
    paths.update(REPO.glob('reports/m8i1a_*.md'))
    paths.update(REPO.glob('docs/M8I1A_*.md'))
    paths.update(p for p in root.rglob('*') if p.is_file())
    persist(REPO / 'reports/m8i1a_release_manifest.json', dict(
        **provenance('OfflineReleaseSeal_NoRuntime'), files=[asset(p) for p in sorted(paths)],
        self_excluded=True, verification_receipt_excluded='reports/m8i1a_verification_receipt.json',
        scope='Versioned offline artifacts; synthetic process records are not CARLA identity',
        runtime_authorized=False))
    print(f'Sealed {len(paths)} files; CARLA NotRun')


if __name__ == '__main__':
    main()
