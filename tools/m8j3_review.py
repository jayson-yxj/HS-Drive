"""Offline M8-J3 provenance audit. No CARLA client, network, model or dataset load."""
import argparse
import ast
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'c4679352669e3a2b3add0ccb0fcc05282e9285d6'
INSTALL = Path('/root/gpufree-data/simulators/CARLA_0.9.15')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def asset(path):
    p = Path(path).resolve()
    return dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                bytes=p.stat().st_size, sha256=sha(p))


def write(name, data):
    with (ROOT / name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def local_import_graph(roots):
    todo = [ROOT / p for p in roots]
    seen, external = {}, set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen[p] = asset(p)
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    parent = p.parent
                    for _ in range(n.level - 1):
                        parent = parent.parent
                    prefix = '.'.join(parent.relative_to(ROOT).parts)
                    names = [prefix + ('.' + n.module if n.module else '')]
                else:
                    names = [n.module or '']
            else:
                continue
            for name in names:
                candidate = ROOT.joinpath(*name.split('.')).with_suffix('.py')
                package = ROOT.joinpath(*name.split('.'), '__init__.py')
                if candidate.is_file():
                    todo.append(candidate)
                elif package.is_file():
                    todo.append(package)
                else:
                    external.add(name)
    return sorted(seen.values(), key=lambda x: x['path']), sorted(external)


def collect():
    historical = json.loads((ROOT / 'reports/m8h0_map_identity_and_runtime.json').read_text())
    installed = []
    for old in historical['packaged_files']:
        row = asset(old['path'])
        row['matches_M8H0_bytes'] = row['sha256'] == old['sha256']
        installed.append(row)
    for rel in ['VERSION', 'LICENSE', 'README', 'CarlaUE4.sh',
                'PythonAPI/examples/generate_traffic.py',
                'CarlaUE4/Content/Carla/Blueprints/Vehicles/Tesla/BP_TeslaM3.uasset',
                'CarlaUE4/Content/Carla/Blueprints/Vehicles/Tesla/BP_TeslaM3.uexp']:
        p = INSTALL / rel
        installed.append(asset(p) if p.exists() else dict(path=str(p), status='NotAvailable'))
    roots = ['tools/record_dataset.py', 'simulator/m8i1_session.py',
             'datasets/m8i1_contract.py', 'datasets/m48_evidence.py',
             'geometry/oracle_bev.py', 'tools/m8i1a_identity.py']
    graph, external = local_import_graph(roots)
    forbidden = ('bench2drive', 'leaderboard', 'srunner', 'think2drive')
    forbidden_imports = [s for s in external if any(x in s.lower() for x in forbidden)]
    versions = {}
    for name in ['carla', 'numpy', 'opencv-python', 'PyYAML', 'pytest']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'NotInstalledInInspectionInterpreter'
    write('reports/m8j3_sources/local_carla_inventory.json', dict(
        checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        kind='ActualServerReadOnlyFilesystemAndStaticASTAudit', base_commit=BASE,
        inspector=asset(__file__), package_version=(INSTALL / 'VERSION').read_text().strip(),
        python_distributions=versions, actual_server_RPC='NotRun',
        installed=installed, import_roots=roots, static_local_import_closure=graph,
        external_import_names=external, prohibited_dependency_imports=forbidden_imports,
        limitations=['Static imports do not prove original authorship or absence of dynamic imports.',
                     'Installed version marker and old hashes are not current world/RPC readiness.',
                     'Selected package hashes do not authenticate every mesh or its third-party rights.'],
        CARLA_connections=0, server_launches=0, new_actors=0, new_sensor_data=0))


def check_rows(rows):
    for row in rows:
        p = ROOT / row['path']
        if not p.is_file() or sha(p) != row['sha256'] or p.stat().st_size != row['bytes']:
            raise RuntimeError('Evidence changed: ' + str(p))


def verify():
    freeze = json.loads((ROOT / 'reports/m8j3_historical_freeze.json').read_text())
    check_rows(freeze['files'])
    subprocess.run(['python3', '-m', 'tools.m8j2_finalize', 'verify'], cwd=ROOT, check=True)
    archive = json.loads((ROOT / 'reports/m8j3_sources/paper_archive.json').read_text())
    check_rows([archive['local_original']])
    for name in ['license_commit.patch', 'readme_commit.patch', 'carla_content_versions.txt']:
        meta = json.loads((ROOT / ('reports/m8j3_sources/' + name + '.meta.json')).read_text())
        check_rows([dict(path='reports/m8j3_sources/' + name, **{k: meta[k] for k in ['bytes', 'sha256']})])
    inventory = json.loads((ROOT / 'reports/m8j3_sources/local_carla_inventory.json').read_text())
    check_rows([r for r in inventory['installed'] if 'sha256' in r])
    check_rows(inventory['static_local_import_closure'])
    assert inventory['package_version'] == '0.9.15'
    assert not inventory['prohibited_dependency_imports']
    assert all(r.get('matches_M8H0_bytes', True) for r in inventory['installed'])
    a = json.loads((ROOT / 'reports/m8j3_acceptance.json').read_text())
    assert a['bench2drive_license'] == 'LicenseConflictUnresolved'
    assert a['training'] == 'TrainingReadinessNoGo'
    assert a['payload'] == 'SingleClipHashAndSchemaVerified_PartialLabelQuality'
    assert a['maintainer_contact'] == 'NotSent'
    assert not a['CARLA_runtime_authorized'] and a['download_authorized']
    private_index = json.loads((ROOT / 'reports/m8j3_private_artifact_index.json').read_text())
    check_rows(private_index['files'])
    for row in private_index['files']:
        assert not Path(row['path']).resolve().is_relative_to(ROOT)
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE], cwd=ROOT, text=True).splitlines()
    assert all('m8j3' in p.lower() for p in changed), changed
    assert not any(Path(p).suffix.lower() in {'.jpg', '.jpeg', '.png', '.npz', '.gz', '.pth', '.pt', '.pdf'} for p in changed)
    manifest = ROOT / 'reports/m8j3_release_manifest.json'
    if manifest.exists():
        check_rows(json.loads(manifest.read_text())['files'])
    print(json.dumps(dict(status='OfflineEvidenceVerified', historical_J2_files=len(freeze['files']),
                          local_sources=len(inventory['static_local_import_closure']),
                          selected_installed_files=len(inventory['installed']),
                          payload_verified=True, label_quality='Partial', runtime_tested=False)))


def seal():
    files = []
    for folder in ['reports', 'docs', 'tools', 'datasets', 'tests']:
        for p in (ROOT / folder).rglob('*'):
            if (p.is_file() and 'm8j3' in str(p.relative_to(ROOT)).lower()
                    and p.name != 'm8j3_release_manifest.json' and '__pycache__' not in str(p)):
                files.append(asset(p))
    write('reports/m8j3_release_manifest.json', dict(base_commit=BASE,
          sealed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
          files=sorted(files, key=lambda x: x['path']),
          source_commit='Use git log -1 --format=%H -- reports/m8j3_release_manifest.json',
          self_hash='Git commit binds this manifest; no circular self hash.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'verify', 'seal'])
    globals()[parser.parse_args().action]()
