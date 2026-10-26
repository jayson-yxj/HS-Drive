"""Read-only M8-J0 release verification; no CARLA imports or network calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'fe8db837d18f033e446475d0bf48534eb54bc0f3'


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def main():
    manifest = read(ROOT/'reports/m8j0_release_manifest.json')
    assert manifest['base_commit']==BASE
    for r in manifest['files']:
        p=ROOT/r['path']; assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
        if p.suffix=='.json': read(p)
        if p.suffix=='.py': ast.parse(p.read_text())
        if p.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                if '://' not in link and not link.startswith('#'):
                    assert (p.parent/link.split('#')[0]).exists(),(p,link)
    for r in read(ROOT/'reports/m8j0_historical_freeze.json')['files']:
        assert sha(r['path'])==r['sha256'],r['path']
    for r in read(ROOT/'reports/m8j0_source_inventory.json')['files']:
        p=ROOT/r['path'];b=p.read_bytes()
        assert sha(p)==r['sha256']
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['git_blob_sha']
    acc=read(ROOT/'reports/m8j0_acceptance.json')
    assert acc['decision']=='AccessOrLicenseNoGo' and acc['status']=='LicenseAndAccessReviewCompleted'
    assert not acc['single_clip_review_proposal_eligible'] and not acc['download_approved']
    assert acc['payload_gate']=='PayloadNotVerified' and acc['training_readiness']=='TrainingReadinessNoGo'
    assert acc['server_launches']==acc['GPU_tasks']==acc['training_steps']==acc['payload_get_requests']==0
    assert set(acc['gates'])=={'A_License','B_Access','C_Identity','D_Schema','E_Time','F_Cost','G_NextStep'}
    identity=read(ROOT/'reports/m8j0_official_asset_identity.json')
    assert identity['expected_bytes']==280176944
    assert identity['expected_sha256']=='b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2'
    assert identity['target_file_revision'] is None and identity['target_lfs_sha256'] is None
    access=read(ROOT/'reports/m8j0_official_access_probe.json')
    requests=access['server_requests'];assert len(requests)==7 and access['metadata_body_bytes']==39060
    versions={sha(p) for p in (ROOT/'reports/m8j0_sources').glob('probe_executed*.py.txt')}
    from tools.m8j0_probe import validate
    for r in requests:
        validate(r['method'],r['url'])
        assert r['source_sha256'] in versions
        assert r['tls_verification'] and not r['range_request'] and not r['automatic_retries']
        if r['saved_body_bytes']:
            p=ROOT/r['body_path'];assert sha(p)==r['body_sha256'] and p.stat().st_size==r['saved_body_bytes']
        if '.tar.gz' in r['url']: assert r['method']=='HEAD' and r['saved_body_bytes']==0
    assert len([r for r in requests if r['category']=='ConnectionFailure'])==2
    schema=read(ROOT/'reports/m8j0_source_schema_contract.json')
    assert len(schema['cameras'])==6 and not schema['top_down_is_inference_input']
    assert all(x['payload_verification']=='PayloadNotVerified' for x in schema['field_evidence'])
    for field in ('source_world_frame','per_sensor_timestamp','source_sensor_frame'):
        x=next(x for x in schema['field_evidence'] if x['field']==field)
        assert x['evidence_level']=='NotAvailable'
    changed=subprocess.check_output(['git','diff',BASE,'--name-status','--diff-filter=MDTR'],cwd=ROOT,text=True)
    assert not changed.strip(),changed
    for args in (['python3','-m','tools.m8i1b_verify'],['python3','-m','unittest','tests.test_m8j0_review','-q']):
        p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
        assert p.returncode==0,(args,p.stdout,p.stderr)
    print(json.dumps(dict(status='Passed',kind='ActualOfflineReviewReleaseVerification',
        decision=acc['decision'],manifest_files=len(manifest['files']),
        source_blobs=len(read(ROOT/'reports/m8j0_source_inventory.json')['files']),
        historical_306_assets='Passed',historical_supplement='Preserved',request_boundary_tests=8,
        server_metadata_requests=7,payload_requests=0,old_files_modified=0,
        network_or_CARLA_connections_in_verifier=0,manifest_sha256=sha(ROOT/'reports/m8j0_release_manifest.json')),indent=2))


if __name__=='__main__': main()
