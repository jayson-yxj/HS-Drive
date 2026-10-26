"""Offline reports and evidence verification for M8-J2. No network or model import."""
import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'reports/m8j2_sources'
BASE='5d6252a6f5a536d29cce5d2fb756134dec4057fe'
CODE='7ec25d1c9f7522d923ce5f3420986cef1cb2d956'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')


def build():
    now=dt.datetime.now(dt.timezone.utc)
    common=dict(stage='M8-J2',base_commit=BASE,checked_at_utc=now.isoformat(),
        local_timezone='Europe/Berlin',checked_at_local=now.astimezone(ZoneInfo('Europe/Berlin')).isoformat(),
        source_sha256=sha(__file__),kind='ActualServerMetadataReview_OfflineSynthesis',
        payload_get_requests=0,range_requests=0,dataset_downloads=0,extractions=0,
        CARLA_connections=0,server_launches=0,GPU_tasks=0,inference_calls=0,training_steps=0,
        maintainer_messages_sent=0,main_merge=False,old_assets_modified=0)
    def report(name,**fields):write(ROOT/f'reports/m8j2_{name}.json',dict(common,**fields))
    rows=[read(p) for p in sorted(SRC.glob('http_*.json'))]
    identity=read(SRC/'repository_identity.json');hub=read(SRC/'hub_repo.json');meta=read(SRC/'hub_file_metadata.json')
    resolve=read(SRC/'http_07.json');storage=read(SRC/'http_08.json');git=read(SRC/'git_proxy.json')
    manifest=read(ROOT/'reports/m8j0_official_asset_identity.json')
    target=identity['target']
    assert target['size']==target['lfs']['size']==meta['size']==int(storage['headers']['content-length'])==manifest['expected_bytes']
    assert target['lfs']['sha256']==meta['etag']==manifest['expected_sha256']
    assert identity['revision']==hub['sha']==meta['commit_hash']==git['remote_head'].split()[0]
    report('official_asset_identity',status='OfficialMetadataIdentityMatched',repository='rethinklab/Bench2Drive',
        filename=target['rfilename'],revision=identity['revision'],git_head=git['remote_head'].split()[0],
        public_repository=not identity['private'],gated=identity['gated'],anonymous_metadata=True,
        bytes=target['size'],official_expected_sha256=manifest['expected_sha256'],
        lfs_sha256=target['lfs']['sha256'],lfs_pointer_git_blob=target['blobId'],lfs_pointer_bytes=target['lfs']['pointerSize'],
        xet_hash=resolve['headers']['x-xet-hash'],storage_etag=storage['headers']['etag'],
        storage_host='us.aws.cdn.hf.co',storage_head_http_status=storage['http_status'],
        storage_content_type=storage['headers']['content-type'],storage_content_length=int(storage['headers']['content-length']),
        official_manifest_commit=CODE,official_manifest_url=manifest['official_manifest_url'],
        official_manifest_sha256=manifest['manifest_sha256'],manifest_match=True,
        evidence_chain=['J0 fixed official Mini manifest (rehash unchanged)','requests anonymous dataset API','huggingface_hub pinned dataset API',
            'Git ls-remote HEAD','HF resolver HEAD302 with x-repo-commit/x-linked-etag/x-linked-size/x-xet-hash',
            'Only official HF-issued storage URL HEAD200'],
        hash_semantics=dict(lfs='Official SHA256 of target archive, matches frozen manifest; archive bytes not independently hashed',
            pointer='Git blob identity of LFS pointer, not archive content hash',xet='Storage representation hash, not substituted for archive SHA256'),
        revision_scope='Repository snapshot containing target; not claiming last modifying commit of the individual object was independently queried',
        release='Legacy Mini subset of Base, not v0.0.4 new data/depth or Dev10 routes',
        payload_status='PayloadNotVerified',transfer_integrity='NotTested',
        license_files_listed=identity['license_files'],license_files_inside_archive='PayloadNotVerified')
    report('network_proxy_audit',access_status='MetadataAccessResolved_WithExplicitProxyOnly',
        discovery=read(SRC/'proxy_discovery.json'),dns=read(SRC/'dns.json'),
        inherited_shell=dict(proxy_env='All eight uppercase/lowercase proxy/NO_PROXY entries unset',
            tun_config=False,requests_result='ConnectionError: network unreachable; no HTTP response',
            git_proxy_config_entries=0,git_default_network_probe='NotRun',hub_default_network_probe='NotRun',
            conclusion='Proxy process active does not mean all shell programs inherit it'),
        explicit_process_scope=dict(proxy='http://127.0.0.1:7897',type='HTTP CONNECT via discovered mixed port',
            requests='Session trust_env=False with explicit http/https proxies; successful loopback peer andTLSv1.3',
            huggingface_hub='0.26.5 configure_http_backend with same audited session; pinned API200 and targetHEAD302',
            git='Per-process HTTP(S)_PROXY and NO_PROXY empty; trace confirms loopback7897, CONNECT, certificate verification and200; ls-remote only',
            git_no_clone=True,global_shell_or_git_config_modified=False,
            same_public_exit='NotMeasured; rule-mode proxy may route each host differently'),
        requests=rows,python_http_transactions=len(rows),python_metadata_wire_body_bytes=sum(r['body_bytes'] for r in rows),
        git_proxy=git,git_http_transactions_observed=len(git['http_methods_observed']),
        total_application_http_transactions=len(rows)+len(git['http_methods_observed']),
        target_or_storage_head_count=sum(r['method']=='HEAD' for r in rows),
        github_result='HTTP403 anonymous API rate limit, not TLS/authentication failure or data license refusal; no retry/change of identity to bypass',
        failure_records=dict(initial_local_assertion='Preserved hf_pipeline_failure.json: URL canonicalization mismatched exact signed redirect whitelist before storage send',
            completion='One bounded completion of unfinished checks; same target HEAD reissued once to obtain ephemeral URL, then one storage HEAD; automatic retries0',
            no_remaining_network_actions=True),
        credential_redaction=dict(proxy_credentials_logged=False,signed_storage_query='REDACTED',
            git_full_trace_not_retained=True,github_error_body='Egress IP redacted; original response SHA retained; no raw body retained'),
        historical_causality='Current paired default-vs-explicit result supports missing effective proxy configuration as present access difference. It does not retroactively prove every M8-J0/J1 failure had only that cause.',
        dependencies='reports/m8j2_environment.json; four small packages installed only in isolated metadata venv, official PyPI, distinct from target dataset payload')
    github_license=ROOT/'reports/m8i0_sources/b2dapi__LICENSE.txt'
    github_readme=ROOT/'reports/m8i0_sources/b2dapi__README.md.txt'
    hf_card=SRC/'http_06.body'
    report('license_evidence',status='LicenseConflictUnresolved',asset_level_conditions='LicenseScopePartial',
        github=dict(commit=CODE,license='CC-BY-NC-ND-4.0',readme_scope='All assets and code unless otherwise specified',
            license_sha256=sha(github_license),readme_sha256=sha(github_readme),
            evidence='M8-J0/J1 fixed official bytes, rehashed unchanged; fresh J2 API403 rate limited, not a fresh200 retrieval'),
        hf=dict(revision=identity['revision'],card_sha256=sha(hf_card),card_bytes=hf_card.stat().st_size,
            actual_server_http_status=200,metadata_license='apache-2.0',prose_license='Apache2.0 for assets and code unless specified otherwise',
            source_url=read(SRC/'http_06.json')['url'],no_extra_license_names_in_returned_siblings=not identity['license_files']),
        asset=dict(filename=target['rfilename'],object_identified=True,
            repo_card_now_bound_to_same_revision=True,repo_card_is_not_per_archive_terms_review=True,
            archive_NOTICES='PayloadNotVerified',third_party_restrictions='Unresolved; existing CARLA code/assets/engine distinctions retained'),
        maintainer_contact='Existing finalized J1 draft retained; no send and no new reply',
        no_automatic_permission_from_http200=True,not_all_internal_research_prohibited=True,
        next_required='Asset-specific official reconciliation for proposed uses or clear applicable exception; do not auto-pick more permissive statement')
    report('acceptance',decision='AccessResolvedLicenseUnresolved',decision_option='B',overall='Partial',
        access='Passed_MetadataOnly_ExplicitProxy',asset_identity='Passed_OfficialMetadataMatch',
        license='LicenseConflictUnresolved',payload='PayloadNotVerified',training='TrainingReadinessNoGo',
        next_stage_proposal_eligible=False,download_authorized=False,model_adaptation_authorized=False,
        source_unavailable_claim=False,schema_or_dataset_engineering_started=False,
        why_not_A='Asset-level license conflict and third-party scope remain unresolved',
        why_not_C='Official anonymous APIs, pinned resolver and issued storage HEAD succeeded via explicit proxy',
        why_not_D='Target exists at verified revision with matching metadata; no affirmative unsuitable-use finding',
        limits=['Metadata availability now verified, not whole-file transfer/integrity','Default shell still lacks proxy environment',
                'No raw frame/timestamp/identity payload validation','No training, inference, CARLA or safety certification'],
        historical_freeze_files=len(read(ROOT/'reports/m8j2_historical_freeze.json')['files']),
        historical_m8j0_m8j1_statuses_preserved=True,old_CARLA_authorization_consumed=1,new_CARLA_authorization=0)


def verify():
    for r in read(ROOT/'reports/m8j2_historical_freeze.json')['files']:
        p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
    for args in (['python3','-m','tools.m8j1_finalize','verify'],['python3','-m','unittest','tests.test_m8j2_access','-v']):
        r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);assert r.returncode==0,(r.stdout,r.stderr)
    source_versions={sha(ROOT/'tools/m8j2_access_probe.py'),sha(SRC/'probe_executed_initial.py.txt'),sha(SRC/'probe_executed_completion.py.txt')}
    rows=[read(p) for p in sorted(SRC.glob('http_*.json'))];assert len(rows)==9
    for r in rows:
        assert r['source_sha256'] in source_versions
        assert r['tls_verify'] and not r['range_request'] and r['retries']==0
        if '.tar.gz' in r['url'] or 'xet-bridge' in r['url']:assert r['method']=='HEAD' and r['body_bytes']==0
        if r['body_bytes']:assert sha(ROOT/r['body_path'])==r['body_sha256']
    a=read(ROOT/'reports/m8j2_acceptance.json');assert a['decision_option']=='B' and a['decision']=='AccessResolvedLicenseUnresolved'
    assert not a['download_authorized'] and a['training']=='TrainingReadinessNoGo'
    asset=read(ROOT/'reports/m8j2_official_asset_identity.json')
    assert asset['bytes']==280176944 and asset['lfs_sha256']==asset['official_expected_sha256']
    assert asset['storage_head_http_status']==200 and asset['manifest_match']
    assert read(SRC/'git_proxy.json')['proxy_connected'] and read(SRC/'git_proxy.json')['certificate_verified']
    assert not subprocess.check_output(['git','diff',BASE,'--name-status','--diff-filter=MDTR'],text=True).strip()
    # Privacy audit: query strings on signed storage links cannot appear in published evidence.
    for p in [*(ROOT/'reports').glob('m8j2*.json'),*SRC.glob('*.json'),*SRC.glob('*.body')]:
        s=p.read_text();assert 'Signature=' not in s and 'X-Amz-Signature=' not in s,p
        if p.suffix=='.body' and 'rate limit exceeded for' in s:assert '<egress_ip_redacted>' in s,p
    for p in [*(ROOT/'reports').glob('m8j2*.md'),*(ROOT/'docs').glob('M8J2*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in link and not link.startswith('#'):assert (p.parent/link).exists(),(p,link)
    manifest=ROOT/'reports/m8j2_release_manifest.json'
    if manifest.exists():
        for r in read(manifest)['files']:
            p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
    print(json.dumps(dict(status='Passed',kind='ActualOfflineArtifactVerificationAndSyntheticBoundaries',
        historical_files=73,boundary_tests=7,metadata_http_records=9,git_advertisement_requests=1,
        payload_requests=0,old_files_modified=0,decision=a['decision']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['build','verify']);a=p.parse_args()
    build() if a.operation=='build' else verify()
