"""Offline M8-J1 evidence synthesis and release verification. No network operations."""
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
SRC=ROOT/'reports/m8j1_sources'
BASE='307f4fb34b501c6400f20e4e437963429527a333'
CLIP='DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz'
CODE='7ec25d1c9f7522d923ce5f3420986cef1cb2d956'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')


def build():
    now=dt.datetime.now(dt.timezone.utc)
    common=dict(stage='M8-J1',base_commit=BASE,checked_at_utc=now.isoformat(),
        checked_at_local=now.astimezone(ZoneInfo('Europe/Berlin')).isoformat(),timezone='Europe/Berlin',
        evidence_kind='ActualServerBoundedMetadataAndOfflineReview',generator_sha256=sha(__file__),
        payload_downloads=0,payload_get_requests=0,range_requests=0,archive_extractions=0,
        CARLA_connections=0,server_launches=0,GPU_tasks=0,training_steps=0,
        external_messages_sent=0,old_files_modified=0)
    def report(name,**fields):write(ROOT/f'reports/m8j1_{name}.json',dict(common,**fields))
    req=[read(p) for p in sorted(SRC.glob('*.request.json'))]
    network={host:read(SRC/(host+'.network.json')) for host in ('huggingface.co','api.github.com')}
    assert len(req)==5 and read(SRC/'batch_finished.json')['batch_count']==1
    matches=[]
    for name,suffix in [('github_fixed_license','LICENSE'),('github_fixed_readme','README.md')]:
        d=read(SRC/(name+'.body'));b=base64.b64decode(d['content'])
        p=ROOT/('reports/m8i0_sources/b2dapi__'+suffix+'.txt')
        assert b==p.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==d['sha']
        matches.append(dict(name=name,bytes=len(b),git_blob_sha=d['sha'],sha256=sha(p),
            frozen_path=str(p.relative_to(ROOT)),url=d['html_url'],matched=True))
    target=read(ROOT/'reports/m8j0_official_asset_identity.json')
    search=read(SRC/'github_license_discussion_search.body')
    report('official_access_evidence',status='OfficialMetadataUnavailable',scope='HF target on workspace GPU server',
        diagnostic_batches=1,batch_closed=True,no_additional_batch_authorized=True,
        limits=read(ROOT/'configs/m8j1_review_protocol.json'),network_layers=network,
        requests=req,http_request_count=len(req),metadata_body_bytes=sum(x['body_bytes'] for x in req),
        github_fixed_sources=matches,
        proxy_observation=read(SRC/'batch_started.json')['proxy_env_present'],
        routing_limitations='Direct socket sample and curl are separate probes. No explicit proxy environment observed; no claim about transparent routing or authoritative DNS correctness. No resolver, route, proxy or timeout configuration changed.',
        tls_interpretation='HF TLS never reached. Curl ssl_verify_result=0 without time_appconnect/connection is not certificate verification success. GitHub separate TLSv1.3 and successful HTTPS support TLS only there.',
        failure_comparison=dict(M8_J0='curl7 connection refused before HTTP',M8_J1='curl28 connection timeout before HTTP',
            not_supported=['Authentication failure','404/object missing','TLS certificate rejection','Permanent access prohibition','Proven DNS poisoning']),
        target=dict(filename=CLIP,expected_bytes=target['expected_bytes'],expected_sha256=target['expected_sha256'],
            official_manifest_commit=CODE,official_manifest_url=target['official_manifest_url'],
            official_manifest_sha256=target['manifest_sha256'],manifest_verification='FrozenJ0BytesReverifiedOffline',
            source_repo='rethinklab/Bench2Drive',expected_path=CLIP,
            attempted_revision='ce56cfbc26edf7796586e26fd44644f0ac2f6bcc',revision_status='CandidateFromPriorBrowserMetadata_NotTargetMapping',
            current_repo_revision=None,file_revision=None,lfs_sha256=None,xet_hash=None,storage_endpoint=None,
            hosted_size=None,content_length=None,etag=None,official_object_to_manifest_binding='Unresolved',
            anonymous_access='NotEstablished',login_required='NotDetermined',additional_terms='NotDetermined',
            redirect_chain=[],payload_status='PayloadNotVerified'),
        api_locate_target='NotReached: repository metadata request failed; no claim file absent',
        skipped_conditional_requests=['Pinned HF metadata API','Pinned HF README'],
        skip_reason='HF repository metadata failed at transport stage; no redundant follow-up requests',
        separate_environment=dict(kind='BrowserResearchService',document_requests=3,browser_calls=1,
            checked_date_utc='2026-10-26',timestamp_precision='DateOnly; exact browser backend request time not available',
            hf_card='Readable cached official page: Apache-2.0 metadata and prose',
            hf_target_blob_metadata_page='ToolInternalError_HTTPStatusUnknown',
            hf_discussions='Cached official page: no posts shown',
            payload_read=False,capture='reports/m8j1_sources/browser_batch_1.json',
            not_evidence_of_gpu_server_recovery=True),
        official_alternative_routes=[
            dict(route='Hugging Face official CLI/direct resolver',source=target['path_evidence'],
                status='SourceDocumented_NotExecutableVerified',reason='Same official host; no CLI installation or file transfer; repository access unresolved'),
            dict(route='Official README Base Baidu share',source=target['official_manifest_url'].replace('docs/bench2drive_mini_10.json','README.md'),
                domain='pan.baidu.com',share_path='/s/1ZIL-MPhLbgdBYmHkHncn8Q',status='SourceDocumented_TargetNotLocated_NotProbed',
                reason='Base-level link does not prove target file identity, revision, current availability or license; no login/link probing or download'),
            dict(route='Official mirror suggestion',status='NotUsed',reason='Recommendation alone not enough to establish a trusted official storage endpoint; no mirror used')],
        bounded_maintainer_statement_search=dict(url=next(x['url'] for x in req if x['name']=='github_license_discussion_search'),
            result_count=search['total_count'],incomplete_results=search['incomplete_results'],
            hits=[dict(number=x['number'],title=x['title'],url=x['html_url'],disposition='UnrelatedOSPermissionError_NotAssetLicense') for x in search['items']],
            asset_specific_resolution_found=False,scope='Search results and issue body text; no new issue/comment posted'))
    report('acceptance',decision='LicenseOrAccessStillUnresolved',decision_option='B',overall='Partial',
        bounded_review='Completed',communication='FinalDraftReady_NotSent',
        reason='One bounded license verification and metadata batch completed; no asset-level reconciliation or executable official access path established',
        why_not_A='Asset license, hosted identity and executable legitimate access are still unclosed',
        why_not_C='No affirmative evidence of incompatible usage or permanent lawful-access impossibility; network failure is insufficient',
        license_status='LicenseConflictUnresolved',gpu_server_access='OfficialMetadataUnavailable',
        payload_status='PayloadNotVerified',training_status='TrainingReadinessNoGo',
        asset_identity=dict(filename=CLIP,expected_bytes=target['expected_bytes'],expected_sha256=target['expected_sha256'],official_manifest_commit=CODE),
        substantive_license_progress=False,substantive_access_progress=False,
        diagnostic_progress='Resolved vs TCP/TLS/HTTP layers distinguished; timeout differs from prior refusal',
        download_proposal_eligible=False,download_authorized=False,schema_application_refined=False,
        maintainer_send_authorized=False,maintainer_sent_at=None,maintainer_reply=None,
        payload_or_schema_execution='NotRun',stop_downloader_development=True,
        no_further_automatic_network_batches=True,
        next_decision='Separately scoped review of small public sources with explicit acquisition terms, such as nuScenes Mini matching PETR pretraining; not started or approved here',
        historical_states_preserved=read(ROOT/'reports/m8j1_historical_freeze.json')['preserved_states'],
        original_CARLA_startup_authorization_consumed=1,new_CARLA_authorization=0,
        external_contact_note='Task permits draft only until independent explicit sending authorization; no maintenance response is pending because nothing was sent')


def verify():
    for r in read(ROOT/'reports/m8j1_historical_freeze.json')['files']:
        p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
    for args in (['python3','-m','tools.m8j0_verify'],['python3','-m','unittest','tests.test_m8j1_metadata_boundaries','-v']):
        r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);assert r.returncode==0,(args,r.stdout,r.stderr)
    source_sha=sha(ROOT/'tools/m8j1_metadata_batch.py')
    start=read(SRC/'batch_started.json');end=read(SRC/'batch_finished.json')
    assert start['source_sha256']==source_sha and start['protocol_sha256']==sha(ROOT/'configs/m8j1_review_protocol.json')
    assert end['batch_count']==1 and end['http_requests']==5 and end['pending_requests']==0
    from tools.m8j1_metadata_batch import validate
    total=0
    for p in SRC.glob('*.request.json'):
        r=read(p);validate(r['method'],r['url']);total+=r['body_bytes']
        assert r['source_sha256']==source_sha and not r['range_request'] and r['tls_verify']
        assert r['retries']==0 and not r['follow_redirects']
        if r['body_bytes']:assert sha(ROOT/r['body_path'])==r['body_sha256']
        if '.tar.gz' in r['url']:assert r['method']=='HEAD' and r['body_bytes']==0
    assert total==74176
    a=read(ROOT/'reports/m8j1_acceptance.json')
    assert a['generator_sha256']==sha(__file__)
    assert a['decision']=='LicenseOrAccessStillUnresolved' and a['decision_option']=='B'
    assert not a['download_authorized'] and not a['download_proposal_eligible'] and not a['schema_application_refined']
    assert a['maintainer_sent_at'] is None and a['maintainer_reply'] is None
    assert a['external_messages_sent']==a['server_launches']==a['payload_downloads']==a['training_steps']==0
    diff=subprocess.check_output(['git','diff',BASE,'--name-status','--diff-filter=MDTR'],cwd=ROOT,text=True)
    assert not diff.strip(),diff
    manifest=ROOT/'reports/m8j1_release_manifest.json'
    if manifest.exists():
        for r in read(manifest)['files']:
            p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
    required=['summary.md','acceptance.json','license_clarification.md','official_access_evidence.json',
              'maintainer_contact_record.md','source_decision.md']
    assert all((ROOT/('reports/m8j1_'+name)).is_file() for name in required)
    assert (ROOT/'docs/M8J1_ARCHITECTURE_DECISION.md').is_file()
    for p in list((ROOT/'reports').glob('m8j1_*.md'))+list((ROOT/'docs').glob('M8J1*.md')):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in link and not link.startswith('#'):assert (p.parent/link.split('#')[0]).exists(),(p,link)
    print(json.dumps(dict(status='Passed',kind='OfflineEvidenceAndSyntheticBoundaryVerification',
        historical_J0_files=44,historical_306_assets='VerifiedViaUnchangedJ0Verifier',
        boundary_tests=5,batches=1,http_requests=5,metadata_bytes=total,
        payload_requests=0,old_modifications=0,decision=a['decision']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['build','verify']);a=p.parse_args()
    build() if a.operation=='build' else verify()
