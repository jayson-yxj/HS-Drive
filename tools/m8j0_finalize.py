"""Build M8-J0 review from frozen public metadata and source. Offline, append-only."""
import ast
import base64
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'reports/m8i0_sources'
SRC = ROOT / 'reports/m8j0_sources'
BASE = 'fe8db837d18f033e446475d0bf48534eb54bc0f3'
CODE = '7ec25d1c9f7522d923ce5f3420986cef1cb2d956'
CLIP = 'DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz'
GH = 'https://github.com/Thinklab-SJTU/Bench2Drive/blob/' + CODE + '/'
HF = 'https://huggingface.co/datasets/rethinklab/Bench2Drive'
HF_README_COMMIT = 'ce56cfbc26edf7796586e26fd44644f0ac2f6bcc'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, value):
    with Path(p).open('x') as f:
        if isinstance(value, str): f.write(value.rstrip() + '\n')
        else: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def frozen_source(name):
    p = OLD / ('b2dapi__' + name.replace('/', '__') + '.txt')
    b = p.read_bytes(); m = read(str(p) + '.meta.json')
    assert sha(p) == m['sha256']
    assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == m['git_blob_sha']
    assert ('ref=' + CODE) in m['url']
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), git_blob_sha=m['git_blob_sha'],
                official_url=GH + name, originally_fetched_at_utc=m['checked_at_utc'],
                rechecked='ActualOfflineHashAndGitBlobVerification', bytes=len(b))


def main():
    now = dt.datetime.now(dt.timezone.utc)
    common = dict(stage='M8-J0', base_commit=BASE, checked_at_utc=now.isoformat(),
                  checked_at_local=now.astimezone(ZoneInfo('Europe/Berlin')).isoformat(),
                  timezone='Europe/Berlin', official_code_commit=CODE,
                  evidence_kind='OfflineSourceReviewPlusBoundedPublicMetadata',
                  generator_sha256=sha(__file__), payload_status='PayloadNotVerified',
                  payload_downloads=0, payload_get_requests=0, range_requests=0,
                  archive_extractions=0, CARLA_connections=0, server_launches=0,
                  GPU_tasks=0, training_steps=0, default_model_changes=0,
                  vehicle_control_changes=0, messages_sent_to_maintainers=0)

    def report(name, **fields): write(ROOT / ('reports/m8j0_' + name + '.json'), dict(common, **fields))
    sources = [frozen_source(x) for x in ('README.md', 'LICENSE', 'docs/anno.md',
        'docs/v004_update.md', 'tools/data_collect.py', 'tools/download_mini.sh',
        'docs/bench2drive_mini_10.json', 'leaderboard/leaderboard/envs/sensor_interface.py')]
    # Fresh official manifest must exactly equal the pre-existing fixed-commit bytes.
    api = read(SRC / 'github_manifest.body')
    manifest_bytes = base64.b64decode(api['content'])
    assert manifest_bytes == (OLD / 'b2dapi__docs__bench2drive_mini_10.json.txt').read_bytes()
    manifest = json.loads(manifest_bytes); target = manifest[CLIP]
    assert target == dict(size=280176944, sha256='b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2')
    for name in ('carla_readme', 'github_utils'):
        d = read(SRC / (name + '.body')); b = base64.b64decode(d['content'])
        assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == d['sha']
        with (SRC / (name + '.decoded.txt')).open('xb') as f: f.write(b)
        sources.append(dict(path='reports/m8j0_sources/' + name + '.decoded.txt',
                            sha256=hashlib.sha256(b).hexdigest(), git_blob_sha=d['sha'],
                            official_url=d['html_url'], bytes=len(b),
                            rechecked='ActualServerOfficialAPIAndGitBlobVerification'))
    report('source_inventory', files=sources)

    requests = [read(p) for p in sorted(SRC.glob('*.request.json'))]
    for r in requests:
        r['display_at_europe_berlin'] = dt.datetime.fromisoformat(r['checked_at_utc']).astimezone(ZoneInfo('Europe/Berlin')).isoformat()
    report('official_access_probe', status='OfficialMetadataUnavailable', scope='TargetAssetOnWorkspaceServer',
        github_fixed_manifest='OfficialMetadataAccessible', server_requests=requests,
        server_request_count=len(requests), metadata_body_bytes=sum(r['saved_body_bytes'] for r in requests),
        max_server_requests=18, retries=0, target_head_count=1, target_get_count=0,
        target_observation=dict(http_status=None, content_length=None, etag=None, object_hash=None,
            redirect_chain=[], login_required='NotDetermined', additional_terms='NotDetermined',
            method_support='NotDetermined', error='curl exit 7: connection refused to huggingface.co:443',
            attribution='ConnectionFailure; no HTTP response, not authentication/404/TLS evidence'),
        browser_observation=dict(kind='SeparateResearchServiceCachedOfficialPages',
            public_card_readable=True, public_partial_tree_readable=True, repository_public_metadata='SupportedByBrowser',
            private_or_gated_api_status='NotAvailable', target_blob_page='ToolError_NotHTTPStatus',
            target_revision_api='ToolError_NotHTTPStatus', payload_access='NotTested',
            cannot_substitute_for_server_access=True),
        metadata_urls_only=True, requests_ended=True,
        time_display_erratum='First six raw request records used explicit UTC+8 display. Their UTC timestamps are authoritative and unmodified; this report adds Europe/Berlin. Executed source copies retained by SHA.',
        head_implementation_note='Executed HEAD command included max-filesize for metadata. Failure occurred before HTTP headers, so it did not establish object size. Final helper removes body-size limit for HEAD; no repeat performed.',
        environment='No explicit HTTP(S)/ALL proxy env observed; transparent routing not established; no network configuration changed')

    report('official_asset_identity', status='OfficialManifestIdentitySupported_HostedRevisionPartial',
        filename=CLIP, expected_bytes=target['size'], expected_sha256=target['sha256'],
        official_manifest_url=GH + 'docs/bench2drive_mini_10.json',
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), manifest_git_blob_sha=api['sha'],
        fresh_manifest_matches_frozen=True, expected_hash_is_download_verification=False,
        repository='rethinklab/Bench2Drive', exact_path=CLIP,
        path_evidence=GH + 'tools/download_mini.sh', source_list_entries=len(manifest),
        mini_manifest_total_bytes=sum(x['size'] for x in manifest.values()),
        release='Legacy Mini (10) subset of legacy Base (1000); NOT v0.0.4 data or Dev10 routes',
        repository_revision_candidate=HF_README_COMMIT,
        revision_evidence='Browser repository main and README history show this revision; metadata API and target file mapping not verified',
        target_file_revision=None, target_lfs_sha256=None, target_storage_version=None,
        target_file_exists_at_revision='NotVerified', archive_license_files='PayloadNotVerified',
        release_distinctions=[
            dict(name='Legacy Mini', meaning='10 selected archives from Base; target explicitly listed'),
            dict(name='Legacy Base', meaning='1000 legacy clips; official download script points to rethinklab/Bench2Drive'),
            dict(name='Legacy Full/Sup', meaning='13638 clips combined in fixed README; out of scope'),
            dict(name='v0.0.4 training', meaning='Separate 1100-route release with updated labels/occupancy; not target'),
            dict(name='v0.0.4 depth', meaning='Separate NPZ depth release; not evidence of target encoding'),
            dict(name='Dev10', meaning='Closed-loop development route definitions selected from 220 routes, not the ten Mini archives'),
            dict(name='v0.0.4 val', meaning='Separate 220-route evaluation XML, not target archive')],
        card_count_discrepancy='HF older description says 10000 training clips; not used as target repository enumeration or replacement for fixed Mini/Base manifest')

    tree = read(OLD / 'b2d_tree.json')
    license_paths = [dict(path=x['path'], git_blob_sha=x['sha']) for x in tree['tree'] if 'license' in x['path'].lower()]
    history = read(SRC / 'github_license_history.body')
    issue_search = read(SRC / 'github_license_issues.body')
    report('license_conflict_evidence', status='LicenseConflictUnresolved', review_scope='FactsAndConditions_NotLegalOpinion',
        github_license=dict(name='CC-BY-NC-ND-4.0', url=GH + 'LICENSE', source=sources[1]),
        github_readme=dict(url=GH + 'README.md', lines=[251,253], scope='All assets and code unless otherwise specified'),
        huggingface=dict(url=HF + '/blob/main/README.md', metadata_license='apache-2.0',
            prose_license='Apache 2.0 for assets and code unless otherwise specified',
            evidence_level='OfficialPageViaBrowser_NotServerResponse',
            visible_readme_commit=HF_README_COMMIT, visible_commit_date='2024-08-14',
            source_capture='reports/m8j0_sources/browser_hf_license_capture.json'),
        github_license_history=[dict(sha=x['sha'], date=x['commit']['author']['date'], message=x['commit']['message']) for x in history],
        time_inference='HF card predates GitHub 2024-11-02 license change; possible stale card, not proof of revocation, dual licensing or asset-specific exception',
        fixed_github_license_paths=license_paths, github_tree_truncated=tree['truncated'],
        extra_terms=dict(github_clip_specific='NotFoundInFixedTree', hf_complete_tree='NotVerified', archive='PayloadNotVerified'),
        upstream_conditions=dict(carla_code='MIT', carla_specific_assets='CC-BY; exact asset chain still unverified',
            unreal_engine='Separate terms; not blanket evidence every rendered output inherits engine license',
            third_party='No complete target asset provenance/NOTICE inventory; do not infer clearance',
            source=sources[-2]),
        maintainer_resolution=dict(scoped_github_issue_query='repo:Thinklab-SJTU/Bench2Drive license',
            returned_count=issue_search['total_count'], incomplete_results=issue_search['incomplete_results'],
            hf_discussions='Browser page shows no discussions; cached scope only',
            conclusion='No asset-specific reconciliation found in bounded review; not proof no statement exists elsewhere',
            inquiry_sent=False),
        internal_use_note='CC 4.0 section 2(a)(1) allows noncommercial reproduction and production/reproduction of unshared adaptations. ND does not prohibit every internal training use. Asset applicability/other rights and publication remain separate questions.')

    code_path = OLD / 'b2dapi__tools__data_collect.py.txt'
    tree_ast = ast.parse(code_path.read_text())
    sensors_fn = next(x for x in ast.walk(tree_ast) if isinstance(x, ast.FunctionDef) and x.name == 'sensors')
    sensor_nodes = next(x.value.elts for x in sensors_fn.body if isinstance(x, ast.Assign) and isinstance(x.value, ast.List))
    cameras = []
    for n in sensor_nodes:
        d = ast.literal_eval(n)
        if d.get('type') != 'sensor.camera.rgb' or not d['id'].startswith('CAM_'): continue
        f = d['width'] / (2 * math.tan(math.radians(d['fov'])/2))
        cameras.append(dict(**d, source_line=n.lineno, evidence_level='SourceCodeDerived',
            payload_verification='PayloadNotVerified', expected_K=[[f,0,d['width']/2],[0,f,d['height']/2],[0,0,1]],
            K_note='Derived pinhole expectation from fixed utils.py; actual saved K not read'))
    assert len(cameras)==6
    fields = []
    def field(name, status, value, source, limitations):
        fields.append(dict(field=name, evidence_level=status, expected=value, source=source,
                           payload_verification='PayloadNotVerified', limitations=limitations))
    field('six_camera_names_resolution_fov','SourceCodeDerived',cameras,GH+'tools/data_collect.py#L430',
          'TOP_DOWN excluded; no payload verification; differs from HS rig')
    field('intrinsic','SourceCodeDerived','sensors.CAM_*.intrinsic; 3x3 pinhole K',GH+'tools/data_collect.py#L1065',
          'Verify original image size, principal point, crop/resize; source formula is not a calibration measurement')
    field('camera_to_ego','SourceCodeDerived','sensors.CAM_*.cam2ego; 4x4 CARLA sensor basis',GH+'tools/data_collect.py#L1074',
          'Optical camera axes require explicit mapping (x,y,z) CARLA -> (y,-z,x); no existing adapter change')
    field('world_to_camera','SourceCodeDerived','sensors.CAM_*.world2cam; inverse actor transform',GH+'tools/data_collect.py#L1069',
          'Not automatically OpenCV optical basis; compare inverse closure and projected corners')
    field('objects_and_boxes','SourceDocumented','bounding_boxes: ego/vehicles/walkers/traffic_light/traffic_sign; id,type_id,center,extent,world_cord,rotation and transforms',GH+'docs/anno.md',
          'extent is half-size. Trigger volumes are not physical obstacles. Static location/rotation known unreliable; full cone/barrier/background identity not assured')
    field('dynamic_actor_identity','SourceCodeDerived','CARLA Actor id/type_id in annotation',GH+'tools/data_collect.py#L1190',
          'Scope to clip/session; not global ID; instance pixel association and lifetime must be audited; do not equate namespaces')
    field('instance','SourceDocumented','camera/instance_{view}/{ordinal}.png',GH+'docs/anno.md',
          'Same mounts as RGB per docs; actual encoding and aliasing not tested; no unique identity guarantee')
    field('semantic','SourceDocumented','camera/semantic_{view}/{ordinal}.png',GH+'docs/anno.md',
          'Semantics not full object-instance identity or complete occupancy; payload not seen')
    field('depth','SourceCodeDerived','Legacy save: convert_depth(BGR) then PNG; v0.0.4 separate NPZ must not be substituted',GH+'tools/data_collect.py#L765',
          'Fixed utils decodes to metric scalar; exact dtype/quantization on archive unknown. Do not apply raw RGB24 decoder to saved scalar PNG')
    field('rgb_encoding','SourceDocumented','JPEG quality20, 1600x900 source rig',GH+'docs/anno.md',
          'Lossy far-field small-object support; pixel size is not measured information quality')
    field('source_world_frame','NotAvailable','Not present in reviewed anno save dictionary',GH+'tools/data_collect.py#L740',
          'Filename uses self.count, not proven source frame. No fabricated world frame')
    field('per_sensor_timestamp','NotAvailable','Not present in reviewed save schema',GH+'tools/data_collect.py#L792',
          'No payload evidence; source synchronization does not preserve timestamps. No dynamic-time certification')
    field('source_sensor_frame','NotAvailable','Source callback sees image.frame; saved anno does not retain it',GH+'leaderboard/leaderboard/envs/sensor_interface.py#L161',
          'get_data(frame) filters intended source frame; actual archive raw sensor frame not verified')
    field('file_sequence_ordinal','SourceCodeDerived','self.count -> five-digit save names',GH+'tools/data_collect.py#L740',
          'Ordinal alone cannot recover elapsed time, dropped frames or original timestamp')
    field('route_and_scenario','SourceDocumented','archive name: DynamicObjectCrossing/Town02/Route13/Weather6; docs describe directory layout',GH+'docs/anno.md',
          'Internal route/session/episode manifest unverified; filename is not runtime provenance')
    field('synchronization_intent','SourceCodeDerived','get_data(frame) discards nonmatching frame buffers',GH+'leaderboard/leaderboard/envs/sensor_interface.py#L222',
          'Only code intent; not proof this legacy archive was collected by the current fixed source')
    field('static_vehicle_association','SourceCodeDerived','Greedy nearest XY level-bbox association up to20m, then remove box',GH+'tools/data_collect.py#L1254',
          'Not a rendered identity proof. center/local/world wording ambiguity and possible mismatches remain')
    field('physical_contact_mesh_full_background','NotAvailable','No complete verified per-asset physical mesh/contact/identity contract',GH+'docs/anno.md',
          'Unknown must remain Unknown; no inferred safe Free')
    report('source_schema_contract', status='SourceSchemaReviewCompleted_PayloadNotVerified', cameras=cameras,
        field_evidence=fields, source_inventory='reports/m8j0_source_inventory.json',
        hs_rig_reference='configs/cameras.yaml', hs_rig_sha256=sha(ROOT/'configs/cameras.yaml'),
        calibration_reuse_forbidden=True, top_down_is_inference_input=False,
        gt_fields_are_supervision_only=['depth','instance','semantic','bounding_boxes','actor_state','world_transform'],
        rgb_inference_contract=['six_rgb_images','audited_calibration_only'],
        temporal_use='NotReady: raw time not established; offline one-frame schema study only',
        split='DevelopmentOnly; official Train not independent HS Final; no split reassignment',
        source_release_caveat='Pinned source establishes expected fields, not exact collector version or actual fields of the legacy target archive')

    freeze = read(ROOT/'reports/m8j0_historical_freeze.json')
    for r in freeze['files']: assert sha(r['path']) == r['sha256'], r['path']
    # Independently check old supplemental manifest as well as original frozen records.
    for r in read(ROOT/'reports/m8i1b_supplement_manifest.json')['files']:
        assert sha(r['path'])==r['sha256'],r['path']
    old_test = subprocess.run(['python3','-m','tools.m8i1b_verify'], cwd=ROOT, capture_output=True,text=True)
    tests = subprocess.run(['python3','-m','unittest','tests.test_m8j0_review','-v'], cwd=ROOT,capture_output=True,text=True)
    write(SRC/'historical_verification.log',old_test.stdout+old_test.stderr)
    write(SRC/'boundary_tests.log',tests.stdout+tests.stderr)
    assert old_test.returncode==tests.returncode==0
    report('verification', status='Passed', kind='ActualOfflineArtifactChecksAndSyntheticRequestBoundaryTests',
        historical_306_verification=read_output(old_test.stdout),
        freeze_entries=len(freeze['files']), unique_real_paths=len({str(Path(x['path']).resolve()) for x in freeze['files']}),
        duplicate_path_note='Absolute and repository-relative aliases; 369 records map to357 unique files, not369 independent assets',
        supplemental_manifest_check='Passed', source_blob_checks='Passed', fresh_manifest_check='Passed',
        request_boundary_tests=8, request_boundary_result='Passed', tests_log='reports/m8j0_sources/boundary_tests.log',
        payload_assertions='No target GET/Range in request logs', runtime_tests='NotRun_NotAuthorized',
        freeze_timing_note='Initial freeze helper failed before writing (Python3.10 has no hashlib.file_digest). Corrected streamed hash freeze completed after first two metadata probes. Old records were not edited; independent original306 verification and supplemental checks passed.')

    gates = {
        'A_License':dict(status='LicenseScopePartial', conflict='LicenseConflictUnresolved', satisfied=False,
            reason='Target asset exception/licensing reconciliation and unexamined in-archive terms unresolved; noncommercial internal reproduction has conditional support, not blanket prohibition'),
        'B_Access':dict(status='OfficialMetadataUnavailable', satisfied=False,
            reason='Server HF API and exact asset HEAD connection refused. GitHub works; browser card is separate evidence'),
        'C_Identity':dict(status='Partial', satisfied=False,
            reason='Fixed official filename/size/SHA and repo root path confirmed; exact hosted object revision/LFS/HEAD unresolved'),
        'D_Schema':dict(status='SourceDocumented', satisfied=True,
            reason='Six-camera K/T and3D annotations have source basis; PayloadNotVerified, source-to-archive version correspondence not proven'),
        'E_Time':dict(status='Partial', satisfied=False,
            reason='Gap explained by reviewed save schema; original per-sensor time/frame not recovered, cannot certify temporal research'),
        'F_Cost':dict(status='ResourcePlanPrepared', satisfied=True,
            reason='280176944 compressed bytes fixed; bounded decompression/one-frame plan, unknown full expansion'),
        'G_NextStep':dict(status='AccessOrLicenseNoGo', satisfied=False,
            reason='No single-clip download proposal yet. Resolve official asset license and server metadata/target revision first')}
    report('acceptance', status='LicenseAndAccessReviewCompleted', overall='Partial', decision='AccessOrLicenseNoGo',
        gates=gates, single_clip_review_proposal_eligible=False, download_approved=False,
        training_readiness='TrainingReadinessNoGo', payload_gate='PayloadNotVerified',
        historical=dict(decision='StartupRootCauseUnresolved', ue_logs='NoRunMatchedUELog',
            M8_I1='VersionRPC succeeded; map identity rejected; actual map not recorded',
            M8_I1a='31 version RPC timeouts; never entered map identity read',
            consumed_single_startup_authorization=1, new_authorization=0,
            authorization_erratum='Three layers preserved; external transcript not platform verified',
            frozen_historical_asset_count=306),
        research_gate_is_not_training_gate=True, offline_test_status='Passed', maintainer_inquiry='DraftOnly_NotSent')

    available = shutil.disk_usage('/root/gpufree-data').free
    write(ROOT/'reports/m8j0_single_clip_resource_plan.md', f'''# M8-J0 单片段资源计划（未获执行授权）

检查时间：{now.isoformat()}；环境 Europe/Berlin。只制定计划，未建立数据解压目录。

唯一候选 `{CLIP}`，官方压缩字节数 **280,176,944**（280.177 MB / {280176944/1024**2:.3f} MiB）。
预期 SHA256 `{target['sha256']}`。来源为[固定官方清单]({GH}docs/bench2drive_mini_10.json)。它不是已下载文件的实测哈希。
当前数据盘可用 {available:,} bytes；该瞬时值不是配额，也不能证明未来空间足够。解压大小和条目数均未知。

未来申请的硬预算：单一官方文件；压缩区320MiB（一个文件，不另留重复完整副本）；累计tar解压流最多2GiB；实际落盘一帧最多256MiB；元数据/日志64MiB；目录硬配额3GiB；至少4GiB可用才执行；内存1GiB；CPU单线程；下载墙钟10分钟，清单/一帧读取10分钟。任何上限触发即失败并留证，不能自动扩大。数字是拟议限制，未声称能覆盖整个压缩包。

拟议独立目录 `/root/gpufree-data/hs-drive-experiments/m8j1-single-clip-review/`，当前未创建。执行顺序：

1. 许可澄清且服务器官方API/HEAD可访问后，固定HF修订和唯一对象；拒绝其他文件、镜像或自动全库下载。重新核对官方size/SHA及适用条款，有差异即停。
2. 独立授权后才用校验证书的客户端下载到独占 `.partial`；限制320MiB；完成后严格比对280176944bytes和官方SHA，再在同目录原子重命名。保留来源和公共重定向证据，脱敏签名URL。不绕过认证、配额或条款。
3. 不运行 `tar -xf` / `extractall`。流式扫描tar，不执行文件；禁止绝对路径、`..`、重复/大小写冲突、软硬链接、设备/FIFO、稀疏条目和越界PAX路径；普通目录和普通文件白名单。计数上限50,000、单条目64MiB、累计解压流2GiB，超限退出；gzip尾部/CRC或tar截断异常也拒绝。
4. 先索引名称/大小和附带LICENSE/NOTICE；出现额外条款立即暂停后续像素读取。只选首个完整六摄+anno的相同文件序号，最多一次复读归档、总累计解压不超过2GiB。只释放该帧白名单成员，最多256MiB；其他帧不解压落盘。若顺序/预算不容许，保持Partial，不改选片段。嵌套json.gz单独限制解压16MiB、JSON大小/深度；图像每张不超过10M像素，拒绝解压炸弹。
5. CPU检查六摄分辨率、K主点/焦距、T逆及坐标轴、相机名称/朝向、GT角点投影与实例关系；保留源数值，不复制HS K/T，不改M8-B Adapter。缺原frame/timestamp则明确null；序号只能做离线索引。
6. RGB+审查标定走输入路径；Actor、Depth、Instance、Semantic仅走监督/诊断。TOP_DOWN、LiDAR、Radar、expert action均不进入正式RGB推理。无需CARLA、PETR、CUDA或训练依赖。

验收只覆盖一帧Schema与文件完整性；不证明同步时序、远场身份、完整数据质量、独立Final或训练资格。本轮未提供可自动执行下载/解压的命令。
''')


def read_output(s): return json.loads(s)


if __name__ == '__main__': main()
