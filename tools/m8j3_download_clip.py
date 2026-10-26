"""Download only the explicitly authorized pinned clip with official HF client."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path('/root/gpufree-data/hs-drive-private/m8j3-bench2drive-single-clip')
REPO = 'rethinklab/Bench2Drive'
REV = 'ce56cfbc26edf7796586e26fd44644f0ac2f6bcc'
NAME = 'DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz'
SIZE = 280176944
SHA = 'b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    os.umask(0o077)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    log = PRIVATE / 'download_events.jsonl'
    if log.exists():
        raise RuntimeError('Existing execution ledger: inspect it instead of starting another batch')

    def event(**row):
        row['time_utc'] = now()
        with log.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    frozen = json.loads((ROOT / 'reports/m8j2_official_asset_identity.json').read_text())
    assert (frozen['repository'], frozen['revision'], frozen['filename'], frozen['bytes'],
            frozen['lfs_sha256'], frozen['official_expected_sha256']) == (REPO, REV, NAME, SIZE, SHA, SHA)
    free = shutil.disk_usage(PRIVATE).free
    assert free >= 12 * 1024**3, 'Require 12 GiB available before single-clip acquisition'
    event(kind='Start', free_bytes=free, repository=REPO, revision=REV, filename=NAME,
          user_authorization='Latest user M8-J3 task: personal noncommercial single-clip download and local audit',
          platform_message_id='NotAvailable', max_outer_attempts=2, subprocess_wall_limit_s=1200,
          read_timeout_s=30, proxy_scheme='http', proxy_host='127.0.0.1', proxy_port=7897,
          TLS_verification=True, dataset_limit=1, maintainer_permission_not_claimed=True)
    os.environ.update(HF_HUB_DOWNLOAD_TIMEOUT='30', HF_HUB_ETAG_TIMEOUT='15',
                      HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_HUB_DISABLE_PROGRESS_BARS='1')
    import requests
    from huggingface_hub import configure_http_backend, hf_hub_download, __version__
    proxy = {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}

    def response_hook(response, *args, **kwargs):
        u = urlsplit(response.url)
        event(kind='HTTP', method=response.request.method, host=u.hostname,
              path=u.path, query='Omitted', status=response.status_code,
              content_length=response.headers.get('Content-Length'),
              content_range=response.headers.get('Content-Range'))

    def backend():
        s = requests.Session()
        s.trust_env = False
        s.proxies = proxy
        s.verify = True
        s.hooks['response'] = [response_hook]
        return s

    configure_http_backend(backend_factory=backend)
    event(kind='Library', huggingface_hub_version=__version__,
          resume='Official .incomplete continuation; at most 5 internal stream retries per outer attempt')
    start = time.monotonic()
    downloaded = None
    for attempt in range(1, 3):
        event(kind='Attempt', number=attempt)
        try:
            downloaded = Path(hf_hub_download(repo_id=REPO, filename=NAME, repo_type='dataset',
                revision=REV, local_dir=PRIVATE / 'download', proxies=proxy,
                token=False, etag_timeout=15, force_download=False))
            break
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            event(kind='Failure', number=attempt, exception_type=type(exc).__name__, http_status=status)
            if status in (401, 403, 404, 429) or attempt == 2:
                raise SystemExit(2)
            time.sleep(2)
    h = hashlib.sha256()
    with downloaded.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    result = dict(status='DownloadedAndHashVerified' if downloaded.stat().st_size == SIZE and h.hexdigest() == SHA else 'IntegrityNoGo',
        local_path=str(downloaded), size_bytes=downloaded.stat().st_size, sha256=h.hexdigest(),
        expected_size=SIZE, expected_sha256=SHA, frozen_J2_metadata_match=True,
        elapsed_s=time.monotonic()-start, checked_at_utc=now(), repository=REPO, revision=REV,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        license='LicenseConflictUnresolved', data_use='PersonalNoncommercialLocalOnly',
        CARLA_launches=0, inference_calls=0, training_steps=0)
    with (PRIVATE / 'download_result.json').open('x') as f:
        json.dump(result, f, indent=2)
    event(kind='Result', **result)
    if result['status'] != 'DownloadedAndHashVerified':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
