"""Complete bounded unfinished metadata checks after recorded local whitelist failure."""
from tools.m8j2_access_probe import *


def main():
    assert len(list(OUT.glob('http_*.json')))==6
    dump(OUT/'completion_started.json',dict(time_utc=now(),source_sha256=sha(__file__),
        probe_sha256=sha(ROOT/'tools/m8j2_access_probe.py'),
        reason='Local whitelist canonicalization failure; remaining finite metadata checks only, no retry after API rate limit'))
    rec=Recorder(offset=6);s=make_session(rec,'explicit_http_proxy','requests')
    rev=json.loads((OUT/'repository_identity.json').read_text())['revision']
    try:
        with s.get(f'https://huggingface.co/datasets/{REPO}/raw/{rev}/README.md') as r:r.raise_for_status()
    except Exception as exc:dump(OUT/'readme_failure.json',dict(kind=type(exc).__name__,detail=redact(str(exc))))
    try:
        with s.head(f'https://huggingface.co/datasets/{REPO}/resolve/{rev}/{CLIP}') as r:
            r.raise_for_status();location=r.headers.get('Location')
        if location and public_url(location) in rec.storage:
            with s.head(location) as r:r.raise_for_status()
    except Exception as exc:
        dump(OUT/'storage_followup_failure.json',dict(kind=type(exc).__name__,detail=redact(str(exc))))
    dump(OUT/'completion_finished.json',dict(time_utc=now(),http_requests=len(rec.rows),
        total_http_recorded=len(list(OUT.glob('http_*.json'))),payload_get=0,range_requests=0,metadata_bytes=rec.total,
        no_more_server_requests=True))


if __name__=='__main__':main()
