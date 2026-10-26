"""Fetch only four fixed official license/source documents; no dataset or runtime."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/m8j3_sources'
DOCUMENTS=[
 ('license_commit.patch','https://github.com/Thinklab-SJTU/Bench2Drive/commit/fd3009a.patch',100000),
 ('readme_commit.patch','https://github.com/Thinklab-SJTU/Bench2Drive/commit/8f9f1da.patch',100000),
 ('neurips2024.pdf','https://papers.nips.cc/paper/2024/file/017761f94a1cd66d01c041aff85492c4-Paper-Datasets_and_Benchmarks_Track.pdf',45000000),
 ('carla_content_versions.txt','https://raw.githubusercontent.com/carla-simulator/carla/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/Util/ContentVersions.txt',100000),
]


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'fetch_started.json').open('x') as f:json.dump(dict(started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),request_cap=4,automatic_retries=0,dataset_payloads=0),f)
    s=requests.Session();s.trust_env=False;s.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
    for name,url,cap in DOCUMENTS:
        row=dict(url=url,time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),method='GET',purpose='OfficialDocumentOnly',tls_verify=True,timeout=[8,30],max_bytes=cap)
        try:
            with s.get(url,stream=True,timeout=(8,30),verify=True,allow_redirects=False) as r:
                row.update(http_status=r.status_code,content_type=r.headers.get('Content-Type'))
                r.raise_for_status();assert not r.is_redirect
                b=bytearray()
                for block in r.iter_content(65536):
                    b.extend(block);assert len(b)<=cap
                with (OUT/name).open('xb') as f:f.write(b)
                row.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='Fetched')
        except Exception as e:row.update(status='Failed',error_type=type(e).__name__)
        with (OUT/(name+'.meta.json')).open('x') as f:json.dump(row,f,indent=2)
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
