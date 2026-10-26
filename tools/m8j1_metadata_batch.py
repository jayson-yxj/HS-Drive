"""One-shot official metadata/DNS/TCP/TLS evidence batch. No payload transport."""
import datetime as dt
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/m8j1_sources'
CLIP = 'DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz'
PIN = 'ce56cfbc26edf7796586e26fd44644f0ac2f6bcc'
CODE = '7ec25d1c9f7522d923ce5f3420986cef1cb2d956'


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    with Path(p).open('x') as f: json.dump(obj,f,indent=2,ensure_ascii=False); f.write('\n')


def validate(method,url):
    p=urllib.parse.urlsplit(url)
    assert p.scheme=='https' and not p.username and not p.password and p.port in (None,443)
    assert urllib.parse.unquote(p.path)==p.path
    if method=='HEAD':
        assert p.hostname=='huggingface.co'
        assert p.path==f'/datasets/rethinklab/Bench2Drive/resolve/{PIN}/{CLIP}'
    else:
        assert method=='GET'
        assert '.tar' not in p.path and '.zip' not in p.path
        if p.hostname=='huggingface.co':
            assert p.path in ('/api/datasets/rethinklab/Bench2Drive',
                f'/api/datasets/rethinklab/Bench2Drive/revision/{PIN}',
                f'/datasets/rethinklab/Bench2Drive/raw/{PIN}/README.md')
        else:
            assert p.hostname=='api.github.com'
            assert p.path in ('/search/issues', '/repos/Thinklab-SJTU/Bench2Drive/contents/README.md',
                '/repos/Thinklab-SJTU/Bench2Drive/contents/LICENSE')


def public_location(value):
    p=urllib.parse.urlsplit(value)
    return urllib.parse.urlunsplit((p.scheme,p.hostname or '',p.path,'REDACTED' if p.query else '', ''))


def network(host):
    row=dict(host=host,started_at_utc=now(),mode='Direct DNS/TCP/TLS; separate from curl environment routing')
    script='import socket,json,sys; print(json.dumps(sorted({(x[0],x[4][0]) for x in socket.getaddrinfo(sys.argv[1],443,type=socket.SOCK_STREAM)})))'
    try:
        r=subprocess.run(['python3','-c',script,host],capture_output=True,text=True,timeout=8)
        if r.returncode:
            row.update(dns='Failed',dns_error=r.stderr[-1000:]);return row
        addresses=json.loads(r.stdout)
        row.update(dns='Resolved',addresses=[dict(family=f,address=a,scope='public' if ipaddress.ip_address(a).is_global else 'non_global') for f,a in addresses])
        family,address=next(((f,a) for f,a in addresses if f==socket.AF_INET),addresses[0])
        row['tcp_target']=address;start=time.monotonic()
        with socket.socket(family,socket.SOCK_STREAM) as s:
            s.settimeout(5)
            try:s.connect((address,443));row.update(tcp='Connected',tcp_ms=(time.monotonic()-start)*1000)
            except OSError as e:
                row.update(tcp='Failed',tcp_errno=e.errno,tcp_error=str(e),tcp_ms=(time.monotonic()-start)*1000,tls='NotReached');return row
            try:
                with ssl.create_default_context().wrap_socket(s,server_hostname=host) as t:
                    row.update(tls='Verified',tls_version=t.version(),peer_certificate_sha256=hashlib.sha256(t.getpeercert(binary_form=True)).hexdigest())
            except (OSError,ssl.SSLError) as e: row.update(tls='Failed',tls_error=str(e))
    except subprocess.TimeoutExpired:row.update(dns='Timeout',tcp='NotAttempted',tls='NotReached')
    finally:row['finished_at_utc']=now()
    return row


def http(name,method,url):
    validate(method,url)
    existing=list(OUT.glob('*.request.json'));assert len(existing)<8
    assert sum(json.loads(p.read_text())['body_bytes'] for p in existing)+2097152<=8388608
    row=dict(name=name,method=method,url=url,started_at_utc=now(),tls_verify=True,
             retries=0,range_request=False,follow_redirects=False,source_sha256=sha(__file__),body_bytes=0)
    with tempfile.TemporaryDirectory(prefix='m8j1_metadata_') as tmp:
        body=Path(tmp)/'body';hdr=Path(tmp)/'headers'
        cmd=['curl','-q','--silent','--show-error','--proto','=https','--max-time','20','--connect-timeout','8',
            '--dump-header',str(hdr),'--output','/dev/null' if method=='HEAD' else str(body),
            '--write-out','%{json}']
        if method=='HEAD':cmd+=['--head']
        else:cmd+=['--max-filesize','2097152']
        cmd+=[url]
        try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=23)
        except subprocess.TimeoutExpired:
            row.update(curl_exit_code=None,error='SupervisorTimeout',category='Timeout',http_status=None)
        else:
            try:metrics=json.loads(r.stdout)
            except json.JSONDecodeError:metrics={}
            row.update(curl_exit_code=r.returncode,http_status=metrics.get('http_code') or None,
                error=r.stderr.strip()[:1500],metrics={k:metrics.get(k) for k in ('time_namelookup','time_connect','time_appconnect','time_starttransfer','time_total','ssl_verify_result','size_download','num_connects')})
            code=row['http_status']
            row['category']=({5:'ProxyDNSFailure',6:'DNSFailure',7:'ConnectionFailure',28:'Timeout',35:'TLSHandshakeFailure',60:'TLSCertificateFailure',63:'MetadataLimit'}.get(r.returncode,'TransportFailure') if r.returncode else
                'Success' if code and 200<=code<300 else 'Redirect' if code and 300<=code<400 else
                'AuthResponse' if code in (401,403) else 'NotFound' if code==404 else 'MethodUnsupported' if code in (405,501) else 'OtherHTTP')
        headers={}
        if hdr.exists():
            for line in hdr.read_text().splitlines():
                if line.startswith('HTTP/'):headers={}
                elif ':' in line:
                    k,v=line.split(':',1);k=k.lower();v=v.strip()
                    if k in {'content-type','content-length','etag','x-linked-etag','x-linked-size','x-repo-commit','location','last-modified','x-error-code'}:
                        headers[k]=public_location(v) if k=='location' else v
        row['headers']=headers
        if body.exists():
            b=body.read_bytes();assert len(b)<=2097152
            if b:
                p=OUT/(name+'.body')
                with p.open('xb') as f:f.write(b)
                row.update(body_path=str(p.relative_to(ROOT)),body_bytes=len(b),body_sha256=sha(p))
    row['finished_at_utc']=now();write(OUT/(name+'.request.json'),row)
    print(json.dumps({k:row[k] for k in ('name','category','http_status','body_bytes')}),flush=True)
    return row


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='307f4fb34b501c6400f20e4e437963429527a333'
    OUT.mkdir(exist_ok=True)
    # Exclusive marker intentionally prevents restart after a crash or partial run.
    write(OUT/'batch_started.json',dict(started_at_utc=now(),batch_limit=1,
        source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'configs/m8j1_review_protocol.json'),
        environment='HS-Drive workspace GPU server; no GPU task',
        proxy_env_present={k:bool(os.getenv(k)) for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy')},
        no_credentials_read=True,no_external_message_authorized=True))
    for host in ('huggingface.co','api.github.com'):
        write(OUT/(host+'.network.json'),network(host))
    hf=http('hf_repo','GET','https://huggingface.co/api/datasets/rethinklab/Bench2Drive?blobs=true')
    http('hf_target_head','HEAD',f'https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/{PIN}/{CLIP}')
    if hf['category']=='Success':
        http('hf_pinned_repo','GET',f'https://huggingface.co/api/datasets/rethinklab/Bench2Drive/revision/{PIN}?blobs=true')
        http('hf_pinned_readme','GET',f'https://huggingface.co/datasets/rethinklab/Bench2Drive/raw/{PIN}/README.md')
    for name,path in [('github_fixed_license','LICENSE'),('github_fixed_readme','README.md')]:
        http(name,'GET',f'https://api.github.com/repos/Thinklab-SJTU/Bench2Drive/contents/{path}?ref={CODE}')
    q=urllib.parse.urlencode({'q':'repo:Thinklab-SJTU/Bench2Drive license OR licensing OR permission OR Apache','per_page':30})
    http('github_license_discussion_search','GET','https://api.github.com/search/issues?'+q)
    write(OUT/'batch_finished.json',dict(finished_at_utc=now(),batch_count=1,
        http_requests=len(list(OUT.glob('*.request.json'))),payload_get=0,range_requests=0,retries=0,
        pending_requests=0,additional_batch_allowed=False))


if __name__=='__main__':main()
