"""Single bounded metadata batch with explicit discovered proxy; never downloads target payload."""
import base64
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from urllib.parse import urlsplit,urlunsplit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/m8j2_sources'
BASE='5d6252a6f5a536d29cce5d2fb756134dec4057fe'
CODE='7ec25d1c9f7522d923ce5f3420986cef1cb2d956'
CLIP='DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz'
PROXY='http://127.0.0.1:7897'
REPO='rethinklab/Bench2Drive'


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')


def public_url(url):
    p=urlsplit(url)
    q=p.query if p.hostname in ('api.github.com','huggingface.co') else ('REDACTED' if p.query else '')
    return urlunsplit((p.scheme,p.hostname or '',p.path,q,''))


def redact(s):
    s=re.sub(r'https?://[^\s\x27\"<>]+',lambda m:public_url(m.group()),str(s))
    s=re.sub(r'(?i)(authorization|cookie|token|password)[=:][^\s]+',r'\1=REDACTED',s)
    s=re.sub(r'(?<![\w])(?:\d{1,3}\.){3}\d{1,3}(?![\w])','<ip_redacted>',s)
    return s


def validate(method,url,storage_authorized=False):
    p=urlsplit(url)
    assert p.scheme=='https' and p.port in (None,443) and not p.username and not p.password
    assert method in ('GET','HEAD')
    if p.hostname=='huggingface.co':
        if '/resolve/' in p.path:
            assert method=='HEAD' and p.path.startswith('/datasets/'+REPO+'/resolve/') and p.path.endswith('/'+CLIP)
        elif p.path.startswith('/api/datasets/'+REPO):
            assert method=='GET' and not any(x in p.path for x in ('.tar','.zip'))
        else:
            assert method=='GET' and p.path.startswith('/datasets/'+REPO+'/raw/')
            assert p.path.rsplit('/',1)[-1] in ('README.md','LICENSE','LICENSE.md','LICENSE.txt','NOTICE','NOTICE.md')
    elif p.hostname=='api.github.com':
        assert method=='GET' and (p.path=='/search/issues' or p.path in {
            '/repos/Thinklab-SJTU/Bench2Drive/contents/'+name for name in ['LICENSE','README.md']})
    else:
        assert method=='HEAD' and storage_authorized and p.hostname.endswith('.hf.co')


def discovery():
    from tools.m8a_proxy_audit import discovery as existing_discovery
    d=existing_discovery()
    d['all_proxy_vars']={k:bool(os.getenv(k)) for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy')}
    r=subprocess.run(['git','config','--show-origin','--get-regexp',r'(proxy|sslVerify)$'],capture_output=True,text=True)
    d['git_proxy_config']=[dict(origin=x.split()[0],key=x.split()[1],value='REDACTED',tls_false=x.split()[-1]=='false') for x in r.stdout.splitlines()]
    d['runtime_versions']={x:importlib.metadata.version(x) for x in ('requests','huggingface_hub','urllib3')}
    d['source_sha256']=sha(__file__);d['checked_at_utc']=now()
    return d


def dns():
    r=subprocess.run(['python3','-c','import socket,json;print(json.dumps(sorted({(x[0],x[4][0]) for x in socket.getaddrinfo("huggingface.co",443,type=socket.SOCK_STREAM)})))'],capture_output=True,text=True,timeout=8)
    return dict(time_utc=now(),returncode=r.returncode,answers=json.loads(r.stdout) if not r.returncode else [],
        error=r.stderr[-500:],interpretation='Local resolver only. HTTP CONNECT resolves the target at proxy; no inferred upstream DNS/exit identity.')


class Recorder:
    def __init__(self,offset=0):
        self.rows=[];self.seen=set();self.total=0;self.storage=set();self.started=time.monotonic();self.offset=offset
    def send(self,session,original,request,**kwargs):
        validate(request.method,request.url,public_url(request.url) in self.storage)
        assert len(self.rows)+self.offset<14 and time.monotonic()-self.started<300
        assert 'Range' not in request.headers and 'Authorization' not in request.headers
        key=(session.audit_mode,request.method,request.url)
        assert key not in self.seen,'no retry';self.seen.add(key)
        kwargs.update(timeout=(8,15),verify=True,allow_redirects=False,stream=True)
        row=dict(index=len(self.rows)+self.offset,started_at_utc=now(),client=session.audit_client,mode=session.audit_mode,
            method=request.method,url=public_url(request.url),proxy_effective=bool(kwargs.get('proxies')),
            proxy_endpoint_kind='configured_loopback' if any(v==PROXY for v in kwargs.get('proxies',{}).values()) else 'none',
            tls_verify=True,range_request=False,retries=0,body_bytes=0,source_sha256=sha(__file__))
        self.rows.append(row);response=None
        try:
            response=original(request,**kwargs)
            row.update(http_status=response.status_code,category='Success' if response.ok and response.status_code<300 else 'Redirect' if response.is_redirect else 'HTTPError')
            sock=getattr(getattr(response.raw,'_connection',None),'sock',None)
            if sock:
                peer=sock.getpeername();row['connected_to_loopback']=peer[0] in ('127.0.0.1','::1')
                row['connected_to_configured_port']=peer[1]==7897
                cert=sock.getpeercert(binary_form=True)
                row.update(tls_version=sock.version(),peer_certificate_sha256=hashlib.sha256(cert).hexdigest())
            row['tls_completed']=True
            headers={}
            for k,v in response.headers.items():
                if k.lower() in {'content-length','content-type','etag','x-linked-etag','x-linked-size','x-repo-commit','x-xet-hash','location','last-modified','x-error-code'}:
                    headers[k.lower()]=public_url(v) if k.lower()=='location' else v
            row['headers']=headers
            if request.method=='GET':
                body=bytearray()
                for block in response.iter_content(65536):
                    body.extend(block)
                    assert len(body)<=2097152 and self.total+len(body)<=8388608,'metadata cap'
                    assert time.monotonic()-self.started<300,'batch timeout'
                response._content=bytes(body);response._content_consumed=True
                path=OUT/f'http_{row["index"]:02}.body'
                saved=bytes(body)
                if response.status_code>=400:
                    saved=redact(saved.decode(errors='replace')).encode()
                with path.open('xb') as f:f.write(saved)
                row.update(body_bytes=len(body),saved_body_bytes=len(saved),raw_response_sha256=hashlib.sha256(body).hexdigest(),body_sha256=sha(path),body_path=str(path.relative_to(ROOT)))
                self.total+=len(body)
            elif response.is_redirect:
                location=response.headers.get('Location','')
                if urlsplit(location).scheme=='https' and (urlsplit(location).hostname or '').endswith('.hf.co'):
                    self.storage.add(public_url(location))
            return response
        except Exception as exc:
            row.update(category=type(exc).__name__,error=redact(str(exc))[:1800],http_status=row.get('http_status'))
            raise
        finally:
            row['finished_at_utc']=now();dump(OUT/f'http_{row["index"]:02}.json',row)
            print(json.dumps({k:row.get(k) for k in ('index','client','mode','method','http_status','category','body_bytes')}),flush=True)


def make_session(rec,mode,client):
    import requests
    from requests.adapters import HTTPAdapter
    s=requests.Session();s.trust_env=(mode=='inherited')
    s.audit_mode=mode;s.audit_client=client
    if mode!='inherited':s.proxies={'http':PROXY,'https':PROXY}
    s.mount('https://',HTTPAdapter(max_retries=0))
    original=s.send
    s.send=lambda request,**kwargs:rec.send(s,original,request,**kwargs)
    return s


def git_probe():
    # Git protocol v0 ls-remote uses GET advertisement, no clone or LFS smudge.
    env=os.environ.copy()
    for k in list(env):
        if k.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'):env.pop(k)
    env.update(HTTPS_PROXY=PROXY,HTTP_PROXY=PROXY,NO_PROXY='',GIT_TERMINAL_PROMPT='0',GIT_TRACE_CURL='1',GIT_TRACE_CURL_NO_DATA='1',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null')
    t=now()
    cmd=['git','-c','http.sslVerify=true','-c','credential.helper=','-c','protocol.version=0','-c','http.lowSpeedLimit=1','-c','http.lowSpeedTime=15',
         'ls-remote','https://huggingface.co/datasets/'+REPO,'HEAD']
    try:
        r=subprocess.run(cmd,cwd='/tmp',env=env,capture_output=True,text=True,timeout=25)
        trace=r.stderr
        # Store only proof flags, not headers, cookies, addresses or tokens.
        result=dict(started_at_utc=t,finished_at_utc=now(),returncode=r.returncode,
            remote_head=r.stdout.strip() if re.fullmatch(r'[0-9a-f]{40}\s+HEAD\s*',r.stdout) else None,
            proxy_connected=('127.0.0.1' in trace and '7897' in trace and 'Connected' in trace),
            connect_tunnel='CONNECT huggingface.co:443' in trace,
            certificate_verified='server certificate verification OK' in trace,
            http_status_lines=re.findall(r'(?i)HTTP/[12](?:\.\d)? \d{3}[^\r\n]*',trace),
            trace_sha256=hashlib.sha256(trace.encode()).hexdigest(),raw_trace_persisted=False,
            repository_config_unchanged=True,clone_or_lfs=False,
            http_methods_observed=re.findall(r'(?:GET|POST) [^\s]+ HTTP/[\d.]+',trace))
    except subprocess.TimeoutExpired:result=dict(started_at_utc=t,returncode=None,category='SupervisorTimeout')
    dump(OUT/'git_proxy.json',result)


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
    OUT.mkdir(exist_ok=True)
    dump(OUT/'batch_started.json',dict(started_at_utc=now(),source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'configs/m8j2_review_protocol.json')))
    d=discovery();dump(OUT/'proxy_discovery.json',d)
    assert any(x['port']==7897 and x['loopback'] and 'verge-mihomo' in x['process'] for x in d['listeners'])
    dump(OUT/'dns.json',dns())
    rec=Recorder();direct=make_session(rec,'inherited','requests');s=make_session(rec,'explicit_http_proxy','requests')
    url='https://huggingface.co/api/datasets/'+REPO+'?blobs=true'
    try:
        with direct.get(url):pass
    except Exception:pass
    try:
        response=s.get(url);response.raise_for_status();info=response.json();response.close()
        revision=info['sha'];assert re.fullmatch('[0-9a-f]{40}',revision)
        target=next((x for x in info.get('siblings',[]) if x['rfilename']==CLIP),None)
        dump(OUT/'repository_identity.json',dict(revision=revision,private=info.get('private'),gated=info.get('gated'),
            license_card=info.get('cardData'),target=target,license_files=[x for x in info.get('siblings',[]) if re.search(r'(^|/)(license|notice)(\.|$)',x['rfilename'],re.I)]))
        from huggingface_hub import HfApi,configure_http_backend,get_hf_file_metadata,hf_hub_url
        configure_http_backend(backend_factory=lambda:make_session(rec,'explicit_http_proxy','huggingface_hub'))
        hf=HfApi(endpoint='https://huggingface.co',token=False)
        hinfo=hf.dataset_info(REPO,revision=revision,files_metadata=True,timeout=15,token=False)
        dump(OUT/'hub_repo.json',dict(sha=hinfo.sha,private=hinfo.private,gated=hinfo.gated,
            target=[dataclasses.asdict(x) for x in hinfo.siblings if x.rfilename==CLIP],library_version=importlib.metadata.version('huggingface_hub')))
        md=get_hf_file_metadata(hf_hub_url(REPO,filename=CLIP,repo_type='dataset',revision=revision),token=False,timeout=15)
        dump(OUT/'hub_file_metadata.json',dict(commit_hash=md.commit_hash,etag=md.etag,size=md.size,
            location=public_url(md.location),xet_file_data=str(getattr(md,'xet_file_data',None)),location_query_redacted=True))
        if public_url(md.location) in rec.storage:
            with s.head(md.location):pass
        with s.get(f'https://huggingface.co/datasets/{REPO}/raw/{revision}/README.md') as r:r.raise_for_status()
        # Inspect any exact root-level license/notice listed by the official repository, max2.
        names=[x['rfilename'] for x in info.get('siblings',[]) if x['rfilename'] in ('LICENSE','LICENSE.md','LICENSE.txt','NOTICE','NOTICE.md')]
        for name in names[:2]:
            with s.get(f'https://huggingface.co/datasets/{REPO}/raw/{revision}/{name}') as r:r.raise_for_status()
    except Exception as exc:
        dump(OUT/'hf_pipeline_failure.json',dict(time_utc=now(),category=type(exc).__name__,detail=redact(str(exc))[:1500]))
    git_probe()
    for name in ('LICENSE','README.md'):
        try:
            with s.get(f'https://api.github.com/repos/Thinklab-SJTU/Bench2Drive/contents/{name}?ref={CODE}') as r:r.raise_for_status()
        except Exception:pass
    dump(OUT/'batch_finished.json',dict(finished_at_utc=now(),requests_recorded=len(rec.rows),metadata_bytes=rec.total,
        git_probe_separate=True,payload_get=0,range_requests=0,automatic_retries=0,batch_count=1))


if __name__=='__main__':main()
