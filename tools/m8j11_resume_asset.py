"""One bounded official resume after user restored proxy; preserve initial failure."""
import hashlib,json,time,urllib.request,urllib.parse,zipfile,shutil
from pathlib import Path
from datetime import datetime,timezone
c=json.loads(Path('configs/m8j11_protocol.json').read_text());root=Path(c['runtime_root'])/'artifacts'
p=root/'stream_petr_r50_flash_704_bs2_seq_90e.part';dst=p.with_suffix('.pth')
r=dict(started_utc=datetime.now(timezone.utc).isoformat(),reason='User reported VPN restored in current conversation; no message ID available',proxy='http://127.0.0.1:7897',TLS_verify=True,url=c['checkpoint_source'],expected_bytes=c['checkpoint_expected_bytes'],prior_failure_preserved=True,max_attempts=1,wall_budget_s=600)
t=time.monotonic();n=p.stat().st_size;r['resume_from_bytes']=n
try:
 assert not dst.exists();assert shutil.disk_usage(root).free>2*c['checkpoint_expected_bytes']
 op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':r['proxy'],'http':r['proxy']}))
 req=urllib.request.Request(c['checkpoint_source'],headers={'Range':f'bytes={n}-','User-Agent':'HS-Drive-M8-J11'})
 with op.open(req,timeout=30) as s:
  r.update(http_status=s.status,final_host=urllib.parse.urlsplit(s.url).hostname,content_range=s.headers.get('Content-Range'),content_length=s.headers.get('Content-Length'))
  assert r['final_host'] in ('release-assets.githubusercontent.com','objects.githubusercontent.com')
  if s.status==206:
   assert r['content_range']==f'bytes {n}-{c["checkpoint_expected_bytes"]-1}/{c["checkpoint_expected_bytes"]}'
   mode='ab'
  else:
   assert s.status==200;mode='wb';n=0;r['resume_not_supported_restarted_same_asset']=True
  with p.open(mode) as f:
   while True:
    b=s.read(1024*1024)
    if not b:break
    n+=len(b);assert n<=c['checkpoint_expected_bytes'];assert time.monotonic()-t<600
    f.write(b)
 assert n==c['checkpoint_expected_bytes'];assert zipfile.is_zipfile(p)
 with zipfile.ZipFile(p) as z:assert z.testzip() is None
 r.update(bytes=n,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),archive_integrity=True,status='OfficialCheckpointDownloaded',official_digest=None)
 p.rename(dst);r['path']=str(dst)
except Exception as e:r.update(status='ResumeFailed',error_type=type(e).__name__,reason_detail=str(getattr(e,'reason',type(e).__name__))[:200])
r['elapsed_s']=time.monotonic()-t
Path('reports/m8j11_sources/checkpoint_resume.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if r['status']=='OfficialCheckpointDownloaded':
 c['initial_asset_sha256']=r['sha256'];Path('configs/m8j11_protocol.json').write_text(json.dumps(c,indent=2)+'\n')
else:raise SystemExit(1)
