"""Bounded download of the single predeclared official model. No deserialization."""
import hashlib,json,time,urllib.request,urllib.error,zipfile
from pathlib import Path
from datetime import datetime,timezone


def main():
    cfg=json.loads(Path('configs/m8j11_protocol.json').read_text());root=Path(cfg['runtime_root'])/'artifacts';root.mkdir(exist_ok=False)
    target=root/'stream_petr_r50_flash_704_bs2_seq_90e.pth';tmp=target.with_suffix('.part')
    url=cfg['checkpoint_source'];record=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),url=url,TLS_verify=True,attempts=[],candidate=cfg['candidate'],official_expected_size=cfg['checkpoint_expected_bytes'],official_digest=None)
    start=time.monotonic()
    for attempt in range(2):
        if time.monotonic()-start>180:break
        row=dict(attempt=attempt+1)
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'HS-Drive-M8-J11','Cache-Control':'no-cache'})
            with urllib.request.urlopen(req,timeout=25) as response:
                final=urllib.parse.urlsplit(response.url)
                assert final.hostname in ('github.com','release-assets.githubusercontent.com','objects.githubusercontent.com')
                row.update(http_status=response.status,final_host=final.hostname,content_length=response.headers.get('Content-Length'))
                total=0;h=hashlib.sha256()
                with tmp.open('wb') as output:
                    while True:
                        data=response.read(1024*1024)
                        if not data:break
                        total+=len(data);assert total<=cfg['checkpoint_expected_bytes']
                        assert time.monotonic()-start<240,'DownloadWallBudget'
                        output.write(data);h.update(data)
                assert total==cfg['checkpoint_expected_bytes']
                assert zipfile.is_zipfile(tmp),'CheckpointNotZip'
                with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
                tmp.rename(target);row.update(success=True,bytes=total,sha256=h.hexdigest())
                record.update(status='OfficialCheckpointDownloaded',path=str(target),bytes=total,sha256=h.hexdigest(),archive_integrity=True)
        except Exception as exc:
            # URL query may contain short-lived signed credentials; never emit it.
            row.update(success=False,error_type=type(exc).__name__,http_status=getattr(exc,'code',None),error_reason=str(getattr(exc,'reason',type(exc).__name__))[:200])
        record['attempts'].append(row)
        if row['success']:break
    record.setdefault('status','AssetUnavailable_BoundedAttemptsExhausted')
    record['elapsed_s']=time.monotonic()-start
    Path('reports/m8j11_sources/checkpoint_download.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    return 0 if record['status']=='OfficialCheckpointDownloaded' else 1


if __name__=='__main__':raise SystemExit(main())
