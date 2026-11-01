"""Bounded downloads of predeclared official assets; no repository-wide snapshot."""
import argparse,hashlib,json,os,time
from pathlib import Path
import requests


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--finish',action='store_true');a=ap.parse_args()
    plan=json.loads(Path('reports/m8j12_sources/asset_plan.json').read_text())
    root=Path('/root/gpufree-data/hs-drive-experiments/m8j12-sparsedrivev2/artifacts');root.mkdir(exist_ok=True)
    proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
    records=[]
    for f in plan['files']:
        name=f['rfilename'];dst=root/name;part=root/(name+'.partial')
        url='https://huggingface.co/'+plan['repository']+'/resolve/'+plan['revision']+'/'+name
        record={'file':name,'url':url,'expected_bytes':f['size'],'expected_sha256':f['lfs']['sha256'],'attempts':[]}
        attempts=1 if a.finish else 2
        for attempt in range(attempts):
            start=time.monotonic()
            try:
                offset=part.stat().st_size if part.exists() else 0
                if not dst.exists() and offset < f['size']:
                    with requests.get(url,proxies=proxies,stream=True,headers={'Range':'bytes='+str(offset)+'-'} if offset else {},timeout=(10,30)) as r:
                        r.raise_for_status()
                        if offset and r.status_code!=206:raise RuntimeError('ResumeNotHonored')
                        if offset and not r.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):raise RuntimeError('ResumeOffsetMismatch')
                        with part.open('ab' if offset else 'wb') as w:
                            for chunk in r.iter_content(1024*1024):
                                w.write(chunk)
                                if w.tell()>f['size'] or time.monotonic()-start>300:raise RuntimeError('DownloadBudgetExceeded')
                path=dst if dst.exists() else part
                h=hashlib.sha256(path.read_bytes()).hexdigest()
                assert path.stat().st_size==f['size'] and h==f['lfs']['sha256'],'SizeOrHashMismatch'
                if not dst.exists():part.rename(dst)
                record.update(actual_bytes=dst.stat().st_size,sha256=h,status='VerifiedOfficialLFSIdentity')
                record['attempts'].append({'attempt':attempt,'elapsed_s':time.monotonic()-start,'status':'Complete'})
                break
            except Exception as e:
                # Do not log signed redirect URLs or any proxy credential.
                record['attempts'].append({'attempt':attempt,'elapsed_s':time.monotonic()-start,'error_type':type(e).__name__})
                if attempt==attempts-1:record['status']='Failed'
        records.append(record)
        Path('reports/m8j12_sources/asset_finish.json' if a.finish else 'reports/m8j12_sources/asset_download.json').write_text(json.dumps(records,indent=2)+'\n')
        print(name,record['status'],flush=True)
        if record['status']=='Failed':raise RuntimeError('AssetDownloadFailed:'+name)

if __name__=='__main__':main()
