"""Resume fixed official PyTorch wheel with bounded byte ranges and official hash."""
from pathlib import Path
import json,time,hashlib,concurrent.futures
import requests
ROOT=Path('/root/gpufree-data/hs-drive-experiments/m8j12-sparsedrivev2/wheels')
PROXY={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}

def download(url,name,expected):
    path=ROOT/name;part=ROOT/(name+'.partial');parts=ROOT/(name+'.parts');parts.mkdir(exist_ok=True)
    with requests.head(url,proxies=PROXY,allow_redirects=True,timeout=(10,30)) as r:
        r.raise_for_status();size=int(r.headers['Content-Length'])
    prefix=part.stat().st_size if part.exists() else 0
    ranges=[(i,min(i+64*1024*1024,size)-1) for i in range(prefix,size,64*1024*1024)]
    def get(pair):
        start,end=pair;p=parts/str(start)
        if p.exists() and p.stat().st_size==end-start+1:return
        for attempt in range(2):
            try:
                t=time.monotonic()
                with requests.get(url,proxies=PROXY,headers={'Range':'bytes=%d-%d'%(start,end)},stream=True,timeout=(10,30)) as r:
                    r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith('bytes %d-%d/'%(start,end))
                    with p.open('wb') as f:
                        for c in r.iter_content(1024*1024):
                            f.write(c)
                            if time.monotonic()-t>180:raise TimeoutError('RangeBudget')
                assert p.stat().st_size==end-start+1;return
            except Exception:
                if attempt:raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(get,ranges))
    with path.open('xb') as out:
        for source in ([part] if prefix else [])+[parts/str(x[0]) for x in ranges]:
            with source.open('rb') as f:
                while True:
                    b=f.read(8*1024*1024)
                    if not b:break
                    out.write(b)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert path.stat().st_size==size and h.hexdigest()==expected
    return {'url':url,'file':str(path),'bytes':size,'sha256':h.hexdigest(),'official_hash':expected,'resumed_prefix_bytes':prefix}

def main():
    ROOT.mkdir(exist_ok=True);records=[]
    for name,entry in [('torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl','torch_index_status.json'),('torchvision-0.15.2+cu118-cp38-cp38-linux_x86_64.whl','torchvision_index.json')]:
        d=json.loads((Path('reports/m8j12_sources')/entry).read_text());v=(d.get('matching_wheels') or d['matches'])[0];url,expected=v.split('#sha256=')
        records.append(download(url,name,expected));print(name,'verified',flush=True)
        Path('reports/m8j12_sources/wheel_download.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
