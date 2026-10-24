"""Bounded ranged download from official URLs, strict lengths/hash/TLS checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from tools.m8a_p1_common import ROOT,CFG,LOG,write,provenance,asset


def download(url,destination,total,expected,workers=16,chunk=4*1024*1024):
    dest=Path(destination);parts=dest.parent/(dest.name+'.parts');parts.mkdir(exist_ok=False)
    started=time.monotonic();events=[]
    def fetch(index):
        lo=index*chunk;hi=min(total,lo+chunk)-1
        for attempt in range(2):
            begin=time.monotonic()
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':CFG['proxy']}))
                req=urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(lo,hi),'Accept-Encoding':'identity'})
                with opener.open(req,timeout=90) as src:
                    assert src.status==206,src.status
                    assert src.headers.get('Content-Range')=='bytes %d-%d/%d'%(lo,hi,total),src.headers.get('Content-Range')
                    content=src.read(hi-lo+2)
                    assert len(content)==hi-lo+1,(len(content),hi-lo+1)
                (parts/('%06d'%index)).write_bytes(content)
                events.append(dict(index=index,attempt=attempt,bytes=len(content),elapsed_s=time.monotonic()-begin,status='Passed'))
                print(index,'complete',len(content),flush=True);return
            except Exception as error:
                events.append(dict(index=index,attempt=attempt,status='Failed',error=repr(error),elapsed_s=time.monotonic()-begin))
                if attempt==1:raise
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(fetch,range((total+chunk-1)//chunk)))
        with dest.open('xb') as out:
            for path in sorted(parts.iterdir()):out.write(path.read_bytes())
        result=asset(dest);assert result['sha256']==expected,(result,expected)
    finally:
        write(LOG/(dest.name+'.download.json'),dict(**provenance('ActualOfficialRangeDownload'),url=url,total=total,expected_sha256=expected,
            elapsed_s=time.monotonic()-started,events=events,proxy_explicit=True,tls_verification=True))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--file',required=True);p.add_argument('--bytes',type=int,required=True);p.add_argument('--sha',required=True)
    a=p.parse_args();print(download(a.url,ROOT/'wheels'/a.file,a.bytes,a.sha))
