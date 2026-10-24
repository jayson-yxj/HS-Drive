"""Extract build headers from verified NVIDIA CUDA11 distributions only."""
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from tools.m8a_p1_common import ROOT,CFG,LOG,asset,write,provenance

def main():
    entries=json.loads((LOG/'cuda_library_headers_sources.json').read_text())['files']
    def fetch(e):
        path=ROOT/'wheels'/e['filename']
        op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':CFG['proxy']}))
        with op.open(e['url'],timeout=60) as src,path.open('xb') as dst:shutil.copyfileobj(src,dst)
        assert path.stat().st_size==e['size'] and asset(path)['sha256']==e['digests']['sha256']
        headers=[]
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if '/include/' in member and not member.endswith('/'):
                    target=ROOT/'cuda-11.8/include'/member.split('/include/',1)[1]
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member));headers.append(asset(target))
        print(e['filename'],'headers',len(headers),flush=True)
        return dict(distribution=asset(path),url=e['url'],headers=headers,shared_libraries_installed=False)
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,entries))
    write(LOG/'cuda_library_headers.json',dict(**provenance('ActualVerifiedCUDA11BuildHeaders'),records=records))
if __name__=='__main__':main()
