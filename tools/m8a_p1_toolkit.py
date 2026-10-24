"""Fetch checksum-verified CUDA 11.8 build components, without system installation."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from tools.m8a_p1_common import ROOT, CFG, LOG, write, provenance, asset

BASE='https://developer.download.nvidia.com/compute/cuda/redist/'

def main():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':CFG['proxy']}))
    manifest=opener.open(BASE+'redistrib_11.8.0.json',timeout=30).read()
    (ROOT/'wheels/redistrib_11.8.0.json').write_bytes(manifest)
    data=json.loads(manifest)
    target=ROOT/'cuda-11.8';target.mkdir(exist_ok=False)
    records=[]
    for name in ['cuda_nvcc','cuda_cudart','cuda_cccl']:
        entry=data[name]['linux-x86_64'];url=BASE+entry['relative_path']
        path=ROOT/'wheels'/Path(entry['relative_path']).name
        with opener.open(url,timeout=60) as src,path.open('xb') as dst:
            while True:
                block=src.read(1024*1024)
                if not block:break
                dst.write(block)
        info=asset(path)
        assert info['sha256']==entry['sha256'],name
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                parts=Path(member.name).parts[1:]
                if not parts:continue
                assert '..' not in parts and not member.name.startswith('/')
                member.name=str(Path(*parts))
                if member.issym() or member.islnk():
                    assert not Path(member.linkname).is_absolute() and '..' not in Path(member.linkname).parts
                archive.extract(member,target)
        records.append(dict(component=name,url=url,manifest=entry,file=info))
        print(name,'verified',flush=True)
    write(LOG/'isolated_cuda_toolkit.json',dict(**provenance('ActualChecksumVerifiedIsolatedToolkit'),components=records,root=str(target)))

if __name__=='__main__':main()
