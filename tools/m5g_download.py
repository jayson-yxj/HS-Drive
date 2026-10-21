"""Official model download with verified HTTP ranges and final official SHA prefix."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request,hashlib,json,torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from simulator.recorder import atomic_json,file_hash


def main():
    w=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1;root=Path('/root/gpufree-data/hs-drive-cache/weights/m5g');dest=root/w.url.split('/')[-1];parts=root/'m5g_parts';parts.mkdir(exist_ok=True);size=185828065;chunk=1048576
    def get(i):
        a=i*chunk;b=min(size-1,a+chunk-1);p=parts/f'{i:04d}.part'
        if p.exists()and p.stat().st_size==b-a+1:return p
        for attempt in range(4):
            try:
                req=urllib.request.Request(w.url+f'?m5gpart={i}',headers={'Range':f'bytes={a}-{b}'})
                with urllib.request.urlopen(req,timeout=40)as r:
                    assert r.status==206 and r.headers['Content-Range']==f'bytes {a}-{b}/{size}';data=r.read();assert len(data)==b-a+1;p.write_bytes(data)
                print(i,flush=True);return p
            except Exception:
                if attempt==3:raise
    with ThreadPoolExecutor(max_workers=12)as pool:files=list(pool.map(get,range((size+chunk-1)//chunk)))
    if dest.exists():raise FileExistsError(dest)
    with dest.open('wb')as f:
        for p in files:f.write(p.read_bytes())
    assert file_hash(dest).startswith('73cbd019')
    atomic_json('reports/m5g_pretrained_provenance.json',dict(kind='actual_official_weight_download_with_verified_ranges',url=w.url,sha256=file_hash(dest),torchvision=torchvision.__version__,categories=w.meta['categories'],parameters=w.meta['num_params'],code_license='BSD-3-Clause https://github.com/pytorch/vision/blob/main/LICENSE',weight_terms='Torchvisionweights inherittrainingdata considerations: https://docs.pytorch.org/vision/stable/models.html. COCOresearchreference, no blanketcommerciallicenseclaim.',unsupported=['trafficcone'],input='native640x384 RGB internalshortside800 max1333',fine_tuned=False,source_sha256=file_hash(__file__)))

if __name__=='__main__':main()
