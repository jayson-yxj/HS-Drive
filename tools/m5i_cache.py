"""Cache frozen MobileNet RGB levels; historicalResNet caches reused by hash."""
import json,time
from pathlib import Path
import torch
from models.m5i_visual import MobilePixel
from tools.m5i_data import rows
from tools.m5f_common import inputs
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    out=Path('runs/m5i_mobile_cache');out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);w=json.loads(Path('reports/m5i_pretrained_provenance.json').read_text())['candidate'];net=MobilePixel(w['path']).cuda().eval();rr=rows();records=[];start=time.perf_counter()
    for r in rr:
        x=inputs(r)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):lev=net.levels(x['images'])
        p=out/(r['sample_id'].replace('/','_')+'.pt');torch.save(dict(levels=[v.cpu()for v in lev]),p);records.append(dict(sample_id=r['sample_id'],path=str(p),sha256=file_hash(p)));print(r['sample_id'],flush=True)
    atomic_json(out/'index.json',dict(kind='actual_GPU_RGB_only_pretrained_mobile_features',records=records,source_sha256=file_hash(__file__),weights_sha256=w['sha256'],elapsed_s=time.perf_counter()-start))
if __name__=='__main__':main()
