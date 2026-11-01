"""Fixed first8 historical accepted inputs, at most3 calls/path/input, no model."""
import json,os,time
from pathlib import Path
import numpy as np
import torch,cv2
from mmcv import Config
import mmcv.parallel
from mmdet.datasets import pipelines
import projects.mmdet3d_plugin
from tools.m8j12_runtime import observation_batch,protocol
from tools.m8j13_preprocess import OfficialPreprocessor,equal_nested
from datasets.m8j12_contract import official_camera_contract
from tools.m8j13_common import write,append

ROOT=Path('/root/gpufree-data/hs-drive-private/m8j13-input-pipeline-20261101')
OLD=Path('/root/gpufree-data/hs-drive-private/m8j13-parked-shadow-20261101/live')

def main():
    c=protocol();rt=Path(c['runtime_root']);cfg=Config.fromfile(str(rt/'runtime-source'/c['config']))
    camera=official_camera_contract(rt/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
    fixed=sorted((OLD/'diagnostics').glob('*.json'))[:8]
    write(ROOT/'profile_request.json',dict(inputs=[x.name for x in fixed],repeats=3,full_forwards=0,
        affinity=sorted(os.sched_getaffinity(0)),cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),torch_threads=torch.get_num_threads(),torch_interop=torch.get_num_interop_threads(),opencv_threads=cv2.getNumThreads()))
    candidate=OfficialPreprocessor(cfg,camera);records=[]
    original_encode=cv2.imencode;original_decode=cv2.imdecode;original_compose=pipelines.Compose;original_collate=mmcv.parallel.collate;original_scatter=mmcv.parallel.scatter
    for f in fixed:
        row=json.loads((OLD/'requests'/f.name).read_text());images=dict(np.load((OLD/'requests'/f.name).with_suffix('.npz'),allow_pickle=False))
        for rep in range(3):
            old_t={}
            def wrap(key,fn,sync=False):
                def call(*args,**kwargs):
                    t=time.monotonic();out=fn(*args,**kwargs)
                    if sync:torch.cuda.synchronize()
                    old_t[key]=old_t.get(key,0)+(time.monotonic()-t)*1000;return out
                return call
            class Compose:
                def __init__(self,*a,**kw):
                    t=time.monotonic();self.p=original_compose(*a,**kw);old_t['compose_construct_ms']=(time.monotonic()-t)*1000
                def __call__(self,data):
                    for tr in self.p.transforms:data=wrap(type(tr).__name__+'_ms',tr)(data)
                    return data
            cv2.imencode=wrap('jpeg_encode_ms',original_encode);cv2.imdecode=wrap('jpeg_decode_ms',original_decode)
            pipelines.Compose=Compose;mmcv.parallel.collate=wrap('collate_ms',original_collate);mmcv.parallel.scatter=wrap('scatter_sync_ms',original_scatter,True)
            append(ROOT/'preprocess_budget.jsonl',dict(input=f.name,path='old',repeat=rep))
            try:
                start=time.monotonic();old=observation_batch(cfg,row,images,camera);torch.cuda.synchronize();old_t['total_ms']=(time.monotonic()-start)*1000
            finally:
                cv2.imencode=original_encode;cv2.imdecode=original_decode;pipelines.Compose=original_compose;mmcv.parallel.collate=original_collate;mmcv.parallel.scatter=original_scatter
            append(ROOT/'preprocess_budget.jsonl',dict(input=f.name,path='candidate',repeat=rep))
            start=time.monotonic();cpu,new_t=candidate.cpu(row,images);new,gpu=candidate.gpu(cpu);new_t['scatter_sync_ms']=gpu;new_t['total_ms']=(time.monotonic()-start)*1000
            equal_nested(old,new);records.append(dict(input=f.name,repeat=rep,old=old_t,candidate=new_t,exact_equal=True))
            del old,new,cpu
    write(ROOT/'profile_result.json',dict(status='ExactInputEquivalencePassed',records=records,full_forwards=0))
    for path in ['old','candidate']:
        print(path,{k:round(float(np.median([x[path].get(k,0) for x in records])),3) for k in records[-1][path]})

if __name__=='__main__':main()
