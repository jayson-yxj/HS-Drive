"""Actual P4 feature/latency checks, supervision eligibility controls training."""
import json,time
from pathlib import Path
import torch,numpy as np
from tools.m5l_common import rows,write,asset,summary
from tools.m5i_data import load
from tools.m5f_common import inputs
from models.m5l_p4 import P4LocalFeatures,P4LocalDepth,sample_map
from training.runtime import prepare_cuda
from simulator.recorder import file_hash

@torch.no_grad()
def main():
 prepare_cuda();torch.set_num_threads(2);cfg=json.loads(Path('configs/m5j_models.json').read_text());records=[]
 for seed in [101,102]:
  mobile,_=load(next(r for r in cfg['pixel']if r['name']=='mobile_gt'and r['seed']==seed));net=P4LocalFeatures(mobile).cuda().eval();x=inputs(rows('train')[1]);uv=torch.tensor([[320.,192.],[324.,192.],[320.,196.]],device='cuda')[None].expand(6,-1,-1)
  with torch.autocast('cuda',dtype=torch.bfloat16):f=net(x['images'],uv)
  shifted=x['images'].clone();shifted[:,:,:,180:208,308:336]=0
  with torch.autocast('cuda',dtype=torch.bfloat16):g=net(shifted,uv)
  delta=float((f-g).abs().mean());assert delta>0 and f.shape==(6,3,88);times=[];torch.cuda.reset_peak_memory_stats()
  uv=torch.stack(torch.meshgrid(torch.arange(2,384,4,device='cuda'),torch.arange(2,640,4,device='cuda'),indexing='ij'),-1).flip(-1).reshape(-1,2)[::32][None].expand(6,-1,-1).float()
  for i in range(25):
   torch.cuda.synchronize();t=time.perf_counter()
   with torch.autocast('cuda',dtype=torch.bfloat16):features=net(x['images'],uv)
   torch.cuda.synchronize()
   if i>=5:times.append((time.perf_counter()-t)*1000)
  records.append(dict(seed=seed,local_rays=uv.shape[0]*uv.shape[1],feature_shape=list(features.shape),shallow_channels=24,semantic_channels=64,occlusion_delta=delta,latency_ms=summary(times),peak_mib=torch.cuda.max_memory_allocated()/2**20))
 support=json.loads(Path('reports/m5l_sampling_failure_analysis.json').read_text());assert not support['p4_training_allowed'],'P4 eligible: execute declared boundedtraining before finalizing'
 write('reports/m5l_p4_feature_quality.json',dict(kind='actual_RTX4090_realP4feature_probe',records=records,training='NotRun_StrictSupportGate',support=asset('reports/m5l_sampling_failure_analysis.json'),source_sha256=file_hash(__file__),limits='S4 features fromMobilelayer3 actualRGB, P8 semantic fusion sampled atconvcenters. No trainedP4Depth/C performance claim; zeroresidualhead onlyinterface. No allsixDenseP4Splat. Localprobe legalfixedUV noGTROI.'))
if __name__=='__main__':main()
