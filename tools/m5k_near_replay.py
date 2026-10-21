"""Read-only original M4 validation replay for stride alignment preservation."""
import json
from pathlib import Path
import numpy as np,torch
from datasets.training import HSDriveDataset
from tools.m5a_learning import encoder_load
from models.m5b_cached_encoder import cache_rgb
from models.m5d_stride8 import cache_p8
from models.m5k_geometry import resample_probability
from tools.m5k_common import write,asset
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

@torch.no_grad()
def main():
 prepare_cuda();torch.set_num_threads(2);net=encoder_load();man=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());data={s:HSDriveDataset(man['config']['index'],'val',feature_stride=s)for s in [8,16]};look={s:{w[-1]['sample_id']:i for i,w in enumerate(d.windows)}for s,d in data.items()};records=[]
 for sid in man['val_sample_ids']:
  sample=data[8][look[8][sid]];x={k:v[None].cuda()for k,v in sample.inputs.items()};c8=cache_p8(net,x);c16=cache_rgb(net,x)
  for method,pr,stride in [('original_p8',c8['probs'],8),('native16_to_p8',resample_probability(c16['probs'],(48,80)),8),('native16',c16['probs'],16)]:
   t=data[stride][look[stride][sid]].targets;dep=t['depth_m'][None].cuda();m=t['depth_valid'][None].cuda()&(dep>=1.5)&(dep<10);exp=(pr*net.splat.depth_centers[None,None,:,None,None]).sum(2);delta=exp-dep;records.append(dict(sample_id=sid,method=method,n=int(m.sum()),abs_sum=float(delta.abs()[m].double().sum()),bias_sum=float(delta[m].double().sum())))
  print(sid,flush=True)
 write('runs/m5k_logs/m4_near_replay_raw.json',records);agg=[]
 for method in ['original_p8','native16_to_p8','native16']:
  rr=[r for r in records if r['method']==method];n=sum(r['n']for r in rr);agg.append(dict(method=method,n=n,mae=sum(r['abs_sum']for r in rr)/n,bias=sum(r['bias_sum']for r in rr)/n))
 write('reports/m5k_near_replay.json',dict(kind='actual_RTX4090_original_M4_validation_readonly',aggregate=agg,raw=asset('runs/m5k_logs/m4_near_replay_raw.json'),source_sha256=file_hash(__file__),limits='OriginalM4 depth valid; no Actor semantics in historicalreplay. NativeS16 distinctrays. P8 versusaligned sharemask. Not used to retune scientificmodel/testthreshold.'))
if __name__=='__main__':main()
