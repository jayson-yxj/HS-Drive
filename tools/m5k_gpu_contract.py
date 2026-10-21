"""Real GPU numeric checks for buffer parity, probability alignment and FP32 geometry."""
import json
from pathlib import Path
import numpy as np,torch
from tools.m5k_common import rows,write,asset
from tools.m5f_common import inputs
from simulator.m5k_input import bgra_to_pinned,to_gpu
from models.m5j_candidate import MobileBEVCandidate
from models.m5b_cached_encoder import cache_rgb
from models.m5k_geometry import resample_probability,sampling_masks
from models.m5k_depth_adapter import DepthResidual
from tools.m5g_depth_data import conditioned
from models.m5d_stride8 import cache_p8
from training.runtime import prepare_cuda
from simulator.recorder import file_hash

@torch.no_grad()
def main():
 prepare_cuda();torch.set_num_threads(2);row=rows('historical_val')[2];cpu=inputs(row,'cpu');rgb=(cpu['images'][0]*255).round().byte().permute(0,2,3,1).numpy();buffers=[np.concatenate([x[:,:,[2,1,0]],np.full((*x.shape[:2],1),255,np.uint8)],-1).tobytes()for x in rgb];p=bgra_to_pinned(buffers);gpu=to_gpu(p,{k:v.pin_memory()for k,v in cpu.items()if k!='images'});assert torch.equal(p.cpu(),(cpu['images']*255).round().byte());rgb_error=float((gpu['images'].cpu()-cpu['images']).abs().max());assert rgb_error<=1e-7
 candidate=MobileBEVCandidate(enable=True);out=candidate(**gpu);c16=cache_rgb(candidate.depth,gpu);pr=resample_probability(c16['probs'],(48,80));normalization=float((pr.sum(2)-1).abs().max());assert normalization<1e-6 and pr.dtype==torch.float32
 c=cache_p8(candidate.depth,gpu);f=conditioned(candidate.depth,c)
 with torch.autocast('cuda',dtype=torch.bfloat16):log=candidate.depth.depth_head(f.flatten(0,1)).float()
 adapter=DepthResidual(mobile_condition=True).cuda();assert torch.equal(adapter(f[0],log,out['pixel_foreground'][0,:,None,4::8,4::8]),log)
 assert all(v.dtype==torch.float32 for v in out['evidence']['scores'].values());rejected=[]
 for field in ['gt_depth','actor_aabb','instance','witness_mask']:
  try:candidate(**gpu,**{field:torch.ones(1,device='cuda')})
  except (ValueError,TypeError,KeyError):rejected.append(field)
 assert len(rejected)==4
 write('reports/m5k_gpu_contract.json',dict(kind='actual_RTX4090_numeric_checks',buffer_uint8_exact_parity=True,normalized_rgb_max_error=rgb_error,probability_normalization_max_error=normalization,fp32_geometry=True,zero_residual_exact_logits=True,forbidden_GT_inputs_rejected=rejected,source_sha256=file_hash(__file__),limits='Research interface check only, not trainedresidualqualification or drivingsafety.'))
if __name__=='__main__':main()
