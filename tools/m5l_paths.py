"""Original P8 and interpolated P4 diagnosis; no untrained C performance invented."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from tools.m5l_common import rows,cache_index,write,asset,group_depth
from tools.m5k_alignment import sample_gt
from tools.m5l_depth_metrics import depth_records
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5d_stride8 import cache_p8
from models.m5k_geometry import resample_probability
from training.runtime import prepare_cuda

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();prepare_cuda();torch.set_num_threads(2);net=encoder_load();records=[]
 for row in rows(a.split):
  x=inputs(row);d=pixel_label(row);c=cache_p8(net,x)
  for stride,name,pr in [(8,'original_p8',c['probs']),(4,'interpolated_p4',resample_probability(c['probs'],(96,160)))]:
   gt=sample_gt(d,stride);records+=depth_records(net,c,gt,pr,name,row)
  print(row['sample_id'],flush=True)
 raw=f'runs/m5l_paths_{a.split}.json';write(raw,dict(records=records));write(f'reports/m5l_paths_{a.split}.json',dict(kind='actual_GPU_readonly_feature_center_probability_geometry',depth=group_depth(records,['method','scope','band']),positions=group_depth([r for r in records if r['scope']=='vehicle'],['method','scope','band','group']),raw=asset(raw),limits='P8 vs P4 differentraydensities, no matched-mask causaleffect claim. P4probability isinterpolation only. CactualP4Depth nottrained supportgatefailed. Exactcenters UV4vs2, same strictdefinition; correctlayer requiresbininside.'))
if __name__=='__main__':main()
