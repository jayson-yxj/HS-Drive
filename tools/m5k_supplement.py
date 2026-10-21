"""Postfreeze diagnostic reports do not alter any model, mask or selection."""
import json
from pathlib import Path
from tools.m5k_common import write,asset,group_depth
from tools.m5k_alignment import RANGES

def main():
 for split in ['val','test']:
  raw=json.loads(Path(f'runs/m5k_alignment_{split}/raw.json').read_text());rr=[r for r in raw['depth']if r['scope']in ['actor','vehicle','person','cone']];write(f'reports/m5k_camera_distance_{split}.json',dict(kind='offline_postfreeze_locked_depth_aggregation',records=group_depth(rr,['method','scope','band','camera']),raw=asset(f'runs/m5k_alignment_{split}/raw.json')))
 # Replay diagnosis and near Actor protection explicitly contrasted.
 aligned=json.loads(Path('reports/m5k_alignment_val.json').read_text());r=[]
 for method in ['original_p8','native16_to_p8','replay101_p8','replay102_p8']:
  for scope in ['all','actor']:
   d=next((v for v in aligned['depth']if v['method']==method and v['scope']==scope and v['band']=='near'),None)
   if d:r.append(d)
 write('reports/m5k_near_protection.json',dict(kind='offline_actual_GPU_nearquality_review',new_val=r,originalM4=asset('reports/m5k_near_replay.json'),positions=asset('reports/m5k_near_position_val.json'),limits='M4 replay allsurface dominated; cannotreplace current nearActor spatial mass. Research candidate must pass all gates; no MAE-only approval.'))
if __name__=='__main__':main()
