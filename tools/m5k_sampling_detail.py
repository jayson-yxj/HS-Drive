"""Postfreeze perposition sampling and empty-scene safeguards from raw records."""
import json
from pathlib import Path
from tools.m5k_common import write,asset
from tools.m5j_metrics import aggregate

def main():
 for split in ['val','test']:
  path=f'runs/m5k_sampling_{split}/raw.json';raw=json.loads(Path(path).read_text())['records'];records=[]
  for seed,dep,sampling,group in sorted({(r['seed'],r['depth'],r['sampling'],r['group'])for r in raw}):
   rr=[r for r in raw if r['seed']==seed and r['depth']==dep and r['sampling']==sampling and r['group']==group]
   for event in ['all','empty','removed']:
    selected=rr if event=='all'else[r for r in rr if r['configuration']==(0 if event=='empty' else 25)]
    records.append(dict(seed=seed,depth=dep,sampling=sampling,group=group,event=event,**aggregate([r['metrics']for r in selected])))
  write(f'reports/m5k_sampling_positions_{split}.json',dict(kind='offline_postfreeze_frozen_sampling_statistics',records=records,raw=asset(path),limits='Empty/removed mayretain existingmapobjects; FPR onlyconfirmedbackground. Unknown separately high. No retuning.'))
if __name__=='__main__':main()
