"""Why nominal small actors fail support: raw semantics vs trusted association."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.m5k_common import rows,write,asset
from simulator.recorder import file_hash
from datasets.m5f_pixels import homogeneous,PEDESTRIAN,VEHICLE

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();records=[]
 for row in rows(a.split):
  frame=json.loads(Path(row['raw_frame']).read_text());root=Path(row['raw_frame']).parent.parent;label=np.load(row['pixel_label'])
  for ci,cam in enumerate(row['camera_names']):
   rec=frame['cameras'][cam]
   with np.load(root/rec['instance'])as z:s=z['semantic'];ins=z['instance']
   with np.load(root/rec['depth'])as z:dep=z['depth_m']
   interior=homogeneous(s,ins)
   for typ,tags,k in [('vehicle',VEHICLE,1),('person',PEDESTRIAN,2)]:
    raw=np.isin(s,tags)&(dep>=20)&(dep<35);valid=(label['classes'][ci]==k)&(label['depth'][ci]>=20)&(label['depth'][ci]<35);records.append(dict(group=row['group'],sample_id=row['sample_id'],camera=cam,type=typ,raw_semantic_20_35=int(raw.sum()),interior_raw=int((raw&interior).sum()),reliable=int(valid.sum()),associated=int((valid&(label['actor'][ci]>0)).sum()),raw_p8=int(raw[4::8,4::8].sum()),interior_p8=int((raw&interior)[4::8,4::8].sum()),reliable_p8=int(valid[4::8,4::8].sum())))
 path=f'runs/m5k_visibility_{a.split}.json';write(path,dict(records=records));agg=[]
 for typ in ['vehicle','person']:
  rr=[r for r in records if r['type']==typ];agg.append(dict(type=typ,**{k:sum(r[k]for r in rr)for k in ['raw_semantic_20_35','interior_raw','reliable','associated','raw_p8','interior_p8','reliable_p8']}))
 write(f'reports/m5k_visibility_{a.split}.json',dict(kind='offline_actual_CARLA_renderer_diagnostic',records=agg,raw=asset(path),limits='Rawsemantic includesunassociated mapactors, not reliable supervision. Cone cannot be identified byclassalone; class21 not relabeled. Counts expose effect of semantic/instance interior beforedepthstrictmask; no rules relaxed.',source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
