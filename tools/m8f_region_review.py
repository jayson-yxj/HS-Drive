"""Pixel-exact reviewed source regions; masks are evaluator-only, never safe free."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m8f_common import *

def main():
 out=ROOT/'reviewed_regions';out.mkdir(exist_ok=False);rows=[]
 for path in sorted((ROOT/'audit').glob('*.json')):
  a=json.loads(path.read_text());f=next(f for f in a['frames'] if f['phase']=='present');s=json.loads(Path(f['sidecar']['path']).read_text())
  for cam,files in s['sensors'].items():
   z=np.load(files['instance']['path'],allow_pickle=False);depth=np.load(files['depth']['path'],allow_pickle=False)['depth_m'];sem=z['semantic'];ids=z['instance']
   valid=np.zeros_like(sem,dtype=bool);target=np.zeros_like(sem,dtype=bool);ambiguous=np.zeros_like(sem,dtype=bool);refs=[]
   for o in f['objects']:
    for v in o['views']:
     if v['camera']!=cam:continue
     mask=(ids==v['render_encoding_id'])&(sem==v['semantic_id'])&(depth>=1.5)&(depth<80)&np.isfinite(depth)
     mask[:151]=False
     if o['identity_grade']=='AmbiguousIdentity':ambiguous|=mask
     if not v['accepted'] or not o['identity_grade'].startswith('RenderLinked'):continue
     valid|=mask
     if o['petr_class'] in ['car','truck','pedestrian','traffic_cone','barrier']:target|=mask
     refs.append(dict(entity_key=o['entity_key'],class_name=o['petr_class'],semantic_id=v['semantic_id'],render_encoding_id=v['render_encoding_id'],pixel_count=int(mask.sum())))
   valid &= ~ambiguous;target &= valid
   p=out/(a['group']+'_'+cam+'.npz')
   with p.open('xb') as stream:np.savez_compressed(stream,source_identity_valid=valid,supported_taxonomy=target,ambiguous=ambiguous)
   rows.append(dict(group=a['group'],camera=cam,frame_id=s['frame_id'],source=s['source'],mask=asset(p),regions=refs,source_supported_pixels=int(valid.sum()),supported_target_pixels=int(target.sum()),ambiguous_pixels=int(ambiguous.sum())))
 write('reports/m8f_background_review_regions.json',dict(**provenance('OfflineActualPixelSourceRegionAudit'),records=rows,
    scope='Only recorded depth-valid retained pixels, not bounding rectangles or full images. Complement is unreviewed/unknown, not negative or free.',
    exact_mesh_truth=False,physical_precision_available=False,model_input=False))
 print('reviewed camera regions',len(rows),'source pixels',sum(x['source_supported_pixels'] for x in rows))
if __name__=='__main__':main()
