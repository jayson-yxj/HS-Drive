"""Coverage audit counts actors/views/independent positions separately."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tools.m5i_data import rows,ROOT
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--suffix',default='');a=p.parse_args();dest=Path('reports/m5i_camera_coverage'+a.suffix+'.json')
    if dest.exists():raise FileExistsError(dest)
    rr=rows(a.root);records=[]
    for r in rr:
        d=pixel_label(r)
        for c,name in enumerate(r['camera_names']):
            for lo,hi in [(0,5),(5,10),(10,20),(20,35),(35,80)]:
                t=d['valid'][c]&d['target'][c]&(d['radial'][c]>=lo)&(d['radial'][c]<hi);actors=[]
                for actor in r['actors'][c]['actors']:
                    m=t&(d['actor'][c]==actor['local_id']);n=int(m.sum())
                    if n>=3:actors.append(dict(actor_id=actor['actor_id'],type_id=actor['type_id'],pixels=n,size='small'if n<=32 else'medium'if n<=512 else'large'))
                records.append(dict(group=r['group'],split=r['split'],sample_id=r['sample_id'],configuration=r['configuration'],camera=name,weather=r['weather'],distance=[lo,hi],positive_pixels=int(t.sum()),negative_pixels=int((d['valid'][c]&~d['target'][c]&(d['radial'][c]>=lo)&(d['radial'][c]<hi)).sum()),classes={str(k):int((t&(d['classes'][c]==k)).sum())for k in [1,2,3]},actors=actors))
    agg=[]
    for split in sorted({r['split']for r in records}):
        for cam in rr[0]['camera_names']:
            rec=[r for r in records if r['split']==split and r['camera']==cam];groups=sorted({r['group']for r in rec if r['actors']});views={(r['sample_id'],actor['actor_id'])for r in rec for actor in r['actors']};agg.append(dict(split=split,camera=cam,positive_actor_views=len(views),positive_groups=groups,group_count=len(groups),negative_pixels=sum(r['negative_pixels']for r in rec),class_groups={str(k):sorted({r['group']for r in rec if r['classes'][str(k)]>=3})for k in [1,2,3]}))
    cfg=json.loads(Path('configs/m5i_protocol.json').read_text())['coverage'];checks=[dict(split=r['split'],camera=r['camera'],passed=r['group_count']>=(cfg['minimum_train_positive_groups_per_camera']if r['split']=='train'else cfg['minimum_val_positive_groups_per_camera']))for r in agg if r['split']in['train','val']]
    class_checks=[]
    for split in ['train','val']:
        if split not in {r['split']for r in records}:continue
        for k in [1,2,3]:
            gs=sorted({r['group']for r in records if r['split']==split and r['classes'][str(k)]>=3});minimum=cfg['minimum_train_groups_per_class']if split=='train'else cfg['minimum_val_groups_per_class'];class_checks.append(dict(split=split,class_id=k,groups=gs,passed=len(gs)>=minimum))
    atomic_json(dest,dict(kind='offline_actual_CARLA_reliable_pixel_support',records=records,aggregate=agg,checks=checks,class_checks=class_checks,passed=all(c['passed']for c in checks+class_checks),source_sha256=file_hash(__file__),limits='Actor-view deduplicated acrossdistancebands; correlated views not independent physicalexamples. Counts supportobservedfirstsurface not allvoxels.'))
    print(agg,flush=True)
if __name__=='__main__':main()
