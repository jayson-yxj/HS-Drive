"""Audit strict raster support by actual associated Actor, not semantic pixels alone."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.m5i_data import rows,ROOT,TEST
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args();out=Path(f'reports/m5i_depth_support_{a.split}.json')
    if out.exists():raise FileExistsError(out)
    report=json.loads(Path(f'reports/m5i_depth_{a.split}.json').read_text());rr={r['sample_id']:r for r in rows(TEST if a.split=='test'else ROOT)};records=[];cached=None;label=None;seen=set()
    for r in report['far_support']:
        row=rr[r['sample_id']]
        if cached!=r['sample_id']:cached=r['sample_id'];label=pixel_label(row)
        c=row['camera_names'].index(r['camera']);local=label['actor'][c,4::8,4::8];cls=label['classes'][c,4::8,4::8];actors={x['local_id']:x for x in row['actors'][c]['actors']};types={};classes={};uv=[]
        for y,x in r['uv_yx']:
            key=(r['sample_id'],c,y,x);assert key not in seen;seen.add(key)
            aid=int(local[y,x])
            if aid<=0 or aid not in actors:continue
            type_id=actors[aid]['type_id'];types[type_id]=types.get(type_id,0)+1;cl=str(int(cls[y,x]));classes[cl]=classes.get(cl,0)+1;uv.append([y,x,actors[aid]['actor_id']])
        records.append(dict(sample_id=r['sample_id'],group=r['group'],camera=r['camera'],semantic_rays=r['n'],associated_actor_rays=len(uv),types=types,classes=classes,uv_actor=uv))
    groups=sorted({r['group']for r in records if r['associated_actor_rays']});classes=sorted({k for r in records for k,v in r['classes'].items()if v});n=sum(r['associated_actor_rays']for r in records)
    atomic_json(out,dict(kind='offline_strict_GPU_mask_rays_and_associated_CARLA_actor_audit',records=records,semantic_rays=len(seen),associated_actor_rays=n,groups=groups,classes=classes,passed=n>=100 and len(groups)>=4 and len(classes)>=2,source_sha256=file_hash(__file__),limits='GTassociation uses rendererinstance and approximateActorAABB; not exactmesh. Frame-camera-UV unique but nearby rays and repeated backgrounds correlated. Qualification usesassociatedActor support, not unassociatedsemanticpositives.'))
if __name__=='__main__':main()
