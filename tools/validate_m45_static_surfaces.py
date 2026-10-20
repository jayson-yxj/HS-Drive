"""Actual-sequence above-ground surface registration; no learned features."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from datasets.carla_dataset import CarlaDataset
from datasets.occupancy_labels import sampled_rays
from geometry.reference import transform_points
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json


def surfaces(ds,index):
    frame,_,depth,objects=ds.read(index);points=[]
    for name,value in depth.items():
        c=ds.calibration['cameras'][name]
        _,end,_,_=sampled_rays(value,c['K'],c['T_hs_ego_from_opt'],stride=8)
        points.append(end)
    pts=np.concatenate(points)
    # Exclude near ground and car body. These recordings contain only ego in
    # actor snapshots; absent semantic labels still preclude a static GT claim.
    keep=(pts[:,2]>.5)&(pts[:,2]<4)&(pts[:,0]>-15)&(pts[:,0]<50)&(np.abs(pts[:,1])<25)
    for obj in objects:
        box=np.array(obj['corners_hs_ego_m']);keep&=~np.all((pts>=box.min(0)-.1)&(pts<=box.max(0)+.1),axis=1)
    return frame,pts[keep]


def nearest(a,b):
    a=torch.tensor(a,dtype=torch.float32,device='cuda');b=torch.tensor(b,dtype=torch.float32,device='cuda')
    with torch.inference_mode():
        dist=torch.cat([torch.cdist(c[None],b[None],compute_mode='donot_use_mm_for_euclid_dist')[0].min(-1).values for c in a.split(256)])
    return dict(count=len(a),median_m=float(dist.median()),p90_m=float(torch.quantile(dist,.9)),within_0_5m=float((dist<.5).float().mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m45_temporal_static_surfaces.json');a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    prepare_cuda();idx=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/hsd-dataset-v2-driving-accepted/index.json').read_text());scene={s['scene_id']:s for s in idx['scenes']};rows=[]
    for sid,i,j in [('m4drive_town04_0_retry',25,30),('m4drive_town01_30',50,55),('m4drive_town02_20',130,131),('town03_spawn45_static',0,5)]:
        ds=CarlaDataset(Path(idx['raw_root'])/scene[sid]['path']);fa,pa=surfaces(ds,i);fb,pb=surfaces(ds,j)
        rel=np.linalg.inv(np.array(fb['ego']['T_world_from_ego']))@np.array(fa['ego']['T_world_from_ego']);changed=transform_points(pa,rel)
        rows.append(dict(scene=sid,frames=[fa['frame_id'],fb['frame_id']],previous_points=len(pa),current_points=len(pb),before=nearest(pa,pb),after=nearest(changed,pb)))
    atomic_json(a.output,dict(schema_version='hsd-m45-static-registration-v1',results=rows,scope='Depth-patch-reliable, above-ground, non-actor surfaces; directed nearest-neighbor registration, no semantic correspondence or perfect static-object ground truth. Occlusion/visibility still differ; not a downstream temporal performance score.'))


if __name__=='__main__':main()
