"""Per-camera self-body returns and known dynamic target surface agreement."""
import argparse
from pathlib import Path
import numpy as np
from datasets.carla_dataset import CarlaDataset
from geometry.oracle_bev import depth_to_ego
from geometry.reference import transform_points
from simulator.recorder import atomic_json
from tools.validate_recording import validate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True)
    p.add_argument('--output',required=True);p.add_argument('--frame-step',type=int,default=5)
    a=p.parse_args();ds=CarlaDataset(a.scene);integrity=validate(a.scene);rows=[];body={c:[] for c in ds.calibration['cameras']}
    for i in range(0,len(ds),a.frame_step):
        sample,_,depths,objects=ds.read(i)
        ego_obj=next(o for o in objects if o['is_ego'])
        box=np.asarray(ego_obj['corners_hs_ego_m']);lo=box.min(0);hi=box.max(0)
        target_id=ds.manifest.get('target_actor_id')
        target=next((o for o in objects if o['actor_id']==target_id),None)
        for name,cam in ds.calibration['cameras'].items():
            pts=depth_to_ego(depths[name],cam['K'],cam['T_hs_ego_from_opt'],depth_convention='optical_z',stride=2,min_m=.05,max_m=80)
            inside=np.all((pts>=lo-.02)&(pts<=hi+.02),axis=1)
            body[name].append({'frame_id':sample['frame_id'],'valid_points':len(pts),'inside_ego_bbox':int(inside.sum()),'fraction':float(inside.mean()) if len(pts) else 1})
            if target and name=='front':
                corners=np.asarray(target['corners_hs_ego_m']);center=corners.mean(0)
                target_lo,target_hi=corners.min(0),corners.max(0)
                in_target=np.all((pts>=target_lo-.08)&(pts<=target_hi+.08),axis=1)
                world_center=transform_points([center],sample['ego']['T_world_from_ego'])[0]
                rows.append({'frame_id':sample['frame_id'],'target_center_world_m':world_center.tolist(),
                    'surface_points_in_gt_bbox':int(in_target.sum()),'target_center_ego_m':center.tolist()})
    displacement=float(np.linalg.norm(np.array(rows[-1]['target_center_world_m'])-rows[0]['target_center_world_m'])) if rows else 0
    report={'scene':a.scene,'recording_integrity':integrity,'self_body':body,'dynamic_target':rows,'target_displacement_m':displacement,
        'dynamic_target_passed':bool(integrity['recording_integrity_checks_passed'] and len(rows)>=3 and displacement>=1 and min(r['surface_points_in_gt_bbox'] for r in rows)>=30),
        'self_body_review_passed':all(max(r['fraction'] for r in values)<.1 for values in body.values()),
        'scope':'Ego AABB conservative body-return audit; dynamic GT bbox contains visible surface, not exact mesh. No claim of complete blind-spot coverage.'}
    atomic_json(a.output,report);print({k:v for k,v in report.items() if k not in ('self_body','dynamic_target','recording_integrity')})


if __name__=='__main__':main()
