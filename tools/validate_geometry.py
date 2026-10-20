"""Bidirectional adjacent-camera and ego-motion compensated depth consistency."""
import argparse
from pathlib import Path
import numpy as np
import yaml
from datasets.carla_dataset import CarlaDataset
from geometry.consistency import compare_depths,summarize
from simulator.recorder import atomic_json
from tools.validate_recording import validate

RING=('front','front_left','rear_left','rear','rear_right','front_right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
    p.add_argument('--thresholds',default='configs/m2_acceptance.yaml')
    p.add_argument('--frame-step',type=int,default=10)
    a=p.parse_args();ds=CarlaDataset(a.scene)
    integrity=validate(a.scene)
    cfg=yaml.safe_load(Path(a.thresholds).read_text());cross=cfg['cross_view']
    records={};temporal={};negative={};poses=[];speeds=[];previous=None
    for index in range(0,len(ds),a.frame_step):
        sample,_,depths,_=ds.read(index)
        pose=np.array(sample['ego']['T_world_from_ego']);poses.append(pose);speeds.append(sample['ego'].get('speed_mps',0))
        cal=ds.calibration['cameras']
        for i,c1 in enumerate(RING):
            c2=RING[(i+1)%6]
            for src,dst in ((c1,c2),(c2,c1)):
                T=np.linalg.inv(cal[dst]['T_hs_ego_from_opt'])@np.asarray(cal[src]['T_hs_ego_from_opt'])
                result=compare_depths(depths[src],depths[dst],cal[src]['K'],cal[dst]['K'],T,cross)
                records.setdefault(src+'->'+dst,[]).append(result)
        if previous:
            prev_sample,prev_depths=previous
            prev_pose=np.array(prev_sample['ego']['T_world_from_ego'])
            for cam in RING:
                E=np.array(cal[cam]['T_hs_ego_from_opt'])
                T=np.linalg.inv(E)@np.linalg.inv(pose)@prev_pose@E
                result=compare_depths(prev_depths[cam],depths[cam],cal[cam]['K'],cal[cam]['K'],T,cross)
                temporal.setdefault(cam,[]).append(result)
                wrong=compare_depths(prev_depths[cam],depths[cam],cal[cam]['K'],cal[cam]['K'],np.eye(4),cross)
                negative.setdefault(cam,[]).append(wrong)
        previous=(sample,depths)
    def aggregate(items):
        out={}
        for name,values in items.items():
            errors=np.concatenate([v['errors_m'] for v in values]);summary=summarize(errors,cross['agreement_tolerance_m'])
            summary.update({key:sum(v.get(key,0) for v in values) for key in ('projected','smooth_valid','occluded','comparable')})
            summary['all_smooth']=summarize(np.concatenate([v.get('all_smooth_errors_m',[]) for v in values]))
            out[name]=summary
        return out
    pairs=aggregate(records);motion=aggregate(temporal)
    def passed(s,threshold):
        return s['count']>=threshold.get('min_points_per_direction',threshold.get('min_temporal_points',100)) and s['median_abs_m']<=threshold['median_abs_depth_m'] and s['p95_abs_m']<=threshold['p95_abs_depth_m'] and s['agreement_fraction']>=cross['min_agreement_fraction']
    displacement=float(np.linalg.norm(poses[-1][:3,3]-poses[0][:3,3]))
    report={'scene':a.scene,'mode':'oracle_geometry','thresholds':cfg,'frames_sampled':len(poses),
        'recording_integrity':integrity,
        'cross_view':pairs,'cross_view_passed':integrity['recording_integrity_checks_passed'] and all(passed(s,cross) for s in pairs.values()),
        'temporal':motion,'negative_control_no_ego_warp':aggregate(negative),'ego_displacement_m':displacement,'max_speed_mps':max(speeds),
        'motion_passed':integrity['recording_integrity_checks_passed'] and displacement>=cfg['motion']['min_translation_m'] and max(speeds)>=cfg['motion']['min_speed_mps'] and bool(motion) and all(passed(s,cfg['motion']) for s in motion.values()),
        'limitations':'Smooth visible surfaces 2-40m; depth discontinuities excluded; occlusions counted separately. Not occupancy or global perception accuracy.'}
    atomic_json(a.output,report)
    print({k:v for k,v in report.items() if k not in ('thresholds','cross_view','temporal','recording_integrity','negative_control_no_ego_warp')})


if __name__=='__main__':main()
