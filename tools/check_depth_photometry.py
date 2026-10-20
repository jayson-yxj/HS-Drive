"""Verify changing RGB photometry did not alter raw depth in a fixed scene."""
import argparse
import numpy as np
from datasets.carla_dataset import CarlaDataset
from simulator.recorder import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    first,second=CarlaDataset(a.first),CarlaDataset(a.second)
    s1,_,d1,_=first.read(-1);s2,_,d2,_=second.read(-1);results={}
    poses_match=np.allclose(s1['ego']['T_world_from_ego'],s2['ego']['T_world_from_ego'],atol=1e-4,rtol=0)
    for name in d1:
        if d1[name].shape!=d2[name].shape:raise ValueError('Compare equal resolutions only')
        mask=(d1[name]>=1.5)&(d1[name]<80)&(d2[name]>=1.5)&(d2[name]<80)
        difference=abs(d1[name][mask]-d2[name][mask])
        results[name]={'count':len(difference),'median_abs_m':float(np.median(difference)),
            'p95_abs_m':float(np.quantile(difference,.95)),'fraction_over_0_1m':float(np.mean(difference>.1))}
    same_rig=set(first.calibration['cameras'])==set(second.calibration['cameras']) and all(
        np.allclose(first.calibration['cameras'][name][key],second.calibration['cameras'][name][key],atol=1e-6,rtol=0)
        for name in first.calibration['cameras'] for key in ('K','T_hs_ego_from_opt'))
    same_target=np.allclose(first.manifest['target_pose_carla'],second.manifest['target_pose_carla'],atol=1e-4,rtol=0)
    passed=poses_match and same_rig and same_target and all(x['p95_abs_m']<.02 and x['fraction_over_0_1m']<.01 for x in results.values())
    atomic_json(a.output,{'first':a.first,'second':a.second,'ego_poses_match':poses_match,'same_rig':same_rig,'same_target':same_target,
        'calibration_ids':[first.manifest['calibration_id'],second.manifest['calibration_id']],'per_camera':results,'passed':passed,
        'scope':'Same frozen geometry, equal resolution, independent captures; occasional moving foliage/particle effects can remain.'})
    print('Depth unchanged within tolerance:',passed)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
