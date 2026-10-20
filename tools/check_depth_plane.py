"""Compare raw depth to an independently known plane; do not auto-certify M2."""
import argparse
import numpy as np
from datasets.carla_dataset import CarlaDataset
from geometry.oracle_bev import plane_depth_comparison
from simulator.recorder import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--camera',default='front')
    p.add_argument('--plane-normal-opt',type=float,nargs=3,required=True)
    p.add_argument('--plane-offset-m',type=float,required=True)
    p.add_argument('--mask',required=True,help='Boolean HxW .npy identifying the known plane')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    ds=CarlaDataset(a.scene)
    sample,_,depths,_=ds.read(a.index)
    d=depths[a.camera]
    mask=np.load(a.mask,allow_pickle=False)
    if mask.dtype!=np.bool_ or mask.shape!=d.shape:
        raise ValueError('Mask must be bool with the depth image shape')
    v,u=np.indices(d.shape)
    x=(u-(d.shape[1]-1)/2)/(d.shape[1]/2)
    y=(v-(d.shape[0]-1)/2)/(d.shape[0]/2)
    regions={'all':mask,'center':mask & (abs(x)<.25) & (abs(y)<.25),
             'edge':mask & ((abs(x)>.7)|(abs(y)>.7))}
    results={key:plane_depth_comparison(d,ds.calibration['cameras'][a.camera]['K'],
                  a.plane_normal_opt,a.plane_offset_m,region) for key,region in regions.items()}
    atomic_json(a.output,{'frame_id':sample['frame_id'],'camera':a.camera,
        'mae_m':results,'plane_normal_opt':a.plane_normal_opt,'plane_offset_m':a.plane_offset_m,
        'calibration_id':sample['calibration_id'],'gate_passed':False,
        'note':'Review independent plane pose, mask, central/edge residuals before confirming convention'})
    print(results)


if __name__=='__main__':
    main()
