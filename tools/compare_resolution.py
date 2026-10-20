"""Resolution evidence from the same angular target ROI, with no model claims."""
import argparse
from pathlib import Path
import cv2,numpy as np
from datasets.carla_dataset import CarlaDataset
from geometry.reference import transform_points,project_optical_points
from geometry.cameras import resize_rgb_and_intrinsics
from simulator.recorder import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];panels=[]
    for w in (640,960,1280):
        ds=CarlaDataset(Path(a.experiment)/f'training-{w}-v3');sample,images,depths,objects=ds.read(-1)
        camera=ds.calibration['cameras']['front'];image=images['front']
        target=next(o for o in objects if o['actor_id']==ds.manifest['target_actor_id'])
        points=transform_points(target['corners_hs_ego_m'],np.linalg.inv(camera['T_hs_ego_from_opt']))
        uv=project_optical_points(points,camera['K']);lo=np.floor(uv.min(0)-4).astype(int);hi=np.ceil(uv.max(0)+4).astype(int)
        lo=np.maximum(lo,0);hi=np.minimum(hi,[image.shape[1],image.shape[0]])
        crop=image[lo[1]:hi[1],lo[0]:hi[0],::-1]
        display=cv2.resize(crop,(400,300),interpolation=cv2.INTER_NEAREST)
        display=cv2.copyMakeBorder(display,35,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
        text=f'{w} native target: {uv[:,0].ptp() if hasattr(uv[:,0], "ptp") else np.ptp(uv[:,0]):.1f}px wide'
        cv2.putText(display,text,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(display)
        small,K=resize_rgb_and_intrinsics(image,camera['K'],(640,384))
        # Simple gradient diagnostic at a fixed angular ROI; not accuracy or MTF.
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)[145:235,270:370]
        rows.append({'capture_width':w,'target_bbox_width_px':float(np.ptp(uv[:,0])),
            'target_bbox_height_px':float(np.ptp(uv[:,1])),
            'downsampled_roi_laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
            'network_intrinsics':K.tolist()})
    cv2.imwrite(str(out/'native_target_comparison.png'),np.concatenate(panels,axis=1))
    atomic_json(out/'report.json',{'results':rows,'scope':'Same 25m target, equivalent FOV. Pixel size increases with resolution; no claim of training improvement. 1280 trial had synchronization failures and is diagnostic only.'})


if __name__=='__main__':main()
