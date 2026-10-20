"""Photometric diagnostics, same-FOV previews and geometry checks for a sweep."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from datasets.carla_dataset import CarlaDataset
from geometry.cameras import resize_rgb_and_intrinsics
from tools.validate_recording import validate
from simulator.recorder import atomic_json


def metrics(rgb,mask=None):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    values=gray[mask] if mask is not None else gray.ravel()
    if not len(values):raise ValueError('Empty evaluation mask')
    return {'mean_luma':float(values.mean()),'near_black_fraction':float(np.mean(values<=5)),
            'near_white_fraction':float(np.mean(values>=250)),
            'luma_p05':float(np.quantile(values,.05)),'luma_p95':float(np.quantile(values,.95))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.experiment);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    experiment=json.loads((root/'experiment.json').read_text());rows=[];panels=[];crops=[]
    for benchmark in experiment['results']:
        scene=Path(benchmark['scene']);ds=CarlaDataset(scene);integrity=validate(scene)
        per_camera={name:[] for name in ds.calibration['cameras']};previous={};differences={c:[] for c in per_camera}
        front=None
        for i in range(len(ds)):
            sample,images,depths,objects=ds.read(i)
            for name,rgb in images.items():
                cam=ds.calibration['cameras'][name];small,_=resize_rgb_and_intrinsics(rgb,cam['K'],(640,384))
                mask=cv2.resize(((depths[name]>=1.5)&(depths[name]<80)).astype(np.uint8),(640,384),interpolation=cv2.INTER_NEAREST).astype(bool)
                info={'all_pixels':metrics(small),'depth_valid_surface':metrics(small,mask)}
                per_camera[name].append(info)
                if name in previous:
                    differences[name].append(float(np.mean(np.abs(small.astype(float)-previous[name].astype(float)))))
                previous[name]=small
                if name=='front' and i==len(ds)-1:front=small
        stats={}
        for name,values in per_camera.items():
            stats[name]={domain:{key:float(np.mean([v[domain][key] for v in values])) for key in values[0][domain]} for domain in ('all_pixels','depth_valid_surface')}
            stats[name]['static_temporal_rgb_mae_8bit']=float(np.mean(differences[name]))
        row={'profile_id':ds.calibration['imaging_profile_id'],'calibration_id':ds.manifest['calibration_id'],
             'integrity':integrity,'per_camera':stats,
             'benchmark':{k:v for k,v in benchmark.items() if k!='gpu_samples'}}
        rows.append(row)
        panel=front[...,::-1].copy();cv2.rectangle(panel,(0,0),(640,32),(0,0,0),-1)
        cv2.putText(panel,ds.calibration['imaging_profile_id'],(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(panel)
        # Same angular ROI; enlarged for comparison, not additional information.
        crop=cv2.resize(front[145:235,270:370,::-1],(400,360),interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop,ds.calibration['imaging_profile_id'],(5,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);crops.append(crop)
        cv2.imwrite(str(out/(ds.calibration['imaging_profile_id']+'.png')),front[...,::-1])
    if panels:
        while len(panels)%2:panels.append(np.zeros_like(panels[0]));crops.append(np.zeros_like(crops[0]))
        cv2.imwrite(str(out/'front_comparison.png'),np.concatenate([np.concatenate(panels[i:i+2],axis=1) for i in range(0,len(panels),2)],axis=0))
        cv2.imwrite(str(out/'target_crops.png'),np.concatenate([np.concatenate(crops[i:i+2],axis=1) for i in range(0,len(crops),2)],axis=0))
    atomic_json(out/'report.json',{'experiment':str(root),'weather':experiment['weather'],'results':rows,
        'metrics_scope':'8-bit diagnostics at common 640x384, not true sensor saturation or trained-model accuracy. Depth mask excludes sky/80m+, temporal MAE assumes fixed scene.'})
    print(json.dumps([{'profile':r['profile_id'],'front':r['per_camera']['front'],'fps':r['benchmark']['complete_frames_per_wall_s'],'gpu_mib':r['benchmark']['gpu_peak_sampled_mib']} for r in rows],indent=2))


if __name__=='__main__':main()
