"""Aggregate imaging trials without treating diagnostics as training accuracy."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from simulator.recorder import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--before', default='reports/imaging_noon_v2/geometry-640-v1.png')
    p.add_argument('--after', default='reports/imaging_noon_v4/training-960-v4.png')
    p.add_argument('--camera-montage-scene', help='Optional six-camera scene preview')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False);rows=[]
    for path in a.reports:
        report=json.loads(Path(path).read_text())
        for entry in report['results']:
            c=entry['per_camera'];b=entry['benchmark']
            rows.append({'weather':report['weather'],'profile':entry['profile_id'],
                'source_report':path,'integrity':entry['integrity']['recording_integrity_checks_passed'],
                'front_surface_black_pct':100*c['front']['depth_valid_surface']['near_black_fraction'],
                'front_surface_white_pct':100*c['front']['depth_valid_surface']['near_white_fraction'],
                'sixcam_max_surface_black_pct':100*max(v['depth_valid_surface']['near_black_fraction'] for v in c.values()),
                'sixcam_max_surface_white_pct':100*max(v['depth_valid_surface']['near_white_fraction'] for v in c.values()),
                'sixcam_max_static_rgb_mae':max(v['static_temporal_rgb_mae_8bit'] for v in c.values()),
                'complete_fps_wall':b['complete_frames_per_wall_s'],'sampled_device_gpu_mib':b['gpu_peak_sampled_mib'],
                'mib_per_frame':b['bytes_on_disk']/b['complete_frames']/1024**2})
    atomic_json(root/'metrics.json',{'rows':rows})
    lines=['| Weather | Profile | Integrity | Front black % | Front white % | Max temporal MAE | Wall FPS | GPU MiB | MiB/frame |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['weather']} | {r['profile']} | {r['integrity']} | {r['front_surface_black_pct']:.3f} | {r['front_surface_white_pct']:.3f} | {r['sixcam_max_static_rgb_mae']:.3f} | {r['complete_fps_wall']:.2f} | {r['sampled_device_gpu_mib']:.0f} | {r['mib_per_frame']:.2f} |")
    (root/'metrics.md').write_text('\n'.join(lines)+'\n')
    # User-facing compact comparison, fixed 640x384 display for all sources.
    choices=[('Original 640 / postprocess OFF',a.before),
             ('Training candidate 960 -> 640',a.after)]
    panels=[]
    for label,path in choices:
        im=cv2.imread(path)
        if im is None:raise FileNotFoundError(path)
        im=cv2.copyMakeBorder(im,36,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        cv2.putText(im,label,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(im)
    cv2.imwrite(str(root/'before_after.png'),np.concatenate(panels,axis=1))
    if a.camera_montage_scene:
        from datasets.carla_dataset import CarlaDataset
        from datasets.schema import CAMERA_IDS
        from geometry.cameras import resize_rgb_and_intrinsics
        ds=CarlaDataset(a.camera_montage_scene)
        _,images,_,_=ds.read(-1);panels=[]
        for name in CAMERA_IDS:
            small,_=resize_rgb_and_intrinsics(images[name],ds.calibration['cameras'][name]['K'],(640,384))
            panel=small[...,::-1].copy()
            cv2.putText(panel,name,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
            panels.append(panel)
        montage=np.concatenate([np.concatenate(panels[i:i+3],axis=1) for i in (0,3)],axis=0)
        cv2.imwrite(str(root/'six_cameras.png'),montage)


if __name__=='__main__':main()
