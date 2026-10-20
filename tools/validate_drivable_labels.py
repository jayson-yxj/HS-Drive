"""Cross-check raster labels against sampled visible optical-depth surface points.

This is an alignment diagnostic, not an independent semantic annotation benchmark.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
import yaml
from datasets.drivable import validate_arrays
from datasets.labels import sample_depth
from simulator.recorder import atomic_json,file_hash
from geometry.reference import transform_points


def validate(scene,label_root,output,bev_config='configs/bev.yaml',samples=5):
    import carla
    scene=Path(scene);root=Path(label_root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    label_manifest=json.loads((root/'manifest.json').read_text())
    if file_hash(scene/'manifest.json')!=label_manifest['raw_manifest_sha256']:raise ValueError('Scene mismatch')
    if file_hash(bev_config)!=label_manifest['bev_config_sha256']:raise ValueError('BEV mismatch')
    if file_hash(root/'map.xodr')!=label_manifest['opendrive_sha256']:raise ValueError('Map mismatch')
    cmap=carla.Map(label_manifest['map'],(root/'map.xodr').read_text())
    cfg=yaml.safe_load(Path(bev_config).read_text());cal=json.loads((scene/'calibration.json').read_text())
    frames=sorted((scene/'frames').glob('*.json'));stats={'road_surface_points':0,'valid_bev_comparisons':0,'driving_agreements':0}
    images=[]
    for count,i in enumerate(np.unique(np.linspace(0,len(frames)-1,min(samples,len(frames)),dtype=int))):
        path=frames[i];f=json.loads(path.read_text());ref=label_manifest['frames'][str(f['frame_id'])]
        if ref['raw_frame_sha256']!=file_hash(path) or ref['sha256']!=file_hash(root/ref['path']):raise ValueError('Frame mismatch')
        with np.load(root/ref['path']) as data:
            label=data['drivable_static'];valid=data['drivable_valid'];rules=data['drivable_rule_valid'];validate_arrays(label,valid,rules)
        c=cal['cameras']['front'];depth_path=scene/f['cameras']['front']['depth']
        with np.load(depth_path) as d:depth=d['depth_m']
        sampled=sample_depth(depth,output_wh=(c['width'],c['height']),stride=16)
        uv=sampled['depth_uv'];z=sampled['depth_m'];mask=sampled['depth_valid']
        q=np.concatenate((uv,np.ones((*uv.shape[:-1],1))),-1)
        opt=(q@np.linalg.inv(np.asarray(c['K'])).T)*z[...,None]
        ego=transform_points(opt.reshape(-1,3),np.asarray(c['T_hs_ego_from_opt']))
        carla_ego=ego.copy();carla_ego[:,1]*=-1
        world=transform_points(carla_ego,np.asarray(f['ego']['T_carla_world_from_ego']))
        rgb=cv2.imread(str(scene/f['cameras']['front']['rgb']))
        for j in np.flatnonzero(mask.ravel()):
            p=world[j];wp=cmap.get_waypoint(carla.Location(x=float(p[0]),y=float(p[1]),z=float(p[2])),
                                           project_to_road=False,lane_type=carla.LaneType.Driving)
            if wp is None or wp.is_junction:continue
            center=wp.transform.location;right=wp.transform.get_right_vector()
            lateral=abs((p[0]-center.x)*right.x+(p[1]-center.y)*right.y+(p[2]-center.z)*right.z)
            if abs(p[2]-center.z)>.25 or wp.lane_width/2-lateral<.6:continue
            stats['road_surface_points']+=1
            ix=int(np.floor((ego[j,0]-cfg['x_range_m'][0])/cfg['resolution_m']))
            iy=int(np.floor((ego[j,1]-cfg['y_range_m'][0])/cfg['resolution_m']))
            if not (0<=ix<label.shape[0] and 0<=iy<label.shape[1]) or not valid[ix,iy]:continue
            stats['valid_bev_comparisons']+=1;agree=label[ix,iy]==1;stats['driving_agreements']+=int(agree)
            u,v=np.rint(uv.reshape(-1,2)[j]).astype(int)
            cv2.circle(rgb,(u,v),3,(0,220,0) if agree else (0,0,255),-1)
        cv2.putText(rgb,'Visible ground -> map Driving -> BEV static lane (green match, red mismatch)',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        images.append(cv2.resize(rgb,(640,384)))
    stats['agreement']=stats['driving_agreements']/max(1,stats['valid_bev_comparisons'])
    stats['alignment_passed']=stats['valid_bev_comparisons']>=20 and stats['agreement']>=.98
    stats.update(schema_version='hsd-static-lane-alignment-v1',scene=str(scene),label_manifest_sha256=file_hash(root/'manifest.json'),
        scope='Front camera smooth road surface samples; shared OpenDRIVE source, not independent semantic ground truth; no claim on unknown/junction/traffic legality')
    atomic_json(output/'report.json',stats);cv2.imwrite(str(output/'front_alignment.png'),np.concatenate(images))
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=validate(a.scene,a.labels,a.output);print(json.dumps(r,indent=2))
    if not r['alignment_passed']:raise SystemExit(1)

if __name__=='__main__':main()
