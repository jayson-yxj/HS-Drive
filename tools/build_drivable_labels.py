"""Build immutable static lane supervision from pinned CARLA OpenDRIVE, offline."""
import argparse,json,shutil,time
from pathlib import Path
import cv2
import numpy as np
import yaml
from datasets.drivable import rasterize,SCHEMA
from simulator.recorder import atomic_json,file_hash
from tools.provenance import source_hashes


def build(scene,output,opendrive_dir,bev_config):
    import carla
    scene=Path(scene);output=Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((scene/'manifest.json').read_text())
    if manifest['status']!='complete' or manifest['carla_version']!='0.9.15':raise ValueError('Need complete CARLA 0.9.15 recording')
    name=manifest['map'].split('/')[-1]
    source=scene/'map.xodr'
    if not source.exists():source=Path(opendrive_dir)/(name+'.xodr')
    if (scene/'map_provenance.json').exists():
        provenance=json.loads((scene/'map_provenance.json').read_text())
        if provenance['opendrive_sha256']!=file_hash(source):raise ValueError('Recorded map hash mismatch')
    config=yaml.safe_load(Path(bev_config).read_text());cmap=carla.Map(name,source.read_text())
    stage=output.with_name(output.name+'.building');stage.mkdir(parents=True,exist_ok=False)
    rows={};totals=np.zeros(3,dtype=np.int64);start=time.monotonic();last_pose=None;last_arrays=None;cache_hits=0
    try:
        shutil.copy2(source,stage/'map.xodr')
        files=sorted((scene/'frames').glob('*.json'))
        if len(files)!=manifest['frames_committed']:raise ValueError('Raw frame count mismatch')
        for i,path in enumerate(files):
            frame=json.loads(path.read_text())
            pose=np.asarray(frame['ego']['T_carla_world_from_ego'])
            if last_pose is not None and np.array_equal(pose,last_pose):
                arrays=last_arrays;cache_hits+=1
            else:
                arrays=rasterize(cmap,pose,config);last_pose=pose;last_arrays=arrays
            out=stage/(path.stem+'.npz');np.savez_compressed(out,**arrays)
            rows[str(frame['frame_id'])]={'path':out.name,'sha256':file_hash(out),
                'raw_frame_sha256':file_hash(path)}
            label=arrays['drivable_static'];totals+=np.array([(label==1).sum(),(label==0).sum(),(label==255).sum()])
            if i==len(files)//2:
                color=np.zeros((*label.shape,3),np.uint8);color[label==255]=[60,60,60]
                color[label==1]=[50,190,50];color[label==0]=[40,40,220]
                cv2.imwrite(str(stage/'preview.png'),cv2.resize(color[::-1,::-1],(480,640),interpolation=cv2.INTER_NEAREST))
            if i%40==0:print(f'{name} {i+1}/{len(files)}',flush=True)
        atomic_json(stage/'manifest.json',{'schema_version':SCHEMA,'status':'complete','frames':rows,
            'raw_manifest_sha256':file_hash(scene/'manifest.json'),'map':name,
            'opendrive_sha256':file_hash(source),'map_source':'recorded export' if source.parent==scene else 'pinned CARLA package; not contemporaneous export',
            'bev_config_sha256':file_hash(bev_config),'source_sha256':source_hashes(),
            'class_counts':dict(zip(('positive','negative','unknown'),totals.tolist())),
            'identical_pose_cache_hits':cache_hits,'boundary_margin_m':.5,'height_tolerance_m':.75,'wall_s':time.monotonic()-start,
            'semantics':'static Driving lane support only; no direction/route/light legality; independent of obstacles and visibility',
            'unknown_policy':'junction, lane boundary, absent map, unsupported type or height mismatch -> 255; rule_valid always false'})
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bev-config',default='configs/bev.yaml')
    p.add_argument('--opendrive-dir',default='/root/gpufree-data/simulators/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive')
    a=p.parse_args();build(a.scene,a.output,a.opendrive_dir,a.bev_config)

if __name__=='__main__':main()
