"""Build immutable M4.5 observed occupancy from legacy or m45 semantic records."""
import argparse,json,time
from pathlib import Path
import numpy as np
import yaml
from geometry.ray_tracing import VoxelGrid
from datasets.occupancy_labels import sampled_rays,build_observed,SCHEMA
from datasets.carla_dataset import CarlaDataset
from simulator.recorder import atomic_json,file_hash

# CARLA semantic labels are resolved from runtime CityObjectLabel names in new recordings.


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--stride',type=int,default=16);p.add_argument('--frames',type=int,default=3);a=p.parse_args()
    source=Path(a.scene);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);cfg=yaml.safe_load(Path('configs/bev.yaml').read_text());g=VoxelGrid(tuple(cfg['x_range_m']),tuple(cfg['y_range_m']),tuple(cfg['z_edges_m']),cfg['resolution_m'])
    custom=(source/'m45_manifest.json').exists();manifest=json.loads((source/('m45_manifest.json' if custom else 'manifest.json')).read_text());cal=json.loads((source/'calibration.json').read_text());records=[];t0=time.monotonic()
    files=sorted((source/'frames').glob('*.json'));chosen=np.unique(np.linspace(0,len(files)-1,min(a.frames,len(files)),dtype=int))
    for idx in chosen:
        f=json.loads(files[idx].read_text());orig=[];ends=[];sem=[];raystats={};ego=None
        if custom:
            corners=np.array(f['ego_bbox_corners_hs']);semantic_map={int(k):v for k,v in manifest['semantic_groups'].items()}
        else:
            objects=json.loads((source/f['labels']['objects']).read_text());entry=next(x for x in objects if x['is_ego']);corners=np.array(entry['corners_hs_ego_m']);semantic_map={}
        ego=(corners.min(0)-.05,corners.max(0)+.05)
        for name,ref in f['cameras'].items():
            path=source/ref['depth'];expected=(f['sha256'].get(ref['depth']) if 'sha256' in f else ref.get('depth_sha256'))
            if expected and file_hash(path)!=expected:raise ValueError('Depth hash mismatch')
            with np.load(path,allow_pickle=False) as d:depth=d['depth_m']
            semantic=None
            if custom:
                sp=source/ref['semantic'];assert file_hash(sp)==ref['semantic_sha256']
                with np.load(sp,allow_pickle=False) as d:semantic=d['semantic']
            c=cal['cameras'][name];o,e,s,stats=sampled_rays(depth,np.array(c['K']),np.array(c['T_hs_ego_from_opt']),stride=a.stride,semantic=semantic,semantic_map=semantic_map)
            orig.append(o);ends.append(e);sem.append(s);raystats[name]=stats
        labels,stats=build_observed(np.concatenate(orig),np.concatenate(ends),g,semantics=np.concatenate(sem),ego_box=ego)
        stem=f"{f['frame_id']:08d}";np.savez_compressed(out/(stem+'.npz'),**labels)
        # Separate visible surface support and ground/obstacle facts, never road permission.
        rec=dict(frame_id=f['frame_id'],timestamp_s=f['timestamp_s'],source_frame_sha256=file_hash(files[idx]),path=stem+'.npz',sha256=file_hash(out/(stem+'.npz')),stats=stats,rays=raystats,
            surface_count_positive=int((labels['surface_count']>0).sum()),ground_voxels=int((labels['surface_semantic']==1).sum()),obstacle_voxels=int(labels['obstacle_mask'].sum()),coverage=float(labels['occupancy_valid'].mean()))
        records.append(rec);print(stem,stats,flush=True)
    atomic_json(out/'manifest.json',dict(schema_version=SCHEMA,status='complete',source=str(source),source_manifest_sha256=file_hash(source/('m45_manifest.json' if custom else 'manifest.json')),bev_config_sha256=file_hash('configs/bev.yaml'),records=records,stride=a.stride,depth_convention='optical_z',surface_margin_m=.1,ego_margin_m=.05,semantic_available=custom,wall_s=time.monotonic()-t0,
        states={'UNKNOWN':0,'FREE':1,'SURFACE_OCCUPIED':2},semantics={'UNKNOWN':0,'GROUND':1,'STATIC_OBSTACLE':2,'DYNAMIC_OBSTACLE':3,'OTHER':4},limitations=['observed surfaces only; unseen interior unknown','coarse height bins combine road and above-ground features','dynamic semantics means potentially movable class, not measured motion','no traffic permission','sparse rays never imply full free space']))

if __name__=='__main__':main()
