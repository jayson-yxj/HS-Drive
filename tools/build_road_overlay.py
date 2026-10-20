"""Derive junction/heading supervision for an immutable M3 index; no input changes."""
import argparse,json,shutil,time
from pathlib import Path
import cv2
import numpy as np
import yaml
from datasets.index import read_json,safe_path
from datasets.road_supervision import lane_ribbons,project_ribbons,SCHEMA
from simulator.recorder import atomic_json,file_hash
from tools.provenance import source_hashes


def main():
    import carla
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bev-config',default='configs/bev.yaml');a=p.parse_args()
    index_root=Path(a.index);ip=index_root/'index.json';index=read_json(ip)
    if file_hash(ip)!=read_json(index_root/'manifest.json')['index_sha256']:raise ValueError('Index checksum')
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    stage=out.with_name(out.name+'.building');stage.mkdir(parents=True,exist_ok=False)
    config=yaml.safe_load(Path(a.bev_config).read_text());raw=Path(index['raw_root']);maps={};records={};summaries={};start=time.monotonic()
    try:
        for scene in index['scenes']:
            sid=scene['scene_id'];source=raw/'labels_static_v1'/sid
            manifest=read_json(source/'manifest.json')
            if manifest['raw_manifest_sha256']!=scene['manifest_sha256'] or manifest['bev_config_sha256']!=file_hash(a.bev_config):raise ValueError('Label source mismatch')
            map_path=source/'map.xodr';sha=file_hash(map_path)
            if sha!=manifest['opendrive_sha256']:raise ValueError('Map checksum')
            if sha not in maps:maps[sha]=lane_ribbons(carla.Map(scene['map'],map_path.read_text()))
            totals={k:0 for k in ('positive','negative','unknown','junction','direction_valid','vertical_ambiguous')}
            cache_pose=None;cached=None;scene_rows=[r for r in index['samples'] if r['scene_id']==sid]
            for n,row in enumerate(scene_rows):
                ref=manifest['frames'][str(row['frame_id'])];fp=raw/scene['path']/'frames'/f"{row['frame_id']:08d}.json"
                if file_hash(fp)!=row['raw_frame_sha256'] or file_hash(fp)!=ref['raw_frame_sha256']:raise ValueError('Frame identity')
                f=read_json(fp);pose=np.asarray(f['ego']['T_carla_world_from_ego'])
                label_path=safe_path(source,ref['path'])
                if file_hash(label_path)!=ref['sha256']:raise ValueError('Static checksum')
                if cache_pose is None or not np.array_equal(pose,cache_pose):
                    with np.load(label_path,allow_pickle=False) as base:cached=project_ribbons(maps[sha],pose,config,base)
                    cache_pose=pose.copy()
                rel=row['sample_id']+'.npz';path=stage/rel;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**cached)
                records[row['sample_id']]={'path':rel,'sha256':file_hash(path),'raw_frame_sha256':row['raw_frame_sha256']}
                label=cached['drivable_static'];vals=[(label==1).sum(),(label==0).sum(),(label==255).sum(),cached['junction_support'].sum(),cached['lane_direction_valid'].sum(),cached['vertical_ambiguity'].sum()]
                for k,v in zip(totals,vals):totals[k]+=int(v)
                if n==len(scene_rows)//2:
                    color=np.full((*label.shape,3),60,np.uint8);color[label==0]=[30,30,210];color[label==1]=[40,180,40];color[cached['junction_support']]=[210,180,20]
                    cv2.imwrite(str(stage/sid/'preview.png'),cv2.resize(color[::-1,::-1],(480,640),interpolation=cv2.INTER_NEAREST))
            summaries[sid]={'frames':len(scene_rows),'counts':totals,'map_sha256':sha,'base_manifest_sha256':file_hash(source/'manifest.json')}
            print(sid,totals,flush=True)
        atomic_json(stage/'manifest.json',{'schema_version':SCHEMA,'status':'complete','index_sha256':file_hash(ip),
            'bev_config_sha256':file_hash(a.bev_config),'samples':records,'scenes':summaries,'wall_s':time.monotonic()-start,
            'source_sha256':source_hashes(),'semantics':'Driving lane ribbon union including junctions; directed headings masked on conflicts; no right-of-way or permitted route',
            'ribbon_spacing_m':1.,'lane_margin_m':.25,'height_tolerance_m':.75,'raw_root':str(raw)})
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage);raise

if __name__=='__main__':main()
