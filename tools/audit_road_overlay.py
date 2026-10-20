"""Validate v2 labels at query cell centers, with the legacy mask held fixed."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml
from datasets.index import safe_path
from datasets.drivable import grid_centers
from datasets.road_supervision import validate_road_arrays
from simulator.recorder import file_hash,atomic_json


def main():
    import carla
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True);p.add_argument('--overlay',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();index=json.loads((Path(a.index)/'index.json').read_text());root=Path(a.overlay);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['index_sha256']!=file_hash(Path(a.index)/'index.json'):raise ValueError('Overlay/index identity')
    cfg=yaml.safe_load(Path('configs/bev.yaml').read_text());centers=grid_centers(cfg);centers[...,1]*=-1;reports={}
    for scene in index['scenes']:
        rows=[r for r in index['samples'] if r['scene_id']==scene['scene_id']];rows=[rows[i] for i in np.unique(np.linspace(0,len(rows)-1,min(3,len(rows)),dtype=int))]
        source=Path(index['raw_root'])/'labels_static_v1'/scene['scene_id'];cmap=carla.Map(scene['map'],(source/'map.xodr').read_text())
        totals={'legacy_valid_preserved':0,'legacy_valid_total':0,'added_positives':0,'added_centers_confirmed_driving':0,'added_centers_checked':0,'junction_cells':0}
        for row in rows:
            ref=manifest['samples'][row['sample_id']];path=safe_path(root,ref['path'])
            if file_hash(path)!=ref['sha256']:raise ValueError('Road checksum')
            with np.load(path,allow_pickle=False) as d:
                validate_road_arrays(d);label=d['drivable_static'];j=d['junction_support']
            frame=json.loads((Path(index['raw_root'])/scene['path']/'frames'/f"{row['frame_id']:08d}.json").read_text())
            with np.load(source/f"{row['frame_id']:08d}.npz") as b:old=b['drivable_static'];old_valid=b['drivable_valid']
            totals['legacy_valid_preserved']+=int(((label==old)&old_valid).sum());totals['legacy_valid_total']+=int(old_valid.sum())
            added=(label==1)&(old!=1);totals['added_positives']+=int(added.sum());totals['junction_cells']+=int(j.sum())
            world=centers@np.array(frame['ego']['T_carla_world_from_ego']).T
            ids=np.argwhere(added)
            for xy in ids[::max(1,len(ids)//250)]:
                x,y,z=world[tuple(xy)][:3];wp=cmap.get_waypoint(carla.Location(x=float(x),y=float(y),z=float(z)),project_to_road=False,lane_type=carla.LaneType.Driving)
                totals['added_centers_checked']+=1;totals['added_centers_confirmed_driving']+=int(wp is not None and abs(wp.transform.location.z-z)<=.75)
        totals['center_agreement']=totals['added_centers_confirmed_driving']/max(1,totals['added_centers_checked']);reports[scene['scene_id']]=totals
    checks={'all_sampled_added_centers_agree_95pct':all(s['center_agreement']>=.95 for s in reports.values()),
        'junctions_added':sum(s['junction_cells'] for s in reports.values())>0}
    atomic_json(a.output,{'schema_version':'hsd-road-overlay-audit-v1','checks':checks,'scenes':reports,'overlay_manifest_sha256':file_hash(root/'manifest.json'),
        'scope':'Map query vs connected-strip raster check; shared OpenDRIVE, not independent semantic annotations; legacy comparisons report actual mask changes'})
    print(json.dumps(checks,indent=2));print(json.dumps({k:round(v['center_agreement'],4) for k,v in reports.items()},indent=2))
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
