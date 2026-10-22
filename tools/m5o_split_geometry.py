"""Actual actor-world support overlap across new split road neighborhoods."""
import json
from pathlib import Path
import numpy as np
from tools.m5o_common import DATA,write,provenance


def main():
    records=[];points={};maps={}
    for split in ['train','val','calibration','sealed']:
        root=DATA/'raw'/f'm5o_{split}_indexroot'
        for p in root.glob('*/manifest.json'):
            m=json.loads(p.read_text());g=m['group'];maps[g]=(split,m['map']);pp=[]
            for fid in m['frames']:
                f=json.loads((p.parent/'frames'/f'{fid:08d}.json').read_text())
                for a in f['actors']:
                    if a['is_ego'] or np.linalg.norm(np.mean(a['corners_hs_ego_m'],axis=0)[:2])>80:continue
                    pp.append(np.array(a['world_matrix'])[:2,3])
            points[g]=np.array(pp)
    violations=[]
    for i,a in enumerate(sorted(points)):
        for b in sorted(points)[i+1:]:
            if maps[a][0]==maps[b][0] or maps[a][1]!=maps[b][1] or not len(points[a]) or not len(points[b]):continue
            distance=float(np.linalg.norm(points[a][:,None]-points[b][None],axis=-1).min());records.append(dict(a=a,b=b,min_target_world_xy_m=distance))
            if distance<45:violations.append([a,b,distance])
    write('reports/m5o_actual_split_geometry.json',dict(kind='offline_actual_CARLA_actor_placement_metadata',records=records,violations=violations,passed=not violations,provenance=provenance(),limits='Cross-split actual target center separation complements 150m Ego+road_id isolation. Does not establish independent backgrounds/weather/renderer distributions.'))
    assert not violations,violations
if __name__=='__main__':main()
