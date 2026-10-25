"""Development-only raw evidence around empty-tag controlled props."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m8e_common import *
from geometry.reference import unproject_pixels_to_optical,transform_points,OPTICAL_TO_CARLA_CAM
from datasets.m8e_reference import points_inside
s=json.loads((ROOT/'capture/m8e_development_00/evaluator/present_sidecar.json').read_text())
rows=[]
for cam in s['calibration']:
    files=s['sensors'][cam['name']]
    with np.load(files['instance']['path'],allow_pickle=False) as z:ids=z['instance'];tags=z['semantic']
    with np.load(files['depth']['path'],allow_pickle=False) as z:d=z['depth_m']
    vv,uu=np.where(np.isfinite(d)&(d>1.5)&(d<80))
    pts=transform_points(unproject_pixels_to_optical(uu,vv,d[vv,uu],np.asarray(cam['K'])),
                         np.asarray(cam['T_world_from_camera'])@OPTICAL_TO_CARLA_CAM)
    for obj in s['objects']:
        if obj['entity_id'] not in s['controlled_entity_ids']:continue
        inside=points_inside(pts,obj['T_world_from_box'],obj['extent_m'])
        hist=Counter(zip(tags[vv[inside],uu[inside]].tolist(),ids[vv[inside],uu[inside]].tolist()))
        rows.append(dict(camera=cam['name'],entity_id=obj['entity_id'],name=obj['name'],
                         pixels_inside=int(inside.sum()),top_semantic_render_counts=[(list(k),v) for k,v in hist.most_common(4)]))
write('reports/m8e_development_raw_probe.json',dict(**provenance('DevelopmentOnlyRawDepthOBBProbe'),rows=rows))
for r in rows:
    if r['pixels_inside']:print(r)
