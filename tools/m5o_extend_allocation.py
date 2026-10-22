"""Append map capacity without changing geographic buffers or inspecting RGB quality."""
import json,math
from pathlib import Path
import numpy as np,carla
from tools.m5o_allocate import layouts
from tools.m5o_common import write,provenance

def main():
    base=json.loads(Path('configs/m5o_capture_dev.json').read_text());groups=base['groups']
    c=carla.Client('localhost',2100);c.set_timeout(120);world=c.load_world('Town04');mp=world.get_map()
    old=[]
    for p in Path('configs').glob('*.json'):
        try:d=json.loads(p.read_text())
        except ValueError:continue
        if isinstance(d,dict):old.extend(g['pose'][:2] for g in d.get('groups',[]) if isinstance(g,dict) and g.get('map','').split('/')[-1]=='Town04' and 'pose' in g)
    selected=[]
    for i in range(4):
        best=None
        for wp in mp.generate_waypoints(8):
            t=wp.transform;xy=np.array([t.location.x,t.location.y]);prior=min([np.linalg.norm(xy-v) for v in old] or [1000]);dist=min([np.linalg.norm(xy-v) for v in selected] or [1000])
            cross=[g for g in groups if g['map']=='Town04' and g['split']!='calibration']
            if prior<90 or dist<90 or any(wp.road_id==g['road_id'] or np.linalg.norm(xy-np.array(g['pose'][:2]))<150 for g in cross):continue
            score=min(dist,300)+prior*.01
            if best is None or score>best[0]:best=(score,wp,xy,prior)
        if best is None:break
        _,wp,xy,prior=best;selected.append(xy);t=wp.transform;pitch=math.degrees(math.asin(math.sin(math.radians(t.rotation.pitch))))
        seed=52000+i;groups.append(dict(group=f'Town04_m5o_calibration_{i:02d}',split='calibration',map='Town04',pose=[t.location.x,t.location.y,t.location.z+.4,t.rotation.pitch,t.rotation.yaw,t.rotation.roll],road_id=wp.road_id,lane_id=wp.lane_id,road_s=wp.s,is_junction=wp.is_junction,min_prior_m=float(prior),actual_pitch_deg=pitch,geometry='slope' if abs(pitch)>1 else 'junction' if wp.is_junction else 'flat_road',weather=['ClearNoon','CloudyNoon','ClearSunset'][i%3],layout_seed=seed,configurations=layouts(seed)))
    write('configs/m5o_capture_dev_v2.json',dict(base,groups=groups,extension=provenance(),additional_calibration_map='Town04_disjoint_road_ids_150m_from_train',added_groups=len(selected)))
    assert len(selected)==4
    print('Allocated',len(groups),'groups')
if __name__=='__main__':main()
