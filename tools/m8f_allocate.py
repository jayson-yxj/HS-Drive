"""Two fixed development sites and lane-derived requests, before any new RGB."""
import json
from pathlib import Path
import numpy as np
from tools.m8f_common import *
from simulator.m8f_placement import pose,box_record

def main():
 import carla
 c=carla.Client('127.0.0.1',2100);c.set_timeout(60);w=c.get_world()
 if w.get_map().name.split('/')[-1]!='Town02':w=c.load_world('Town02')
 mp=w.get_map();history=json.loads((ROOT/'historical_index.json').read_text());contract=json.loads(Path('configs/m8f/contract.json').read_text())
 src=Path('/root/gpufree-data/hs-drive-experiments/m8e-groundtruth/capture/m8e_development_00/evaluator/present_sidecar.json');old=json.loads(src.read_text())
 assets=[]
 for name,cat in [('vehicle.tesla.model3','vehicle'),('walker.pedestrian.0002','person'),('static.prop.constructioncone','cone'),('static.prop.streetbarrier','barrier')]:
  a=next(a for a in old['objects'] if a['name']==name)
  assets.append(dict(blueprint=name,category=cat,extent=a['extent_m'],local_box_matrix=(np.linalg.inv(a['entity_transform'])@a['T_world_from_box']).tolist(),dimension_source=asset(src)))
 candidates=[]
 for wp in mp.generate_waypoints(5):
  if wp.is_junction or str(wp.lane_type)!='Driving' or wp.lane_width<3.3:continue
  forward=wp.next(35);back=wp.previous(35)
  if not forward or not back or forward[0].road_id!=wp.road_id or back[0].road_id!=wp.road_id:continue
  side=wp.get_right_lane()
  for _ in range(3):
   if side is not None and str(side.lane_type)=='Sidewalk':break
   side=side.get_right_lane() if side else None
  if side is None or str(side.lane_type)!='Sidewalk':continue
  xy=np.array([wp.transform.location.x,wp.transform.location.y]);oldr=[r for r in history['records'] if r['map']=='Town02']
  clearance=min([float(np.linalg.norm(xy-r['center_xy']))-r['radius_m'] for r in oldr] or [1e6])
  candidates.append((clearance,wp))
 chosen=[]
 for clearance,wp in sorted(candidates,key=lambda x:x[0],reverse=True):
  if any(wp.transform.location.distance(x.transform.location)<150 for _,x in chosen):continue
  chosen.append((clearance,wp))
  if len(chosen)==2:break
 assert len(chosen)==2
 groups=[]
 for i,(clearance,ego) in enumerate(chosen):
  objects=[]
  for j,a in enumerate(assets):
   distance=[25,28,-24,-32][j]
   def at_distance(d):
    p=(ego.next(d) if d>=0 else ego.previous(-d))[0]
    if a['category']=='person':
     p=p.get_right_lane()
     while p and str(p.lane_type)!='Sidewalk':p=p.get_right_lane()
     if p is None:raise RuntimeError('No requested sidewalk')
    return p
   phases={}
   for phase,shift in [('present',0),('shifted',3)]:
    p=at_distance(distance+shift)
    offset=(p.lane_width/2-a['extent'][1]-.25) if a['category'] in ['cone','barrier'] else 0
    phases[phase]=dict(road_id=p.road_id,lane_id=p.lane_id,s=p.s,offset_m=offset,waypoint=pose(p.transform))
   objects.append(dict(key='target_%d'%j,asset=a,phases=phases))
  groups.append(dict(group='m8f_development_%02d'%i,split='development',map='Town02',weather='ClearNoon',ego=pose(ego.transform),objects=objects,nearest_history_envelope_clearance_m=clearance,independence_grade='HistoricalCoverageIncomplete',RepeatedLayoutFamily=True))
 write('configs/m8f/protocol.json',dict(**provenance('FrozenBeforeCARLARGBCapture'),groups=groups,contract=asset('configs/m8f/contract.json'),limits={'groups':8,'frames':32},current_authorized='DevelopmentSmokeOnly',later_collection='Requires both smoke scenes quality gate; otherwise NotRun',Final='NotAllocated_HistoricalCoverageIncomplete',raw_map_fresh=False))
 write('reports/m8f_split_preflight.json',dict(**provenance('ActualMapGeometryPreflight_NoRGB'),groups=groups,history=asset(ROOT/'historical_index.json'),final_status='HeldoutIndependenceNoGo',new_final_groups=0,development_use='Historical-support overlap permitted for debugging only; not independent natural training qualification',preselection_candidates=len(candidates)))
 print([(g['group'],g['ego']['xyz'],g['nearest_history_envelope_clearance_m']) for g in groups])
if __name__=='__main__':main()
