"""Finite lane-center Development scan, with target footprints preflighted before selection."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from tools.m8g_common import *
from tools.m8g_history import preflight
from datasets.m8g_geometry import planned_box
from datasets.m8f_quality import polygon_distance
from simulator.m8f_placement import validate_request,pose

def main():
 import carla
 cfg=json.loads(Path('configs/m8g/protocol.json').read_text());old=json.loads(Path('configs/m8f/protocol.json').read_text());contract=json.loads(Path('configs/m8f/contract.json').read_text())
 # The lane-center work-zone scenario is explicit; same geometric/semantic thresholds.
 contract['class_regions']['cone']['scenario']='declared temporary lane-center work-zone closure'
 contract['class_regions']['barrier']['scenario']='declared temporary lane-center work-zone closure'
 write('configs/m8g/capture_contract.json',contract)
 history=json.loads((ROOT/'unified_history_v1.json').read_text());assert len(history['records'])==16562
 c=carla.Client('127.0.0.1',2100);c.set_timeout(40);w=c.get_world();mp=w.get_map();assert mp.name.endswith('Town02')
 inv=json.loads((ROOT/'live_inventory.json').read_text())['environment'];rows=[];groups=[]
 for g in old['groups']:
  ego=mp.get_waypoint(carla.Location(*g['ego']['xyz']),project_to_road=True,lane_type=carla.LaneType.Driving)
  for spec in g['objects']:
   cat=spec['asset']['category']
   for distance in cfg['scan_signed_distances_m']:
    phases={};requests={};hist={};reject=[]
    for phase,shift in [('present',0),('shifted',3)]:
     d=distance+shift;points=ego.next(d) if d>=0 else ego.previous(-d)
     if len(points)!=1 or points[0].road_id!=ego.road_id or points[0].is_junction:
      reject.append('OutsideFrozenRoadNeighborhood');continue
     wp=points[0]
     if cat=='person':
      wp=wp.get_right_lane()
      for _ in range(4):
       if wp is None or str(wp.lane_type)=='Sidewalk':break
       wp=wp.get_right_lane()
     if wp is None or str(wp.lane_type) not in contract['class_regions'][cat]['lanes']:
      reject.append('NoClassCompatibleLane');continue
     where=dict(road_id=wp.road_id,lane_id=wp.lane_id,section_id=wp.section_id,s=wp.s,offset_m=0,waypoint=pose(wp.transform));phases[phase]=where
     result=validate_request(w,mp,wp,spec['asset'],contract,inv);requests[phase]=result
     box=planned_box(result)
     if box:
      poly=np.array(box['corners_world_m'])[[0,4,6,2],:2];hist[phase]=dict(footprint=poly.tolist(),hypothetical_box=box,preflight=preflight(poly,'Town02',history,g['ego']['xyz'][:2]))
     else:reject.append('NoCompleteSurfaceForActualFootprint')
     if result['status']!='GroundedByGeometryProxy':reject.extend(result['reasons'])
    key=g['group'].replace('m8f','m8g')+'_'+cat+'_'+str(distance)
    row=dict(key=key,group=g['group'].replace('m8f','m8g'),category=cat,distance_m=distance,asset=spec['asset'],phases=phases,requests=requests,historical_preflight=hist,reasons=sorted(set(reject)),geometry_pass=not reject and len(requests)==2,independent_training_eligible=False,history_evaluated_before_acceptance=True)
    rows.append(row);print(key,row['geometry_pass'],row['reasons'],flush=True)
  # Frozen deterministic selection; both phases must be collision-free amongst subjects.
  selected=[];selection_rejections=[]
  for cat in ['vehicle','person','cone','barrier']:
   for r in [x for x in rows if x['group']==g['group'].replace('m8f','m8g') and x['category']==cat and x['geometry_pass']]:
    conflicts=[]
    for phase in ['present','shifted']:
     poly=r['historical_preflight'][phase]['footprint']
     for s in selected:
      if polygon_distance(poly,s['historical_preflight'][phase]['footprint'])<=.1:conflicts.append(s['key'])
    if conflicts:selection_rejections.append(dict(key=r['key'],reason='SimultaneousSubjectEnvelopeConflict',others=conflicts));continue
    selected.append(r);break
  groups.append(dict(group=g['group'].replace('m8f','m8g'),split='development',map='Town02',weather=g['weather'],ego=g['ego'],objects=[dict(key=r['category'],asset=r['asset'],phases=r['phases'],scan_key=r['key'],historical_preflight=r['historical_preflight']) for r in selected],core_geometry_pass={'vehicle','person'}<=set(r['category'] for r in selected),selection_rejections=selection_rejections,independence_grade='HistoricalCoverageIncomplete',RepeatedLayoutFamily=True))
 assert len(rows)<=80 and all(n<=20 for n in Counter(x['category'] for x in rows).values())
 write(ROOT/'scan_all_candidates.json',dict(**provenance('ActualCARLABoundedGeometry_NoRGB_NoActors'),records=rows,world_episode=w.id,frame=w.get_snapshot().frame,history=asset(ROOT/'unified_history_v1.json'),protocol=asset('configs/m8g/protocol.json')))
 write('reports/m8g_feasibility_scan.json',dict(**provenance('ActualCARLABoundedGeometry_NoRGB_NoActors'),raw=asset(ROOT/'scan_all_candidates.json'),attempts=len(rows),counts={cat:dict(attempted=sum(r['category']==cat for r in rows),passed=sum(r['category']==cat and r['geometry_pass'] for r in rows)) for cat in ['vehicle','person','cone','barrier']},reasons=dict(Counter(k for r in rows for k in r['reasons'])),groups=groups,pre_smoke_core_geometry_pass=all(g['core_geometry_pass'] for g in groups),no_threshold_change=True))
 write('reports/m8g_preflight_footprint_audit.json',dict(**provenance('BeforeSelection_PerTargetFootprint'),history=asset(ROOT/'unified_history_v1.json'),records=[dict(key=r['key'],geometry_pass=r['geometry_pass'],phases=r['historical_preflight']) for r in rows],coverage='HistoricalCoverageIncomplete',layout_family='RepeatedDevelopmentLaneCenterPlacement',new_Final=0,ego_center_only_selection=False))
 write('configs/m8g/smoke_protocol.json',dict(**provenance('FrozenAfterGeometryBeforeRGB'),groups=groups,geometry_gate=all(g['core_geometry_pass'] for g in groups),max_frames=8,max_attempts=1,contract=asset('configs/m8g/capture_contract.json'),no_model_outputs=True))
 print('core geometry',[(g['group'],g['core_geometry_pass']) for g in groups],flush=True)
if __name__=='__main__':main()
