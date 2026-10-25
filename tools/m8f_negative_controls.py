"""Actual ray/road/OBB rejection probes, no extra RGB or unsafe actor spawning."""
import json
from pathlib import Path
import numpy as np
from tools.m8f_common import *
from simulator.m8f_placement import ray,validate_request,box_record
from datasets.m8f_quality import choose_surface

def main():
 import carla
 c=carla.Client('127.0.0.1',2100);c.set_timeout(30);w=c.get_world();mp=w.get_map()
 assert not w.get_settings().synchronous_mode
 cfg=json.loads(Path('configs/m8f/contract.json').read_text());g=json.loads(Path('configs/m8f/protocol.json').read_text())['groups'][0]
 wh=g['objects'][0]['phases']['present'];wp=mp.get_waypoint_xodr(wh['road_id'],wh['lane_id'],wh['s']);asset_spec=g['objects'][0]['asset']
 env=[]
 for e in w.get_environment_objects(carla.CityObjectLabel.Any):env.append(dict(entity_id=int(e.id),name=e.name,semantic_tags=[int(e.type)],**box_record(e.bounding_box)))
 rows=[]
 # Full-body lane spill: center near edge, corners outside, even if center has a waypoint.
 r=validate_request(w,mp,wp,asset_spec,cfg,env,offset=wp.lane_width/2-.1)
 rows.append(dict(name='FullBodyRoadBoundarySpill',kind='ActualCARLARayLaneProbe',result=r,expected='Rejected',passed='FootprintOutsideAllowedRegion' in r['reasons']))
 rr=ray(w,[10000,10000,10],[10000,10000,-20]);d=choose_surface(rr['hits'],0,[1,24],cfg['ground'])
 rows.append(dict(name='NoValidRayHit',kind='ActualCARLARayProbe',raw=rr,result=d,expected='GroundSupportUnknown',passed=d['status']=='GroundSupportUnknown'))
 # Existing static facility center; no actor is created at the rejected location.
 center=np.array(g['ego']['xyz'][:2]);bld=[e for e in env if e['semantic_tags'][0]==3 and max(e['dimensions_lwh_m'])<40]
 e=min(bld,key=lambda e:np.linalg.norm(np.array(e['center_world_m'][:2])-center))
 r=validate_request(w,mp,wp,asset_spec,cfg,env,xy_override=e['center_world_m'][:2])
 rows.append(dict(name='StaticBuildingOverlap',kind='ActualCARLAEnvironmentOBB_RayLaneProbe',environment=e,result=r,expected='Rejected',passed=r['status']=='Rejected'))
 # Query actual water assets if present; preserve no-support result rather than inventing water hits.
 water=[e for e in env if e['semantic_tags'][0]==23]
 trials=[]
 for e in water[:8]:
  x,y,z=e['center_world_m'];rr=ray(w,[x,y,z+10],[x,y,z-20]);trials.append(dict(environment_id=e['entity_id'],raw=rr,decision=choose_surface(rr['hits'],z,[1],cfg['ground'])))
 rows.append(dict(name='WaterOrMultilayer',kind='ActualCARLAAssetSearch',water_asset_count=len(water),trials=trials,status='ActualWaterQueried' if trials else 'NotAvailableInSelectedTown_NoSyntheticClaim'))
 write('reports/m8f_negative_controls.json',dict(**provenance('ActualCARLARejectionProbes_NoRGB_NoNewActors'),map=mp.name,rows=rows,synthetic_water_multilayer_tests='tests/test_m8f_quality.py; explicitly synthetic',no_unsafe_spawn=True))
 print([(r['name'],r.get('passed',r.get('status'))) for r in rows])
if __name__=='__main__':main()
