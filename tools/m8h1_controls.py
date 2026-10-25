"""Preserve original negative semantics; actual rays, offline replay, synthetic tests separated."""
import json,copy
from pathlib import Path
from tools.m8h1_common import *
from tools.m8h1_preflight import tf
from simulator.m8f_placement import validate_request,ray
from datasets.m8f_quality import choose_surface

def main():
 import carla
 c=carla.Client('127.0.0.1',2100);c.set_timeout(60);w=c.get_world();assert w.get_map().name.endswith('Town02');cfg=json.loads(Path('configs/m8f/contract.json').read_text());env=json.loads((ROOT/'environment_inventory.json').read_text())['objects'];old=json.loads(Path('reports/m8f_negative_controls.json').read_text());g=json.loads(Path('configs/m8f/protocol.json').read_text())['groups'][0];wh=g['objects'][0]['phases']['present'];wp=w.get_map().get_waypoint_xodr(wh['road_id'],wh['lane_id'],wh['s']);a=g['objects'][0]['asset'];rows=[]
 for r in old['rows']:
  if r['name']=='FullBodyRoadBoundarySpill':d=validate_request(w,w.get_map(),wp,a,cfg,env,offset=wp.lane_width/2-.1)
  elif r['name']=='StaticBuildingOverlap':d=validate_request(w,w.get_map(),wp,a,cfg,env,xy_override=r['environment']['center_world_m'][:2])
  elif r['name']=='NoValidRayHit':
   rr=ray(w,r['raw']['start'],r['raw']['end']);d=dict(**choose_surface(rr['hits'],0,[1,24],cfg['ground']),raw=rr)
  else:continue
  rows.append(dict(name=r['name'],kind='ActualCARLAQuery_NoActorSpawn',map='Town02',result=d,passed=d['status'] in ['Rejected','GroundSupportUnknown']))
 # Archived unexplained NONE/vegetation/static conflicts remain rejected by unchanged code.
 archived=json.loads(Path('reports/m8f_ray_hit_and_contact_audit.json').read_text())['requests']
 for group,key in [('m8f_development_00','target_0'),('m8f_development_00','target_1'),('m8f_development_00','target_2')]:
  r=next(x for x in archived if x['group']==group and x['key']==key);rr=r['preflight']['present'];decisions=[choose_surface(x['hits'],rr['waypoint']['xyz'][2],rr['declared_region']['semantics'],cfg['ground']) for x in rr['rays']]
  if key!='target_2':passed=any(d['status']!='GroundedByGeometryProxy' for d in decisions)
  else:passed=bool(rr['static_competitors']) and rr['status']=='Rejected'
  rows.append(dict(name='Frozen_'+key,kind='OfflineHistoricalEvidence_NotNewCARLA',source=asset('reports/m8f_ray_hit_and_contact_audit.json'),decisions=decisions,static_competitors=rr['static_competitors'],passed=passed))
 water=json.loads(Path('reports/m8f_water_multilayer_probe.json').read_text())['records']
 for town in sorted(set(r['map'] for r in water if r['decision']['status']!='GroundedByGeometryProxy')):
  w=c.load_world(town)
  for r in [r for r in water if r['map']==town and r['decision']['status']!='GroundedByGeometryProxy']:
   rr=ray(w,r['raw']['start'],r['raw']['end']);hits=[h for h in rr['hits'] if h['semantic']==23];expected=hits[0]['xyz'][2] if hits else r['raw']['start'][2]-8;d=choose_surface(rr['hits'],expected,[1,2,24],cfg['ground']);rows.append(dict(name='Water_'+r['entity_key'],kind='ActualCARLAQuery_NoActorSpawn',map=town,raw=rr,result=d,passed=d['status']!='GroundedByGeometryProxy'))
 # Multilayer is expressly synthetic, no claim of new multi-level capture.
 d=choose_surface([dict(xyz=[0,0,0],semantic=1),dict(xyz=[0,0,4],semantic=1)],0,[1],cfg['ground']);rows.append(dict(name='MultipleGroundLayers',kind='SyntheticGeometry',result=d,passed='MultipleGroundLayers' in d['reasons']))
 write('reports/m8h1_negative_controls.json',dict(**provenance('SeparatedActualOfflineSyntheticControls'),records=rows,all_pass=all(x['passed'] for x in rows),timestamp_controls='tests/test_m8e_source_time.py, actualM8H1sensor capture NotRun',unchanged_validator=[asset('simulator/m8f_placement.py'),asset('datasets/m8f_quality.py')]))
 print([(r['name'],r['passed']) for r in rows],flush=True)
if __name__=='__main__':main()
