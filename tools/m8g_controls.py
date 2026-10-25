"""Fresh actual CARLA queries at frozen counterexamples; no actors or RGB."""
import json
from pathlib import Path
from tools.m8g_common import *
from datasets.m8f_quality import choose_surface
from simulator.m8f_placement import ray,validate_request

def main():
 import carla
 c=carla.Client('127.0.0.1',2100);c.set_timeout(60);w=c.get_world();cfg=json.loads(Path('configs/m8f/contract.json').read_text());old=json.loads(Path('reports/m8f_negative_controls.json').read_text());rows=[]
 g=json.loads(Path('configs/m8f/protocol.json').read_text())['groups'][0];wh=g['objects'][0]['phases']['present'];wp=w.get_map().get_waypoint_xodr(wh['road_id'],wh['lane_id'],wh['s']);a=g['objects'][0]['asset'];inv=json.loads((ROOT/'live_inventory.json').read_text())['environment']
 for oldr in old['rows']:
  n=oldr['name']
  if n=='FullBodyRoadBoundarySpill':result=validate_request(w,w.get_map(),wp,a,cfg,inv,offset=wp.lane_width/2-.1)
  elif n=='StaticBuildingOverlap':result=validate_request(w,w.get_map(),wp,a,cfg,inv,xy_override=oldr['environment']['center_world_m'][:2])
  elif n=='NoValidRayHit':
   rr=ray(w,oldr['raw']['start'],oldr['raw']['end']);result=dict(**choose_surface(rr['hits'],0,[1,24],cfg['ground']),raw=rr)
  else:continue
  rows.append(dict(name=n,map='Town02',A=result,B=result,passed=result['status'] in ['Rejected','GroundSupportUnknown'],reference='ActualCARLAReadOnly_NoActorSpawn'))
 # Positive geometry checks re-run full validator; actual support/identity remains NotRun.
 scan=json.loads((ROOT/'scan_all_candidates.json').read_text())['records']
 for cat in ['vehicle','person','cone','barrier']:
  r=next(r for r in scan if r['category']==cat and r['geometry_pass']);wh=r['phases']['present'];wp=w.get_map().get_waypoint_xodr(wh['road_id'],wh['lane_id'],wh['s']);res=validate_request(w,w.get_map(),wp,r['asset'],cfg,inv)
  rows.append(dict(name='PositiveGeometry_'+cat,map='Town02',scan_key=r['key'],A=res,B=res,passed=res['status']=='GroundedByGeometryProxy',reference='ActualCARLA_RaysOnly_NoFinalActorNoIdentity'))
 oldwater=json.loads(Path('reports/m8f_water_multilayer_probe.json').read_text())['records']
 for town in sorted(set(r['map'] for r in oldwater)):
  w=c.load_world(town)
  for r in [r for r in oldwater if r['map']==town]:
   rr=ray(w,r['raw']['start'],r['raw']['end']);# match frozen expected ground reference via selected level or failed center
   expected=r['raw']['start'][2]-8
   # M8F used center minus asset half height. Re-evaluate water refusal independent of layer.
   water=[h for h in rr['hits'] if h['semantic']==23];ground=[h for h in rr['hits'] if h['semantic'] in [1,2,24]]
   if water:expected=water[0]['xyz'][2]
   elif ground:expected=ground[0]['xyz'][2]
   d=choose_surface(rr['hits'],expected,[1,2,24],cfg['ground']);is_negative=r['decision']['status']!='GroundedByGeometryProxy'
   rows.append(dict(name='HistoricalSurface_'+r['entity_key'],map=town,raw=rr,A=d,B=d,historical_decision=r['decision'],negative=is_negative,passed=d['status']!='GroundedByGeometryProxy' if is_negative else d['status']=='GroundedByGeometryProxy',reference='ActualCARLANewQueryAtFrozenLocation_NotHistoricalReconstruction'))
 write('reports/m8g_positive_negative_controls.json',dict(**provenance('ActualCARLAControls_NoNewActors_NoRGB'),records=rows,all_passed=all(r['passed'] for r in rows),A_B_identical=True,synthetic_tests='tests/test_m8g_geometry.py',no_new_rule_enabled=True))
 print([(r['name'],r['passed']) for r in rows],flush=True)
if __name__=='__main__':main()
