"""Conditional E_slow admission uses development only; no final tuning."""
from dataclasses import replace,asdict
from pathlib import Path
import time,carla
from tools.m7c_common import *
from tools.m7c_equivalence import request_from
from tools.m7b_fixtures import map_fixture
from tools.m7b_experiment import reference_audit
from diagnostics.m7b_motion import ArcRoute,ObservedMotion,build_hypotheses,evaluate_dynamic_hypotheses
from diagnostics.m7c_geometry import GeometryCache,evaluate_cached
from diagnostics.m7c_slow import slow_candidate,stopping_reserve


def main():
 cfg=read('configs/m7b/protocol.json');m=carla.Map('Town03',Path(cfg['planning_map']).read_text());cache=GeometryCache(sha(cfg['planning_map']));data=read('reports/m7b_development_v2.json');fixtures={};out=[]
 for r in data['records']:
  if r['fixture']['status']!='FixtureValid' or r['spec'].get('fault'):continue
  if r['road'] not in fixtures:fixtures[r['road']]=map_fixture(m,r['road'],'development')
  f=fixtures[r['road']];req=request_from(r);c=slow_candidate(req);obs=[ObservedMotion(o['object_id'],ArcRoute(o['route']),o['s0'],o['mode']) for o in r['observed_motion']];hs,u=build_hypotheses(req.scene,obs,cfg)
  allowed={f['lane']};reg=cache.region(f['polygons'],allowed)
  def exact(x,y):
   w=m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
   return w is not None and w.road_id==r['road'] and w.lane_id in allowed
  static={o.object_id for o in obs if o.mode=='static_geometry'};greq=replace(req,scene=replace(req.scene,boxes=tuple(b for b in req.scene.boxes if b.object_id in static)))
  start=time.perf_counter();geom=evaluate_cached(greq,c,reg,100.,cache=cache,exact_road=cache.exact([(r['road'],f['lane'])],f['z'],exact));dyn=evaluate_dynamic_hypotheses(c,req.scene,hs,u);reserve=stopping_reserve(c,req.scene,hs,ArcRoute(req.route));elapsed=(time.perf_counter()-start)*1000
  accepted=geom.planning_feasible and dyn['robust'] and reserve['passed'];ref=reference_audit(c,req.scene,r['evaluator_only_references']);old=r['decisions']['B_hypotheses'];gain=accepted and old in [None,'D_stop']
  out.append(dict(scene_id=r['scene_id'],candidate=asdict(c),geometry=asdict(geom),dynamic=dyn,reserve=reserve,accepted=accepted,reference=ref,old_selected=old,continue_gain=gain,elapsed_ms=elapsed));print(r['scene_id'],accepted,gain,flush=True)
 gains=[r['scene_id'] for r in out if r['continue_gain']];write('reports/m7c_slow_follow_feasibility.json',dict(kind='OfflineDevelopmentConditionalWitness',provenance=provenance(),records=out,gain_scenes=gains,enabled=bool(gains),family=dict(target_speed=.5,duration=4.5),criteria='Unchanged road/kinematics/motion hypotheses plus extra stop reserve; selected ahead of D_stop only',C_stop='NotRun',reference_conflicts=sum(bool(r['reference']['collisions']) for r in out if r['accepted'])))
 print('conditional admission gains',gains)
if __name__=='__main__':main()
