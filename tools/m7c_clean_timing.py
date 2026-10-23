"""Clean core-only paired stream: no witness/CV/GT audit prewarming."""
from pathlib import Path
from dataclasses import replace,asdict
import time
import numpy as np
import carla
from shapely.ops import unary_union
from tools.m7c_common import *
from tools.m7c_equivalence import FILES,request_from
from tools.m7b_fixtures import map_fixture
from diagnostics.m7a_planning import generate,evaluate
from diagnostics.m7b_motion import ArcRoute,ObservedMotion,build_hypotheses,evaluate_dynamic_hypotheses
from diagnostics.m7c_geometry import GeometryCache,evaluate_cached


def main():
 cfg=read('configs/m7b/protocol.json');m=carla.Map('Town03',Path(cfg['planning_map']).read_text());cache=GeometryCache(sha(cfg['planning_map']));fixtures={};out=[]
 for file in FILES:
  for r in read('reports/m7b_'+file+'.json')['records']:
   if r['fixture']['status']!='FixtureValid':continue
   if r['road'] not in fixtures:fixtures[r['road']]=map_fixture(m,r['road'],'timing')
   f=fixtures[r['road']];req=request_from(r);obs=[ObservedMotion(o['object_id'],ArcRoute(o['route']),o['s0'],o['mode']) for o in r['observed_motion']];static={o.object_id for o in obs if o.mode=='static_geometry'};greq=replace(req,scene=replace(req.scene,boxes=tuple(b for b in req.scene.boxes if b.object_id in static)))
   now=100.201 if r['spec'].get('fault')=='expired' else 99.99 if r['spec'].get('fault')=='future' else 100.
   arms={};metrics={}
   for mode in (['baseline','cached'] if len(out)%2==0 else ['cached','baseline']):
    start=time.perf_counter();side=f['sides'];cs=generate(req,left_offset=side.get('left',{}).get('offset'),right_offset=side.get('right',{}).get('offset'),lane_permitted=r['spec']['kind']!='prohibited')
    if r['spec'].get('fault') in ['missing','worker_error']:cs=[]
    hs,u=build_hypotheses(req.scene,obs,cfg);rows=[]
    for c in cs:
     allowed={f['lane']};s='left' if c.name=='B_left' else 'right' if c.name=='C_right' else None
     if s in side:
      permitted=side[s]['permitted'] and r['spec']['kind']!='prohibited';c=replace(c,lane_change_permitted=permitted)
      if permitted:allowed.add(side[s]['lane'])
     def exact(x,y):
      w=m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
      return w is not None and w.road_id==r['road'] and w.lane_id in allowed
     if mode=='baseline':reg=unary_union([f['polygons'][k] for k in allowed]).buffer(-.03);g=evaluate(greq,c,reg,now,exact_road=exact)
     else:reg=cache.region(f['polygons'],allowed);g=evaluate_cached(greq,c,reg,now,cache=cache,exact_road=cache.exact([(r['road'],k) for k in allowed],f['z'],exact))
     d=evaluate_dynamic_hypotheses(c,req.scene,hs,u);rows.append(dict(name=c.name,geometry=asdict(g),dynamic=d,accepted=g.planning_feasible and d['robust']))
    selected=next((x['name'] for x in rows if x['accepted']),None);arms[mode]=(time.perf_counter()-start)*1000;metrics[mode]=dict(rows=rows,selected=selected)
   eq=equal(metrics['baseline'],metrics['cached']) and metrics['cached']['selected']==r['decisions']['B_hypotheses'];out.append(dict(scene_id=r['scene_id'],timing_ms=arms,equal=eq,selected=metrics['cached']['selected']));print(r['scene_id'],eq,arms,flush=True)
 stats=lambda v:{k:float(np.quantile(v,q)) for k,q in [('p50',.5),('p95',.95),('p99',.99),('max',1)]}|dict(over200ms=sum(t>200 for t in v),count=len(v))
 write('reports/m7c_clean_core_latency.json',dict(kind='ActualServerSerialCoreOnlyPairedStream',provenance=provenance(),records=out,all_equal=all(x['equal'] for x in out),summary={a:stats([x['timing_ms'][a] for x in out]) for a in ['baseline','cached']},cache=dict(hits=cache.hits,misses=cache.misses),scope='No witness/CV/independent-reference prewarming. Alternating arm order. Includes generation,hypotheses,all base candidates,union/buffer,selection. Excludes static map setup,deserialization,IPC and CARLA tick. Cache shared only among normal base candidates.',online='NotRun'))
if __name__=='__main__':main()
