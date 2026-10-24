"""All205/1662 paired original-vs-static-region reuse, no F equivalence claim."""
import time
from pathlib import Path
from dataclasses import asdict,replace
import carla
from shapely.ops import unary_union
from tools.m7c_equivalence import FILES,request_from,candidate_from
from tools.m7b_fixtures import map_fixture
from diagnostics.m7c_geometry import GeometryCache,evaluate_cached
from diagnostics.m7e_region import StaticRegionCache
from diagnostics.m7b_motion import ArcRoute,ObservedMotion,build_hypotheses,evaluate_dynamic_hypotheses
from tools.m7c_common import equal
from tools.m7e_common import *


def main():
    cfg=read('configs/m7b/protocol.json');mp=cfg['planning_map'];m=carla.Map('Town03',Path(mp).read_text());geomcache=GeometryCache(sha(mp));regions=StaticRegionCache();fixtures={};out=[]
    for filename in FILES:
        for r in read('reports/m7b_'+filename+'.json')['records']:
            if r['fixture']['status']!='FixtureValid':continue
            if r['road'] not in fixtures:fixtures[r['road']]=map_fixture(m,r['road'],'regression')
            f=fixtures[r['road']];req=request_from(r);now=100.201 if r['spec'].get('fault')=='expired' else 99.99 if r['spec'].get('fault')=='future' else 100.
            obs=[ObservedMotion(o['object_id'],ArcRoute(o['route']),o['s0'],o['mode']) for o in r['observed_motion']];hs,u=build_hypotheses(req.scene,obs,cfg)
            static={o.object_id for o in obs if o.mode=='static_geometry'};greq=replace(req,scene=replace(req.scene,boxes=tuple(b for b in req.scene.boxes if b.object_id in static)));es=[]
            for old in r['candidate_evaluations']:
                c=candidate_from(old['candidate']);allowed={f['lane']};side='left' if c.name=='B_left' else 'right' if c.name=='C_right' else None
                if side in f['sides'] and f['sides'][side]['permitted'] and r['spec']['kind']!='prohibited':allowed.add(f['sides'][side]['lane'])
                base=unary_union([f['polygons'][k] for k in allowed]).buffer(-.03)
                polygons={(r['road'],k,0):v for k,v in f['polygons'].items()};scope={(r['road'],k,0) for k in allowed}
                optimized=regions.region(sha(mp),polygons,scope,.03)
                def exact(x,y):
                    w=m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
                    return w is not None and w.road_id==r['road'] and w.lane_id in allowed
                ec=replace(c,name='D_stop') if c.name.startswith('E_stop') else c
                vals=[]
                for region in [base,optimized]:
                    g=evaluate_cached(greq,ec,region,now,cache=geomcache,exact_road=geomcache.exact(scope,f['z'],exact));dyn=evaluate_dynamic_hypotheses(c,req.scene,hs,u);vals.append((asdict(g),dyn))
                nominal=evaluate_cached(req,ec,optimized,now,cache=geomcache,exact_road=geomcache.exact(scope,f['z'],exact))
                match=equal(vals[0],vals[1]) and equal(vals[1][0],old['geometry']) and equal(vals[1][1],old['dynamic']) and equal(asdict(nominal),old['nominal'])
                es.append(dict(name=c.name,equivalent=match,region_equal=base.equals_exact(optimized,0),robust=vals[1][0]['planning_feasible'] and vals[1][1]['robust'],reasons=vals[1][0]['reasons'],dynamic=vals[1][1]))
            selected=next((e['name'] for e in es if not e['name'].startswith('E_stop') and e['robust']),None)
            match=all(e['equivalent'] and e['region_equal'] for e in es) and selected==r['decisions']['B_hypotheses'] and equal([asdict(h) for h in hs],r['hypotheses']) and equal(u,r['prediction_unknown'])
            out.append(dict(scene_id=r['scene_id'],equivalent=match,candidates=es,selected=selected));print(r['scene_id'],match,flush=True)
    write('reports/m7e_geometry_equivalence.json',dict(kind='ActualServerOfflineFrozenHistoricalPairedReplay',provenance=provenance(),records=out,scenes=len(out),candidates=sum(len(r['candidates']) for r in out),all_equal=all(r['equivalent'] for r in out),region_cache=dict(hits=regions.hits,misses=regions.misses),scope='A only. F changes candidate library and is evaluated separately.'))
if __name__=='__main__':main()
