"""New frozen scene campaign: cached baseline vs conditional slow-follow."""
import argparse,time
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import carla
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tools.m7c_common import *
from tools.m7b_fixtures import map_fixture,construct,scenarios
from tools.m7a_scenarios import lane_polygon,pose
from tools.m7b_experiment import reference_audit
from diagnostics.m7b_motion import ArcRoute,build_hypotheses,evaluate_dynamic_hypotheses
from diagnostics.m7a_planning import generate
from diagnostics.m7c_geometry import GeometryCache,evaluate_cached
from diagnostics.m7c_routes import extend_existing
from diagnostics.m7c_slow import slow_candidate,stopping_reserve
from simulator.oracle_driver import footprint


def connected_fixture(cmap,road,split):
    f=map_fixture(cmap,road,split);original=f['arc'];path,conn,join=extend_existing(cmap,f['route'],f['z']);f['arc']=path
    # Keep ego route intent unchanged; only uniquely justified actor support extends.
    # For original-lane actors, union forward corridor strips from true successors.
    full=[cmap.get_waypoint_xodr(n['road'],n['lane'],n['s']) for n in conn.nodes]
    full=[w for w in full if w is not None]
    f['polygons'][f['lane']]=unary_union([f['polygons'][f['lane']],lane_polygon(full)])
    f['continuation']=dict(original_length=original.length,extended_length=path.length,reason=conn.stop_reason,status=conn.status,transitions=conn.transitions,join_distance=join,nodes=conn.nodes)
    return f


def run(split,suffix):
    cfg=read('configs/m7b/protocol.json');protocol=read('configs/m7c/protocol.json');roads=protocol[split+'_roads'];m=carla.Map('Town03',Path(cfg['planning_map']).read_text());cache=GeometryCache(sha(cfg['planning_map']));records=[];road_records=[]
    for road in roads:
        start=time.perf_counter()
        try:f=connected_fixture(m,road,split)
        except ValueError as e:road_records.append(dict(road=road,status='FixtureInvalid',reason=str(e)));continue
        road_records.append(dict(road=road,status='MapReady',setup_ms=(time.perf_counter()-start)*1000,route=f['route'],continuation=f['continuation'],lane=f['lane'],sides=f['sides']))
        for i,spec in enumerate(scenarios()):
            begin=time.perf_counter();req,obs,refs,validity=construct(f,spec,i,cfg);req=replace(req,route=tuple(map(tuple,f['route'])),goal_xy_yaw=tuple(f['route'][-1]));rec=dict(scene_id=req.scene_id,road=road,spec=spec,fixture=validity,request=asdict(req),references_evaluator_only=refs)
            if validity['status']!='FixtureValid':records.append(rec);print(req.scene_id,'FixtureInvalid',flush=True);continue
            start=time.perf_counter();hs,u=build_hypotheses(req.scene,obs,cfg)
            # Full body support of enumerated futures. Topology is not itself a road certificate.
            support_fail=[]
            for h in hs:
                o=next(x for x in obs if x.object_id==h.object_id);b=next(x for x in req.scene.boxes if x.object_id==h.object_id)
                if o.mode=='crossing_corridor':continue
                lane=next(x['expected_lane'] for x in validity['placements'] if x['object_id']==h.object_id)
                if any(not f['polygons'][lane].covers(Polygon(footprint(*p[:2],p[2],b.half_length,b.half_width))) for p in h.poses):support_fail.append(dict(object_id=h.object_id,hypothesis=h.name,reason='FutureBodyOutsideSupportedRoad'))
            u+=support_fail;hypms=(time.perf_counter()-start)*1000
            start=time.perf_counter();side=f['sides'];candidates=generate(req,left_offset=side.get('left',{}).get('offset'),right_offset=side.get('right',{}).get('offset'),lane_permitted=spec['kind']!='prohibited');genms=(time.perf_counter()-start)*1000
            if spec.get('fault') in ['missing','worker_error']:candidates=[]
            now=100.201 if spec.get('fault')=='expired' else 99.99 if spec.get('fault')=='future' else 100.
            es=[];basecost=genms+hypms;slowcost=0;slowgen=0
            if not spec.get('fault') and read('reports/m7c_slow_follow_feasibility.json')['enabled']:
                start=time.perf_counter()
                try:candidates.append(slow_candidate(req))
                except ValueError:pass
                slowgen=(time.perf_counter()-start)*1000
            for c in candidates:
                allowed={f['lane']};s='left' if c.name=='B_left' else 'right' if c.name=='C_right' else None
                if s in side:
                    permit=side[s]['permitted'] and spec['kind']!='prohibited';c=replace(c,lane_change_permitted=permit)
                    if permit:allowed.add(side[s]['lane'])
                t0=time.perf_counter();region=cache.region(f['polygons'],allowed)
                def exact(x,y):
                    w=m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
                    return w is not None and w.road_id==road and w.lane_id in allowed
                static={o.object_id for o in obs if o.mode=='static_geometry'};greq=replace(req,scene=replace(req.scene,boxes=tuple(b for b in req.scene.boxes if b.object_id in static)))
                geom=evaluate_cached(greq,c,region,now,cache=cache,exact_road=cache.exact([(road,k) for k in allowed],f['z'],exact));dyn=evaluate_dynamic_hypotheses(c,req.scene,hs,u)
                reserve=stopping_reserve(c,req.scene,hs,ArcRoute(req.route)) if c.name=='E_slow_follow' else None
                elapsed=(time.perf_counter()-t0)*1000
                if c.name=='E_slow_follow':slowcost+=elapsed
                else:basecost+=elapsed
                accepted=geom.planning_feasible and dyn['robust'] and (reserve is None or reserve['passed']);ref=reference_audit(c,req.scene,refs)
                es.append(dict(name=c.name,candidate=asdict(c),geometry=asdict(geom),dynamic=dyn,stop_reserve=reserve,accepted=accepted,reference=ref,core_ms=elapsed))
            start=time.perf_counter();baseline=next((e['name'] for e in es if e['name']!='E_slow_follow' and e['accepted']),None)
            order=['A_keep_route','B_left','C_right','E_slow_follow','D_stop'];selected=next((name for name in order if any(e['name']==name and e['accepted'] for e in es)),None);selection=(time.perf_counter()-start)*1000
            rec.update(evaluations=es,baseline_selected=baseline,selected=selected,prediction_unknown=u,hypotheses=[asdict(h) for h in hs],timing_ms=dict(baseline_core=basecost+selection,with_slow_core=basecost+slowcost+slowgen+selection,slow_extra=slowcost+slowgen,hypotheses_and_road_support=hypms,research_total=(time.perf_counter()-begin)*1000),VehicleExecutedSuccessfully='NotRun')
            records.append(rec);print(req.scene_id,baseline,'->',selected,'unknown',len(u),flush=True)
    write(f'reports/m7c_{split}{suffix}.json',dict(kind='ActualOfflineCARLA0915Map_SyntheticActors',provenance=provenance(),protocol=asset('configs/m7c/protocol.json'),records=records,roads=road_records,cache=dict(hits=cache.hits,misses=cache.misses),reference_scope='0.025s independent polygon future samples, not exact mesh truth',online='NotRun'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','validation','fresh_final'],required=True);p.add_argument('--suffix',required=True);a=p.parse_args();run(a.split,a.suffix)
