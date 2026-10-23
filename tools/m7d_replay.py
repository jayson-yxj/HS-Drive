"""Planner-only subprocess. Receives current packets, map, frozen protocol only."""
import argparse, json, sys, time
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
import carla
from shapely.geometry import Polygon
from tools.m7d_common import read, write, sha, provenance
from datasets.m7d_temporal import from_packet, result_clock
from diagnostics.m7d_scene import RoadIndex
from diagnostics.m7d_geometry import SnapshotPolygons, stopping_reserve_reused
from diagnostics.m7c_geometry import GeometryCache, evaluate_cached
from diagnostics.m7c_slow import slow_candidate, stopping_reserve
from diagnostics.m7b_motion import ArcRoute, build_hypotheses, evaluate_dynamic_hypotheses
from diagnostics.m7a_planning import generate
from simulator.oracle_driver import footprint


def plan(inp, roads, cache, cfg, reused=True):
    start=time.perf_counter();req=inp.request();req.validate(req.scene.stamp.timestamp_s)
    t=time.perf_counter();keys,sides,missing=roads.ego_support(inp)
    obs,routes,u,actor_regions=roads.observations(inp)
    if missing:u.append(dict(reason='EgoRouteMapSupportMissing',points=missing))
    roadms=(time.perf_counter()-t)*1000
    t=time.perf_counter();hs,hu=build_hypotheses(req.scene,obs,cfg);u+=hu
    hypms=(time.perf_counter()-t)*1000
    t=time.perf_counter();polys=SnapshotPolygons(req.scene,hs) if reused else None
    polyms=(time.perf_counter()-t)*1000
    t=time.perf_counter()
    for h in hs:
        b=next(x for x in req.scene.boxes if x.object_id==h.object_id)
        ps=polys.actors[(h.object_id,h.name)] if reused else [Polygon(footprint(*p[:2],p[2],b.half_length,b.half_width)) for p in h.poses]
        if any(not actor_regions[h.object_id].covers(p) for p in ps):
            u.append(dict(object_id=h.object_id,hypothesis=h.name,reason='FutureBodyOutsideSupportedRoad'))
    supportms=(time.perf_counter()-t)*1000
    t=time.perf_counter();cs=generate(req,left_offset=sides.get('left',{}).get('offset'),right_offset=sides.get('right',{}).get('offset'))
    slow_error=None
    try:cs.append(slow_candidate(req))
    except ValueError as e:slow_error=str(e)
    genms=(time.perf_counter()-t)*1000
    es=[]
    for c in cs:
        allowed=set(keys);side='left' if c.name=='B_left' else 'right' if c.name=='C_right' else None
        if side in sides:
            permitted=sides[side]['permitted'];c=replace(c,lane_change_permitted=permitted)
            if permitted:allowed.update(sides[side]['keys'])
        t=time.perf_counter();region=roads.region(allowed).buffer(-.03)
        def exact(x,y):
            w=roads.waypoint((x,y),inp.ego_z)
            return w is not None and roads.key(w) in allowed
        # Stopped vehicles retain acceleration hypotheses; never relabel as static geometry.
        greq=replace(req,scene=replace(req.scene,boxes=()))
        geom=evaluate_cached(greq,c,region,req.scene.stamp.timestamp_s,cache=cache,
                             exact_road=cache.exact(allowed,inp.ego_z,exact))
        geomms=(time.perf_counter()-t)*1000
        t=time.perf_counter();dyn=evaluate_dynamic_hypotheses(c,req.scene,hs,u);dynms=(time.perf_counter()-t)*1000
        t=time.perf_counter();reserve=None
        if c.name=='E_slow_follow':
            reserve=stopping_reserve_reused(c,req.scene,hs,ArcRoute(req.route),polys) if reused else stopping_reserve(c,req.scene,hs,ArcRoute(req.route))
        reservems=(time.perf_counter()-t)*1000
        es.append(dict(name=c.name,candidate=asdict(c),geometry=asdict(geom),dynamic=dyn,stop_reserve=reserve,
                       accepted=geom.planning_feasible and dyn['robust'] and (reserve is None or reserve['passed']),
                       allowed_lanes=sorted(allowed),timing_ms=dict(road_body=geomms,dynamic_SAT=dynms,slow_reserve=reservems)))
    order=['A_keep_route','B_left','C_right','E_slow_follow','D_stop']
    selected=next((n for n in order if any(e['name']==n and e['accepted'] for e in es)),None)
    end=time.perf_counter()
    return dict(episode_id=inp.episode_id,frame=inp.frame,request=asdict(req),evaluations=es,
                selected=selected,baseline_selected=next((n for n in order if n!='E_slow_follow' and any(e['name']==n and e['accepted'] for e in es)),None),
                hypotheses=[asdict(h) for h in hs],prediction_unknown=u,actor_route_support=routes,slow_generation_error=slow_error,
                clock=asdict(result_clock(inp,start,end,selected)),
                counts=dict(candidates=len(es),geometry_pass=sum(e['geometry']['planning_feasible'] for e in es),dynamic_pass=sum(e['dynamic']['robust'] for e in es),accepted=sum(e['accepted'] for e in es)),
                timing_ms=dict(route_query=roadms,hypotheses=hypms,actor_polygon_setup=polyms,future_body_support=supportms,generation=genms,total=(end-start)*1000),
                cache=dict(hits=cache.hits,misses=cache.misses),FeasibleToExecute='NotValidated',control_authorized=False)


def run(episode,mode,limit=None):
    manifest_path=Path('reports/m7d_inputs')/episode/'manifest.json';manifest=read(manifest_path)
    mp=manifest['map']['path'];cfg=read('configs/m7b/protocol.json');meta=provenance()
    cmap=carla.Map('Town03',Path(mp).read_text());start=time.perf_counter();roads=RoadIndex(cmap);setup_ms=(time.perf_counter()-start)*1000
    allowed={str(Path(x['input']['path']).resolve()) for x in manifest['records']}
    opened=[];denied=[]
    # Audit after imports/map setup. No raw episode/evaluator files are allowed.
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(args[0]).resolve();m=args[1];flags=args[2]
            isread=(isinstance(m,str) and 'r' in m) or (m is None and isinstance(flags,int) and flags&3==0)
            if isread:
                if str(path) not in allowed:
                    denied.append(str(path));raise PermissionError('PlannerReadOutsideCurrentPacketAllowlist:'+str(path))
                opened.append(str(path))
    sys.addaudithook(audit)
    cache=GeometryCache(sha(mp)) if False else GeometryCache(manifest['map']['sha256'])
    previous=None;records=[]
    for item in manifest['records'][:limit]:
        p=read(item['input']['path']);inp=None
        try:
            inp=from_packet(p);inp.validate(previous)
            if mode in ['cold','baseline']:cache=GeometryCache(manifest['map']['sha256'])
            rec=plan(inp,roads,cache,cfg,reused=mode!='baseline');previous=inp
        except (ValueError,TypeError) as e:
            rec=dict(episode_id=episode,frame=p['frame'],input_error=str(e),selected=None,control_authorized=False,FeasibleToExecute='NotValidated')
        rec['input_index']=item['index'];records.append(rec)
        print(episode,mode,item['index'],rec.get('selected'),rec.get('input_error'),round(rec.get('timing_ms',{}).get('total',0),2),flush=True)
    write(f'reports/m7d_replays/{episode}_{mode}.json',dict(kind='ActualServerOfflineReplayOfMeasuredCARLA',provenance=meta,episode=episode,split=manifest['split'],mode=mode,setup_ms=setup_ms,
        records=records,access_audit=dict(allowed_current_packet_files=sorted(allowed),opened=opened,denied=denied,future_available_to_planner=False),FeasibleToExecute='NotValidated'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--mode',choices=['stream','cold','baseline'],default='stream');p.add_argument('--limit',type=int);a=p.parse_args();run(a.episode,a.mode,a.limit)
