"""Planner-only subprocess. Receives current packets, map, frozen protocol only."""
import argparse, json, sys, time
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
import carla
from shapely.geometry import Polygon
from tools.m7e_common import read, write, sha, provenance
from diagnostics.m7e_trajectory import state_anchored_slow, InitialStateConnectionInfeasible
from diagnostics.m7e_region import StaticRegionCache
from datasets.m7d_temporal import from_packet, result_clock
from diagnostics.m7d_scene import RoadIndex
from diagnostics.m7d_geometry import SnapshotPolygons, stopping_reserve_reused
from diagnostics.m7c_geometry import GeometryCache, evaluate_cached
from diagnostics.m7c_slow import slow_candidate, stopping_reserve
from diagnostics.m7b_motion import ArcRoute, build_hypotheses, evaluate_dynamic_hypotheses
from diagnostics.m7a_planning import generate
from simulator.oracle_driver import footprint


def plan(inp, roads, cache, cfg, *, region_cache, map_sha, connection_cfg, enable_f=False):
    reused=True
    start=time.perf_counter();req=inp.request();req.validate(req.scene.stamp.timestamp_s);request_ms=(time.perf_counter()-start)*1000
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
    f_error=None;f_info=None;f_gen_start=time.perf_counter()
    if enable_f:
        try:
            if inp.signed_speed_mps<0:raise InitialStateConnectionInfeasible('negative_measured_speed_no_clipping')
            fc,f_info=state_anchored_slow(req,connection_cfg);cs.append(fc)
        except InitialStateConnectionInfeasible as e:f_error=str(e)
    f_gen_ms=(time.perf_counter()-f_gen_start)*1000
    es=[];object_ms=0.
    for c in cs:
        allowed=set(keys);side='left' if c.name=='B_left' else 'right' if c.name=='C_right' else None
        if side in sides:
            permitted=sides[side]['permitted'];c=replace(c,lane_change_permitted=permitted)
            if permitted:allowed.update(sides[side]['keys'])
        t=time.perf_counter();region=region_cache.region(map_sha,roads.polygons,allowed,.03) if region_cache is not None else roads.region(allowed).buffer(-.03)
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
        if c.name in ['E_slow_follow','F_state_anchored_slow']:
            reserve=stopping_reserve_reused(c,req.scene,hs,ArcRoute(req.route),polys) if reused else stopping_reserve(c,req.scene,hs,ArcRoute(req.route))
        reservems=(time.perf_counter()-t)*1000
        object_start=time.perf_counter()
        es.append(dict(name=c.name,candidate=asdict(c),geometry=asdict(geom),dynamic=dyn,stop_reserve=reserve,
                       accepted=geom.planning_feasible and dyn['robust'] and (reserve is None or reserve['passed']),
                       allowed_lanes=sorted(allowed),timing_ms=dict(road_body=geomms,dynamic_SAT=dynms,slow_reserve=reservems)))
        object_ms+=(time.perf_counter()-object_start)*1000
    selection_start=time.perf_counter()
    order=['A_keep_route','B_left','C_right','E_slow_follow','F_state_anchored_slow','D_stop']
    selected=next((n for n in order if any(e['name']==n and e['accepted'] for e in es)),None)
    selection_ms=(time.perf_counter()-selection_start)*1000
    end=time.perf_counter();object_start=time.perf_counter()
    result=dict(episode_id=inp.episode_id,frame=inp.frame,request=asdict(req),evaluations=es,
                selected=selected,baseline_selected=next((n for n in order if n!='F_state_anchored_slow' and any(e['name']==n and e['accepted'] for e in es)),None),
                hypotheses=[asdict(h) for h in hs],prediction_unknown=u,actor_route_support=routes,slow_generation_error=slow_error,
                clock=asdict(result_clock(inp,start,end,selected)),
                f_connection_error=f_error,f_connection_info=f_info,
                counts=dict(candidates=len(es),geometry_pass=sum(e['geometry']['planning_feasible'] for e in es),dynamic_pass=sum(e['dynamic']['robust'] for e in es),accepted=sum(e['accepted'] for e in es)),
                timing_ms=dict(snapshot_to_request=request_ms,selection=selection_ms,return_objects=object_ms,F_generation=f_gen_ms,route_query=roadms,hypotheses=hypms,actor_polygon_setup=polyms,future_body_support=supportms,generation=genms,total=(end-start)*1000),
                cache=dict(hits=cache.hits,misses=cache.misses),FeasibleToExecute='NotValidated',control_authorized=False)
    result['timing_ms']['return_objects']+=(time.perf_counter()-object_start)*1000
    result['region_cache']=dict(hits=region_cache.hits,misses=region_cache.misses) if region_cache else None
    return result


def run(episode,arm,suffix='',input_root='reports/m7d_inputs'):
    manifest=read(Path(input_root)/episode/'manifest.json');mp=manifest['map']['path'];map_sha=manifest['map']['sha256'];cfg=read('configs/m7b/protocol.json');ccfg=read('configs/m7e/protocol.json');meta=provenance()
    setup_start=time.perf_counter();cmap=carla.Map('Town03',Path(mp).read_text());roads=RoadIndex(cmap);setup_ms=(time.perf_counter()-setup_start)*1000
    cache=GeometryCache(map_sha);regions=None if arm=='A_uncached' else StaticRegionCache();previous=None;records=[];opened=[];denied=[]
    allowed={str(Path(x['input']['path']).resolve()) for x in manifest['records']}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=str(Path(args[0]).resolve());mode=args[1];flags=args[2]
            reading=(isinstance(mode,str) and 'r' in mode) or (mode is None and isinstance(flags,int) and flags&3==0)
            if reading:
                if path not in allowed:denied.append(path);raise PermissionError('PlannerReadOutsideCurrentPacket:'+path)
                opened.append(path)
    sys.addaudithook(guard)
    for item in manifest['records']:
        delivery_start=time.perf_counter();packet=read(item['input']['path']);decode_start=time.perf_counter()
        try:
            inp=from_packet(packet);inp.validate(previous);decode_ms=(time.perf_counter()-decode_start)*1000
            rec=plan(inp,roads,cache,cfg,region_cache=regions,map_sha=map_sha,connection_cfg=ccfg,enable_f=arm=='B');previous=inp
        except (ValueError,TypeError) as e:
            rec=dict(episode_id=episode,frame=packet['frame'],selected=None,input_error=str(e),evaluations=[],control_authorized=False)
            decode_ms=(time.perf_counter()-decode_start)*1000
        rec['input_index']=item['index'];serial_start=time.perf_counter();payload=json.dumps(rec,allow_nan=False).encode();ready=time.perf_counter()
        rec['delivery']=dict(original_timestamp_s=packet['sim_time'],input_available_wall_s=packet['input_available_wall_s'],read_start_wall_s=delivery_start,
            payload_readable_wall_s=ready,full_ms=(ready-delivery_start)*1000,snapshot_decode_validation_ms=decode_ms,serialization_ms=(ready-serial_start)*1000,bytes=len(payload),
            modeled_sim_available=packet['sim_time']+ready-delivery_start,expired=ready-delivery_start>.2,valid_until_s=packet['sim_time']+.2,
            meaning='Complete scientific result bytes in memory, excludes subsequent diagnostic envelope/archive; no IPC delivery or actual vehicle execution')
        records.append(rec);print(episode,arm,item['index'],rec['selected'],rec.get('f_connection_error'),round(rec['delivery']['full_ms'],2),flush=True)
    write(f'reports/m7e_replays/{episode}_{arm}{suffix}.json',dict(kind='ActualServerOfflineMeasuredCARLAReplay',provenance=meta,episode=episode,arm=arm,split=manifest['split'],input_root=input_root,map_setup_ms=setup_ms,records=records,
        access_audit=dict(opened=opened,denied=denied,future_available=False),FeasibleToExecute='NotValidated'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--arm',choices=['A_uncached','A','B'],required=True);p.add_argument('--suffix',default='');p.add_argument('--input-root',default='reports/m7d_inputs');a=p.parse_args();run(a.episode,a.arm,a.suffix,a.input_root)
