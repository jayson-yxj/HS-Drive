"""CARLA SDK offline OpenDRIVE fixtures. No Client, world.tick, or actuation."""
import argparse,math,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import carla
from shapely.geometry import Polygon
from shapely.ops import unary_union
from datasets.m7a_contracts import Stamp,OracleSceneState,PlanningRequest,Box,WORLD
from diagnostics.m7a_planning import generate,evaluate,select
from tools.m7a_common import read,write,asset,provenance


def pose(w):
    t=w.transform
    return (t.location.x,-t.location.y,-math.radians(t.rotation.yaw))


def route_for(cmap,road):
    # Deterministic map-only choice, no candidate result consulted.
    ws=[w for w in cmap.generate_waypoints(2.) if w.road_id==road and not w.is_junction and w.lane_type==carla.LaneType.Driving]
    lane=min(w.lane_id for w in ws)
    ws=[w for w in ws if w.lane_id==lane]
    ws.sort(key=lambda w:w.s,reverse=lane>0)
    if len(ws)<15:raise ValueError('insufficient_road_support')
    start=ws[3];route=[start];cur=start
    for _ in range(240):
        nxt=[w for w in cur.next(.5) if w.road_id==road and w.lane_id==lane and not w.is_junction]
        if not nxt:break
        cur=nxt[0];route.append(cur)
        if len(route)>=181:break
    if len(route)<35:raise ValueError('short_independent_road')
    # Trim final 3m to avoid a footprint crossing unrepresented connector.
    return route[:-6]


def lane_polygon(wps):
    quads=[]
    for a,b in zip(wps[:-1],wps[1:]):
        pts=[]
        for w,sign in [(a,1),(b,1),(b,-1),(a,-1)]:
            x,y,yaw=pose(w);h=w.lane_width/2
            pts.append([x-sign*h*math.sin(yaw),y+sign*h*math.cos(yaw)])
        q=Polygon(pts)
        if q.is_valid:quads.append(q)
    return unary_union(quads)


def scene_fixture(cmap,road,split):
    route=route_for(cmap,road);first=route[0]
    # Road region extends behind/after route for complete body envelope.
    allwp=[w for w in cmap.generate_waypoints(.5) if w.road_id==road and w.lane_type==carla.LaneType.Driving and w.lane_id*first.lane_id>0]
    lanes={lane:sorted([w for w in allwp if w.lane_id==lane],key=lambda w:w.s) for lane in {w.lane_id for w in allwp}}
    polys={lane:lane_polygon(wps) for lane,wps in lanes.items()}
    legal_sides={}
    for side,neighbor,flag,back in [('left',first.get_left_lane(),carla.LaneChange.Left,carla.LaneChange.Right),('right',first.get_right_lane(),carla.LaneChange.Right,carla.LaneChange.Left)]:
        if neighbor is None or neighbor.road_id!=road or neighbor.lane_id*first.lane_id<=0 or neighbor.lane_type!=carla.LaneType.Driving:continue
        permitted=True
        for w in route:
            n=w.get_left_lane() if side=='left' else w.get_right_lane()
            mark=w.left_lane_marking if side=='left' else w.right_lane_marking
            if n is None or not(int(w.lane_change)&int(flag)) or not(int(n.lane_change)&int(back)) or mark.type!=carla.LaneMarkingType.Broken:
                permitted=False;break
        legal_sides[side]=dict(offset=(1 if side=='left' else -1)*.5*(first.lane_width+neighbor.lane_width),permitted=permitted,lane=neighbor.lane_id)
    return dict(road=road,split=split,route=[pose(w) for w in route],lane=first.lane_id,
      polygons=polys,sides=legal_sides,z=first.transform.location.z)


def construct(f,kind,frame):
    r=np.array(f['route']);start=r[0];stamp=Stamp(frame,100.,100.2,WORLD)
    tangent=np.array([math.cos(start[2]),math.sin(start[2])]);left=np.array([-tangent[1],tangent[0]])
    def obj(distance,lateral=0.,speed=0.,yaw=None,ident='obstacle',hw=.9):
        return Box(ident,tuple(start[:2]+tangent*distance+left*lateral),start[2] if yaw is None else yaw,1.9,hw,tuple(tangent*speed))
    boxes=();future={};speed=2.5;fault=None
    if kind in ['static_bypass','lane_change_prohibited']:boxes=(obj(40),)
    if kind in ['blocked_far_stop','blocked_near_reject']:
        d=40 if kind=='blocked_far_stop' else 3.8
        boxes=tuple(obj(d,lateral,ident='block'+str(i)) for i,lateral in enumerate([0]+[v['offset'] for v in f['sides'].values()]))
    if kind in ['lead_braking','lead_stopped','lead_restart']:
        lead_speed={'lead_braking':2.,'lead_stopped':0.,'lead_restart':1.}[kind]
        boxes=(obj(14,speed=lead_speed,ident='lead'),)
        times=np.arange(0,6.01,.1)
        distances=np.minimum(times,2)*2-.5*np.minimum(times,2)**2 if kind=='lead_braking' else (np.maximum(times-2,0) if kind=='lead_restart' else np.zeros_like(times))
        future={'lead':[(float(t),tuple(np.array(boxes[0].xy)+tangent*d)) for t,d in zip(times,distances)]}
    if kind in ['crossing_conflict','crossing_clear_time']:
        # Same spatial crossing, different time. CV future uses only snapshot velocity.
        crossing_t=3. if kind=='crossing_conflict' else .2
        boxes=(Box('crossing',tuple(start[:2]+tangent*7.5+left*(-2*crossing_t)),start[2]+np.pi/2,1.,.4,tuple(left*2)),)
    if kind in ['expired','future','missing','worker_error']:fault=kind
    body=read('reports/m6b_body_reference.json')
    scene=OracleSceneState(stamp,tuple(start)+(speed,),boxes,
        tuple(body['extent'][:2]),(body['location'][0],-body['location'][1]))
    request=PlanningRequest(scene,tuple(map(tuple,r)),f"{f['split']}_road{f['road']}_{kind}",tuple(r[-1]))
    return request,future,fault


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['development','validation','final'],required=True);ap.add_argument('--suffix',default='');args=ap.parse_args()
    cfg=read('configs/m7a/protocol.json');cmap=carla.Map('Town03',Path(cfg['planning_map']).read_text());catalog=[];records=[];timings=[]
    for road in cfg['location_split'][args.split]:
        fixture=scene_fixture(cmap,road,args.split)
        for i,kind in enumerate(cfg['scenarios']):
            req,future,fault=construct(fixture,kind,road*100+i);start=time.perf_counter()
            side=fixture['sides'];candidates=generate(req,left_offset=side.get('left',{}).get('offset'),right_offset=side.get('right',{}).get('offset'),lane_permitted=kind!='lane_change_prohibited')
            genms=(time.perf_counter()-start)*1000;evaluations=[]
            if fault in ['missing','worker_error']:candidates=[]
            for c in candidates:
                allowed={fixture['lane']}
                s='left' if c.name=='B_left' else 'right' if c.name=='C_right' else None
                if s in side:
                    permitted=side[s]['permitted'] and kind!='lane_change_prohibited'
                    c=replace(c,lane_change_permitted=permitted)
                    if permitted:allowed.add(side[s]['lane'])
                region=unary_union([fixture['polygons'][k] for k in allowed]).buffer(-.03)
                def exact(x,y):
                    w=cmap.get_waypoint(carla.Location(x=float(x),y=-float(y),z=fixture['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
                    return w is not None and w.road_id==road and w.lane_id in allowed
                now=100.201 if fault=='expired' else 99.99 if fault=='future' else 100.
                e=evaluate(req,c,region,now,exact_road=exact,future_reference=future);evaluations.append(e)
            decision=select(evaluations)
            expectation='reject' if kind in ['blocked_near_reject','expired','future','missing','worker_error'] else 'stop' if kind in ['blocked_far_stop','lane_change_prohibited','lead_stopped'] else 'safe_candidate'
            # These are predeclared desired outcomes, not a relabeling of failures.
            correct=(decision['selected'] is None) if expectation=='reject' else decision['selected']=='D_stop' if expectation=='stop' else decision['selected'] is not None
            elapsed=(time.perf_counter()-start)*1000;timings.append(elapsed)
            rec=dict(scene_id=req.scene_id,road=road,split=args.split,kind=kind,request=asdict(req),
              candidates=[asdict(c) for c in candidates],evaluations=[asdict(e) for e in evaluations],decision=decision,
              expected=expectation,expectation_met=correct,generation_ms=genms,total_ms=elapsed,
              future_reference_evaluator_only=future,vehicle_executed_successfully='NotRun',evidence='OfflineCARLA0915Map_WithExplicitSyntheticActors')
            records.append(rec)
            catalog.append(dict(scene_id=req.scene_id,road=road,split=args.split,kind=kind,expected=expectation,
              source='Privileged_map_synthetic_actor_fixture',route=fixture['route'],lane=fixture['lane'],sides=fixture['sides'],
              geometric_road_area_m2=sum(p.area for p in fixture['polygons'].values()),vehicle_execution='NotRun'))
            print(req.scene_id,decision,correct,flush=True)
    write(f'reports/m7a_planner_{args.split}{args.suffix}.json',dict(kind='OfflineCARLA0915Map_WithSyntheticScenarioActors',provenance=provenance(),
      map=asset(cfg['planning_map']),catalog=catalog,records=records,
      summary=dict(scenarios=len(records),expectation_met=sum(r['expectation_met'] for r in records),generation_success=sum(bool(r['candidates']) for r in records),
        timing_ms={k:float(np.quantile(timings,q)) for k,q in [('p50',.5),('p95',.95),('p99',.99),('max',1)]}),
      limits=['No live vehicle execution','OpenDRIVE lane polygon approximation not mesh truth','Future scripted acceleration only evaluator; planner uses CV snapshot','No intersection traffic light or right-of-way certification','Road IDs isolated; pattern family shared; not independent map generalization']))

if __name__=='__main__':main()
