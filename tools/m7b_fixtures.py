"""Map-only fixture construction and independent finite-difference future audit."""
from dataclasses import replace
from itertools import combinations
import numpy as np
from shapely.geometry import Polygon
from tools.m7a_scenarios import scene_fixture,pose
from diagnostics.m7b_motion import ArcRoute,ObservedMotion,longitudinal
from datasets.m7a_contracts import Box,Stamp,OracleSceneState,PlanningRequest,WORLD
from simulator.oracle_driver import footprint
from simulator.motion_oracle import wrap
from tools.m7b_common import read


def scenarios():
    out=[dict(name='clear',kind='clear',ego=2.5),dict(name='static_bypass',kind='static',ego=2.5,d=40),dict(name='blocked_far',kind='blocked',ego=2.5,d=40),dict(name='blocked_close',kind='blocked',ego=4.,d=8),dict(name='prohibited_lane',kind='prohibited',ego=2.5,d=40)]
    for i,(ego,v,d,a,onset,length) in enumerate([(2.5,2,14,-1,0,1.9),(4,3,18,-2.2,.5,2.4),(5,3,22,-3,1,2.8),(3,1,10,-.7,0,1.5),(4,2,12,-1.7,1.5,1.9),(2,3,25,-2,2,2.4),(2.5,2,16,-.4,.5,1.9)]):
        out.append(dict(name=f'braking_{i}',kind='lead',ego=ego,v=v,d=d,a=a,onset=onset,hl=length))
    for i,(ego,d,a,onset) in enumerate([(2.5,14,1.,2.),(3,18,.5,1.),(2,10,1.3,3.)]):
        out.append(dict(name=f'restart_{i}',kind='lead',ego=ego,v=0,d=d,a=a,onset=onset))
    out.extend([dict(name='lead_accelerating',kind='lead',ego=4,v=2,d=13,a=1.2,onset=.5),dict(name='lead_stopped',kind='lead',ego=2.5,v=0,d=14,a=0,onset=0)])
    for name,arrival in [('crossing_conflict',3.),('crossing_early',.2)]:out.append(dict(name=name,kind='crossing',ego=2.5,arrival=arrival))
    for fault in ['expired','future','missing','worker_error']:out.append(dict(name=fault,kind='clear',ego=2.5,fault=fault))
    return out


def map_fixture(cmap,road,split):
    f=scene_fixture(cmap,road,split)
    f['arc']=ArcRoute(f['route'],[f['lane']]*len(f['route']))
    f['cmap']=cmap
    return f


def construct(f,spec,index,cfg):
    r=f['arc'];start=r.at(0);boxes=[];observations=[];references={};placements=[];errors=[]
    t=np.arange(0,6.000001,cfg['reference_dt_s'])
    def add(ident,path,s0,speed,mode,a=0.,onset=0.,hl=1.9,hw=.8,lane=None,road_required=True):
        try:
            p=path.at(s0);v=path.tangent(s0)*speed
            b=Box(ident,tuple(p[:2]),float(p[2]),hl,hw,tuple(v));boxes.append(b)
            observations.append(ObservedMotion(ident,path,s0,mode))
            ds,vel=longitudinal(t,speed,a,onset);future=path.at(s0+ds)
            references[ident]=dict(times=t.tolist(),poses=np.c_[future,vel].tolist(),a=a,onset=onset,mode=mode,source='EvaluatorOnlyScript_NotPlannerInput')
            poly=Polygon(footprint(*p[:2],p[2],hl,hw));valid=True
            if road_required:
                valid=lane in f['polygons'] and f['polygons'][lane].covers(poly)
                for xy in poly.exterior.coords:
                    import carla
                    w=f['cmap'].get_waypoint(carla.Location(x=float(xy[0]),y=-float(xy[1]),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
                    valid=valid and w is not None and w.road_id==f['road'] and w.lane_id==lane
            placements.append(dict(object_id=ident,expected_lane=lane,s=s0,route_center_error_m=path.project(p[:2])[1],heading_error=abs(float(wrap(p[2]-path.at(s0)[2]))),OBB_on_intended_lane=bool(valid),road_required=road_required))
            if not valid:errors.append('actor_OBB_outside_intended_lane:'+ident)
        except ValueError as e:errors.append(ident+':'+str(e))
    k=spec['kind']
    if k in ['static','blocked','prohibited']:
        add('obstacle',r,spec['d'],0.,'static_geometry',lane=f['lane'])
        if k=='blocked':
            # Resolve each neighbor at actual arc station through CARLA, not constant lateral offset.
            p=r.at(spec['d'])
            import carla
            wp=f['cmap'].get_waypoint(carla.Location(x=float(p[0]),y=-float(p[1]),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
            for side in ['left','right']:
                nb=getattr(wp,'get_'+side+'_lane')() if wp else None
                if nb is None or nb.road_id!=f['road'] or nb.lane_id*f['lane']<=0 or nb.lane_id not in f['polygons']:continue
                npose=np.array(pose(nb));waypoints=[nb]
                for _ in range(10):
                    nxt=waypoints[-1].next(.5)
                    if not nxt:break
                    waypoints.append(nxt[0])
                path=ArcRoute([pose(w) for w in waypoints],[w.lane_id for w in waypoints]);add('block_'+side,path,0.,0.,'static_geometry',lane=nb.lane_id)
    elif k=='lead':
        add('lead',r,spec['d'],spec['v'],'road_vehicle',spec['a'],spec['onset'],hl=spec.get('hl',1.9),lane=f['lane'])
    elif k=='crossing':
        p=r.at(12.);normal=np.array([-np.sin(p[2]),np.cos(p[2])]);origin=p[:2]-normal*2*spec['arrival']
        path=ArcRoute([(*origin,p[2]+np.pi/2),(*(origin+normal*80),p[2]+np.pi/2)])
        add('crossing',path,0.,2.,'crossing_corridor',hl=.4,hw=.3,road_required=False)
    st=Stamp(1000+f['road']*100+index,100.,100.2,WORLD);body=read('reports/m6b_body_reference.json')
    scene=OracleSceneState(st,tuple(start)+(spec['ego'],),tuple(boxes),tuple(body['extent'][:2]),(body['location'][0],-body['location'][1]))
    req=PlanningRequest(scene,tuple(map(tuple,r.points)),f"{f['split']}_road{f['road']}_{spec['name']}",tuple(r.points[-1]))
    ego=Polygon(footprint(*(start[:2]+r.tangent(0)*scene.body_center_offset[0]),start[2],*scene.body_half_extent))
    polys=[(b.object_id,Polygon(footprint(*b.xy,b.yaw,b.half_length,b.half_width))) for b in boxes]
    for name,p in polys:
        if ego.intersects(p):errors.append('initial_ego_overlap:'+name)
    for (an,a),(bn,b) in combinations(polys,2):
        if a.intersects(b):errors.append('initial_actor_overlap:'+an+':'+bn)
    future_checks=[]
    for b in boxes:
        if b.object_id not in references:continue
        ref=references[b.object_id];p=np.array(ref['poses']);tt=np.array(ref['times']);dt=np.diff(tt)
        fd=np.diff(p[:,:2],axis=0)/dt[:,None];a=np.diff(p[:,3])/dt
        v0err=float(np.linalg.norm(fd[0]-b.velocity));poserr=float(np.linalg.norm(p[0,:2]-b.xy))
        initial_v_error=abs(float(p[0,3])-np.linalg.norm(b.velocity))
        direction=np.arctan2(fd[:,1],fd[:,0]);moving=np.linalg.norm(fd,axis=1)>1e-4
        heading=float(np.max(abs(wrap(direction[moving]-(p[:-1,2]+np.diff(p[:,2])/2)[moving])))) if moving.any() else 0.
        road_good=True
        if ref['mode']!='crossing_corridor':
            lane=next(x['expected_lane'] for x in placements if x['object_id']==b.object_id)
            road_good=all(f['polygons'][lane].covers(Polygon(footprint(*x[:2],x[2],b.half_length,b.half_width))) for x in p[::4])
        ok=poserr<1e-6 and initial_v_error<1e-6 and v0err<=cfg['initial_velocity_tolerance_mps'] and a.min()>=-3.00001 and a.max()<=1.50001 and heading<.08 and road_good
        future_checks.append(dict(object_id=b.object_id,valid=bool(ok),initial_position_error_m=poserr,initial_speed_error=initial_v_error,first_difference_velocity_error_mps=v0err,acceleration_range=[float(a.min()),float(a.max())],heading_difference_rad=heading,full_OBB_road_reference=bool(road_good),jump_check='analytic bounded-speed displacement; finite difference checked'))
        if not ok:errors.append('future_invalid:'+b.object_id)
    return req,observations,references,dict(status='FixtureInvalid' if errors else 'FixtureValid',errors=errors,placements=placements,future_checks=future_checks)
