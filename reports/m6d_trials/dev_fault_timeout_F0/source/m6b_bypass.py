"""Static privileged lane-change candidate, full-box geometry before execution."""
import math
import numpy as np
from simulator.oracle_driver import footprint,polygons_intersect,wrap


def smoothstep(t):
    t=np.clip(t,0,1);return 10*t**3-15*t**4+6*t**5


def offset_path(route,offset,entry=(5.,30.),exit=(55.,80.),spacing=.25):
    r=np.asarray(route,float);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(r[:,:2],axis=0),axis=1))]
    q=np.arange(0,s[-1]+1e-8,spacing)
    if q[-1]<s[-1]:q=np.r_[q,s[-1]]
    base=np.stack([np.interp(q,s,r[:,i]) for i in (0,1)],1);heading=np.interp(q,s,np.unwrap(r[:,2]))
    d=offset*smoothstep((q-entry[0])/(entry[1]-entry[0]))*(1-smoothstep((q-exit[0])/(exit[1]-exit[0])))
    xy=base+d[:,None]*np.stack([-np.sin(heading),np.cos(heading)],1)
    delta=np.gradient(xy,axis=0);yaw=np.arctan2(delta[:,1],delta[:,0]);return np.c_[xy,yaw]


def body_samples(x,y,yaw,hl,hw,spacing=.2):
    xx=np.linspace(-hl,hl,int(np.ceil(2*hl/spacing))+1);yy=np.linspace(-hw,hw,int(np.ceil(2*hw/spacing))+1)
    a,b=np.meshgrid(xx,yy);v=np.c_[a.ravel(),b.ravel()];c,s=np.cos(yaw),np.sin(yaw)
    return v@np.array([[c,s],[-s,c]])+[x,y]


def audit_candidate(path,obstacles,road,hl,hw,curvature_limit=.10):
    p=np.asarray(path);ds=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
    curvature=abs(wrap(np.diff(p[:,2])))/np.maximum(ds,1e-6)
    report=dict(max_curvature=float(curvature.max()),road_queries=0,source='Privileged',accepted=False)
    if curvature.max()>curvature_limit:report['reason']='curvature_limit';return report
    # Conservative rotation/translation padding covers between sampled poses
    # for static rectangles; roads remain discretely queried, not exact mesh.
    pad=float(ds.max()/2+np.hypot(hl,hw)*abs(wrap(np.diff(p[:,2]))).max()/2)
    for x,y,yaw in p:
        ego=footprint(x,y,yaw,hl+.2+pad,hw+.15+pad)
        for o in obstacles:
            if np.linalg.norm(o.get('velocity',[0,0]))>.05:
                report['reason']='not_static';return report
            if polygons_intersect(ego,footprint(*o['xy'],o['yaw'],o['half_length'],o['half_width'])):
                report['reason']='obstacle_full_rectangle';return report
        for xx,yy in body_samples(x,y,yaw,hl+.2,hw+.15):
            report['road_queries']+=1
            if not road(float(xx),float(yy)):
                report['reason']='body_outside_legal_driving_lanes';return report
    report.update(accepted=True,reason=None,collision_padding_m=pad,
                  road_limit='dense .2m body grid/.25m path samples; not exact continuous curb or mesh proof')
    return report


def choose_bypass(cmap,route,obstacles,hl,hw,z):
    import carla
    r=np.asarray(route,float)
    def wp_at(x,y):return cmap.get_waypoint(carla.Location(x=float(x),y=float(-y),z=z),project_to_road=False,lane_type=carla.LaneType.Driving)
    first=wp_at(*r[0,:2]);reports=[];choices=[]
    original_lane=(first.road_id,first.lane_id)
    for side,sign in [('left',1),('right',-1)]:
        neighbor=first.get_left_lane() if side=='left' else first.get_right_lane()
        reason=None
        if neighbor is None or neighbor.lane_type!=carla.LaneType.Driving or neighbor.lane_id*first.lane_id<=0:
            reason='no_same_direction_driving_neighbor'
        else:
            for pt in r:
                w=wp_at(*pt[:2]);n=w.get_left_lane() if side=='left' else w.get_right_lane()
                flag=carla.LaneChange.Left if side=='left' else carla.LaneChange.Right
                back=carla.LaneChange.Right if side=='left' else carla.LaneChange.Left
                mark=w.left_lane_marking if side=='left' else w.right_lane_marking
                if n is None or n.lane_id*w.lane_id<=0 or n.lane_type!=carla.LaneType.Driving or w.is_junction or not(int(w.lane_change)&int(flag)) or not(int(n.lane_change)&int(back)) or mark.type!=carla.LaneMarkingType.Broken:
                    reason='lane_change_or_return_not_permitted';break
        if reason:reports.append(dict(side=side,accepted=False,reason=reason));continue
        allowed={original_lane,(neighbor.road_id,neighbor.lane_id)}
        def legal(x,y):
            w=wp_at(x,y)
            return w is not None and (w.road_id,w.lane_id) in allowed
        width=.5*(first.lane_width+neighbor.lane_width)
        path=offset_path(r,sign*width)
        report=audit_candidate(path,obstacles,legal,hl,hw)
        report.update(side=side,allowed_lanes=[list(v) for v in sorted(allowed)],offset_m=sign*width)
        reports.append(report)
        if report['accepted']:choices.append((path,report))
    return (choices[0][0] if choices else r),dict(source='Privileged',status='FeasibleBypass' if choices else 'NoFeasibleBypass',candidates=reports,selected=choices[0][1]['side'] if choices else None)
