"""Current-snapshot privileged stopping geometry. All XY in HS world metres."""
import math
import numpy as np
from simulator.oracle_driver import footprint,polygons_intersect,wrap
from simulator.motion_oracle import route_projection,clearance


def polygon_gap(a,b):
    if polygons_intersect(a,b):return 0.
    distances=[]
    for vertices,edges in [(a,b),(b,a)]:
        delta=np.roll(edges,-1,axis=0)-edges
        offset=vertices[:,None,:]-edges[None]
        f=np.clip((offset*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-12),0,1)
        distances.append(np.linalg.norm(offset-f[:,:,None]*delta,axis=-1).min())
    return float(min(distances))


def body_box(measurement,reference):
    import carla
    bb=carla.BoundingBox(carla.Location(*reference['location']),carla.Vector3D(*reference['extent']))
    roll,pitch,yaw=reference['rotation'];bb.rotation=carla.Rotation(roll=roll,pitch=pitch,yaw=yaw)
    roll,pitch,yaw=measurement['rotation'];tf=carla.Transform(carla.Location(*measurement['position']),carla.Rotation(roll=roll,pitch=pitch,yaw=yaw))
    xyz=np.array([[p.x,-p.y,p.z] for p in bb.get_world_vertices(tf)]);center=xyz.mean(0);angle=-math.radians(yaw+bb.rotation.yaw)
    c,s=math.cos(angle),math.sin(angle);extent=abs((xyz[:,:2]-center[:2])@np.array([[c,-s],[s,c]])).max(0)
    return dict(xy=center[:2].tolist(),yaw=angle,half_length=float(extent[0]),half_width=float(extent[1]))


def stopping_geometry(route,state,goal,ego_box,obstacles):
    r=np.asarray(route);s,arc,index,_=route_projection(r,[state.x,state.y]);target=np.asarray(goal.target_position_xy)
    tangent=np.array([math.cos(goal.target_yaw),math.sin(goal.target_yaw)]);offset=target-[state.x,state.y]
    signed=float(offset@tangent);lateral=float(offset@np.array([-tangent[1],tangent[0]]))
    route_remaining=float(arc[-1]-s)
    if signed<0:route_remaining=min(route_remaining,signed)
    ego=footprint(*ego_box['xy'],ego_box['yaw'],ego_box['half_length'],ego_box['half_width'])
    gaps=[];relevant=[]
    for o in obstacles:
        gap=polygon_gap(ego,footprint(*o['xy'],o['yaw'],o['half_length'],o['half_width']))
        gaps.append(dict(actor_id=o['actor_id'],gap_m=gap))
        # Relevant current path obstacle uses frozen route selection semantics;
        # all boxes still reach original six-second Safety and current guard.
        ahead=clearance(route,state,[o],ego_box['half_length'],ego_box['half_width'])
        if ahead:relevant.append((gap,o['actor_id']))
    obstacle_gap=min((g for g,_ in relevant),default=None)
    available=route_remaining if obstacle_gap is None else min(route_remaining,obstacle_gap-3.)
    return dict(source='Privileged',coordinate_frame='hs_world_x_y_left_z_up_m',timestamp=state.timestamp_s,
        route_remaining_s=route_remaining,goal_signed_distance_m=signed,goal_lateral_error_m=lateral,
        goal_heading_error_rad=float(wrap(goal.target_yaw-state.yaw)),obstacle_obb_clearance_m=obstacle_gap,
        all_obstacle_gaps=gaps,required_safe_gap_m=3.,available_clearance_to_safe_stop=available,
        reference='Ego Actor origin for goal; exact projected Actor OBB including local offset/tilt for clearance; not render mesh')


def road_grade(cmap,location):
    import carla
    wp=cmap.get_waypoint(carla.Location(*location));nexts=wp.next(2.)
    grade=math.sin(math.radians(wp.transform.rotation.pitch))
    if nexts:
        n=min(nexts,key=lambda q:abs(wrap(math.radians(q.transform.rotation.yaw-wp.transform.rotation.yaw))))
        a=wp.transform.location;b=n.transform.location;distance=math.hypot(b.x-a.x,b.y-a.y)
        if distance>.1:grade=(b.z-a.z)/distance
    return dict(grade=float(grade),road_id=wp.road_id,lane_id=wp.lane_id,map_pitch_deg=float(wp.transform.rotation.pitch),
        reference='OpenDRIVE height difference over2m; positive uphill; vehicle pitch separately measured')
