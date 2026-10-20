"""Versioned OpenDRIVE lane ribbons, including junctions, supervision only.

A junction ribbon is geometric road support, never right-of-way or an allowed
route. Conflicting lane headings and overlapping vertical levels are unknown.
"""
import cv2
import numpy as np
from datasets.drivable import validate_arrays

SCHEMA='hsd-road-supervision-v2'


def lane_ribbons(carla_map,spacing=1.,margin=.25):
    """Use CARLA's connected Driving lane waypoints, not a junction bounding box."""
    polygons=[];headings=[];junction=[]
    for wp in carla_map.generate_waypoints(spacing):
        for end in wp.next(spacing):
            if str(end.lane_type)!='Driving':continue
            a=wp.transform;b=end.transform
            ca=np.array([a.location.x,a.location.y,a.location.z]);cb=np.array([b.location.x,b.location.y,b.location.z])
            if np.linalg.norm(cb-ca)>spacing*2:continue
            wa=wp.lane_width/2-margin;wb=end.lane_width/2-margin
            if min(wa,wb)<=0:continue
            ra=a.get_right_vector();rb=b.get_right_vector()
            ra=np.array([ra.x,ra.y,ra.z]);rb=np.array([rb.x,rb.y,rb.z])
            polygons.append([ca-ra*wa,ca+ra*wa,cb+rb*wb,cb-rb*wb])
            f=a.get_forward_vector();headings.append([f.x,f.y,f.z])
            junction.append(wp.is_junction or end.is_junction)
    return {'polygons':np.asarray(polygons,np.float64),'headings':np.asarray(headings,np.float64),
            'junction':np.asarray(junction,bool)}


def project_ribbons(ribbons,carla_world_from_ego,config,base):
    """Rasterize small connected strips at fixed-point precision; preserve Unknown."""
    pose=np.asarray(carla_world_from_ego,dtype=np.float64);inv=np.linalg.inv(pose)
    poly=(ribbons['polygons']-pose[:3,3])@inv[:3,:3].T;poly[...,1]*=-1
    direction=ribbons['headings']@inv[:3,:3].T;direction[:,1]*=-1
    r=config['resolution_m'];x0,x1=config['x_range_m'];y0,y1=config['y_range_m']
    shape=base['drivable_static'].shape
    nearby=((poly[...,0].max(1)>=x0)&(poly[...,0].min(1)<x1)&
            (poly[...,1].max(1)>=y0)&(poly[...,1].min(1)<y1))
    label=base['drivable_static'].copy();positive=np.zeros(shape,bool);junction=np.zeros(shape,bool)
    high=np.zeros(shape,bool);heading=np.zeros((2,*shape),np.float32);direction_valid=np.zeros(shape,bool)
    ambiguous=np.zeros(shape,bool)
    for i in np.flatnonzero(nearby):
        # OpenCV coordinates are (column=Y, row=X); coordinates refer to cell centers.
        xy=np.stack(((poly[i,:,1]-y0)/r-.5,(poly[i,:,0]-x0)/r-.5),-1)
        mask=np.zeros(shape,np.uint8)
        cv2.fillConvexPoly(mask,np.rint(xy*256).astype(np.int32),1,shift=8)
        inside=mask.astype(bool)
        # Drawing rasterization includes edge-touched cells. Supervision requires
        # the cell center inside the continuous polygon, not only pixel coverage.
        rows,cols=np.nonzero(inside);centers=np.stack((cols,rows),-1)
        edges=np.roll(xy,-1,axis=0)-xy
        offsets=centers[:,None,:]-xy[None]
        cross=edges[None,:,0]*offsets[:,:,1]-edges[None,:,1]*offsets[:,:,0]
        contained=(cross>=-1e-8).all(1)|(cross<=1e-8).all(1)
        inside[rows[~contained],cols[~contained]]=False
        if np.max(np.abs(poly[i,:,2]))>.75:
            high|=inside;continue
        d=direction[i,:2];d=d/max(np.linalg.norm(d),1e-8)
        conflict=positive&inside&((heading*d[:,None,None]).sum(0)<.8)
        ambiguous|=conflict
        new=inside&~positive;heading[:,new]=d[:,None]
        positive|=inside
        if ribbons['junction'][i]:junction|=inside
    # An upper road crossing the local layer is ambiguous; do not collapse levels.
    vertical=positive&high
    conflict=positive&(label==0)
    blocked=vertical|conflict
    label[positive]=1;label[blocked]=255
    valid=label!=255
    direction_valid=positive&~ambiguous&~high&~conflict
    heading[:,~direction_valid]=0
    result=dict(drivable_static=label,drivable_valid=valid,drivable_rule_valid=np.zeros(shape,bool),
                lane_direction=heading,lane_direction_valid=direction_valid,
                junction_support=junction&valid,vertical_ambiguity=vertical)
    validate_road_arrays(result)
    return result


def validate_road_arrays(data):
    validate_arrays(data['drivable_static'],data['drivable_valid'],data['drivable_rule_valid'])
    shape=data['drivable_static'].shape;direction=data['lane_direction'];valid=data['lane_direction_valid']
    if direction.shape!=(2,*shape) or direction.dtype!=np.float32 or not np.isfinite(direction).all():
        raise ValueError('Invalid lane direction')
    for key in ('lane_direction_valid','junction_support','vertical_ambiguity'):
        if data[key].shape!=shape or data[key].dtype!=np.bool_:raise ValueError('Invalid road mask')
    if np.any(valid&(~data['drivable_valid']|(data['drivable_static']!=1))):raise ValueError('Direction outside road')
    if valid.any() and not np.allclose(np.linalg.norm(direction[:,valid],axis=0),1,atol=1e-4):raise ValueError('Direction not unit')
    if np.any(direction[:,~valid]!=0):raise ValueError('Unknown direction not zero')
    if np.any(data['junction_support']&(~data['drivable_valid']|(data['drivable_static']!=1))):raise ValueError('Junction support outside road')
