"""Static lane support labels, separate from occupancy and current traffic legality.

255 is unknown. Positive = CARLA Driving lane interior. Negative = explicitly
mapped sidewalk / shoulder / biking / border / median interior. Junctions,
map gaps, height ambiguity and lane boundaries are unknown, never negative.
"""
import numpy as np

SCHEMA = 'hsd-drivable-static-v1'
NEGATIVE_LANES = {'Sidewalk', 'Shoulder', 'Biking', 'Border', 'Median'}


def classify_lane(lane_type, *, junction, height_error, edge_clearance,
                  height_tolerance=.75, boundary_margin=.5):
    if junction or abs(height_error)>height_tolerance or edge_clearance<boundary_margin:
        return 255
    if lane_type=='Driving':return 1
    if lane_type in NEGATIVE_LANES:return 0
    return 255


def grid_centers(config):
    r=config['resolution_m'];xr=config['x_range_m'];yr=config['y_range_m']
    x=np.arange(xr[0]+r/2,xr[1],r);y=np.arange(yr[0]+r/2,yr[1],r)
    xx,yy=np.meshgrid(x,y,indexing='ij')
    return np.stack((xx,yy,np.zeros_like(xx),np.ones_like(xx)),-1)


def validate_arrays(label,valid,rule_valid,shape=(160,120)):
    if label.shape!=tuple(shape) or valid.shape!=label.shape or rule_valid.shape!=label.shape:
        raise ValueError('Drivable shape mismatch')
    if label.dtype!=np.uint8 or valid.dtype!=np.bool_ or rule_valid.dtype!=np.bool_:
        raise ValueError('Drivable dtype mismatch')
    if not np.isin(label,[0,1,255]).all() or not np.array_equal(valid,label!=255):
        raise ValueError('Unknown must be masked, not negative')
    if rule_valid.any():raise ValueError('Static lane labels cannot assert current traffic legality')


def rasterize(carla_map,world_from_ego,config):
    import carla  # offline label generation only
    points=grid_centers(config)
    # Raw T is CARLA world from CARLA ego: HS ego Y-left -> CARLA Y-right.
    points[...,1]*=-1
    world=points@np.asarray(world_from_ego,dtype=np.float64).T
    label=np.full(world.shape[:2],255,dtype=np.uint8)
    for index in np.ndindex(label.shape):
        x,y,z=world[index][:3]
        wp=carla_map.get_waypoint(carla.Location(x=float(x),y=float(y),z=float(z)),
                                  project_to_road=False,lane_type=carla.LaneType.Any)
        if wp is None:continue
        center=wp.transform.location;right=wp.transform.get_right_vector()
        lateral=abs((x-center.x)*right.x+(y-center.y)*right.y+(z-center.z)*right.z)
        label[index]=classify_lane(str(wp.lane_type),junction=wp.is_junction,
            height_error=z-center.z,edge_clearance=wp.lane_width/2-lateral)
    valid=label!=255;rule_valid=np.zeros_like(valid)
    validate_arrays(label,valid,rule_valid,label.shape)
    return dict(drivable_static=label,drivable_valid=valid,drivable_rule_valid=rule_valid)
