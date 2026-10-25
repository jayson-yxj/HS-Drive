"""Evaluator-only object taxonomy and oriented geometry, never inference features."""
import numpy as np
from geometry.reference import transform_points

SEMANTIC = {0:'NONE',1:'Roads',2:'Sidewalks',3:'Buildings',4:'Walls',5:'Fences',6:'Poles',
7:'TrafficLight',8:'TrafficSigns',9:'Vegetation',10:'Terrain',11:'Sky',12:'Pedestrians',13:'Rider',
14:'Car',15:'Truck',16:'Bus',17:'Train',18:'Motorcycle',19:'Bicycle',20:'Static',21:'Dynamic',
22:'Other',23:'Water',24:'RoadLines',25:'Ground',26:'Bridge',27:'RailTrack',28:'GuardRail'}
DIRECT={12:'pedestrian',14:'car',15:'truck',16:'bus',18:'motorcycle',19:'bicycle'}
IGNORE={1,2,3,6,7,8,9,10,11,23,24,25,26,27}

def require_source_frame(sidecar):
    """Reject fallback actor transforms or asynchronous sensor evidence."""
    frame, timestamp = sidecar['frame_id'], sidecar['timestamp_s']
    if not np.isfinite(timestamp):
        raise ValueError('Nonfinite source timestamp')
    for actor in sidecar['objects']:
        if (not actor.get('snapshot_present', False) or actor['frame_id'] != frame
                or not np.isfinite(actor['timestamp_s'])
                or abs(actor['timestamp_s'] - timestamp) >= 1e-5):
            raise ValueError('Actor snapshot missing or unsynchronized')
    if len(sidecar['sensors']) != 6:
        raise ValueError('Six camera evidence required')
    for camera in sidecar['sensors'].values():
        if set(camera) != {'rgb', 'depth', 'instance', 'semantic'}:
            raise ValueError('Four synchronized modalities required')
        for sensor in camera.values():
            if (sensor['frame_id'] != frame or not np.isfinite(sensor['timestamp_s'])
                    or abs(sensor['timestamp_s'] - timestamp) >= 1e-5):
                raise ValueError('Unsynchronized sensor evidence')

def taxonomy(tags,name):
    n=name.lower()
    # Semantic alone cannot distinguish cone/barrier. Explicit asset identity required.
    if any(x in n for x in ['trafficcone','constructioncone']):return 'traffic_cone','AssetIdentity'
    if 'streetbarrier' in n:return 'barrier','AssetIdentity'
    values={DIRECT[t] for t in tags if t in DIRECT}
    if len(values)==1:return next(iter(values)),'SemanticClass'
    if values:return 'Unknown','ConflictingSemanticClasses'
    if tags and set(tags).issubset(IGNORE):return 'Ignore','KnownNonTargetSemantic'
    return 'Unknown','SemanticInsufficientForPETRClass'

def coarse(label):
    if label in ['car','truck','bus','trailer','construction_vehicle','motorcycle','bicycle']:return 'vehicle'
    return {'pedestrian':'person','traffic_cone':'cone','barrier':'barrier'}.get(label,'unknown')

def points_inside(points,T_world_from_box,extent,tolerance=.25):
    local=transform_points(points,np.linalg.inv(np.asarray(T_world_from_box)))
    return (np.abs(local)<=np.asarray(extent)+tolerance).all(axis=1)

def render_grade(views):
    valid={v.get('camera', str(i)) for i,v in enumerate(views) if v['accepted']}
    if any(v.get('identity_conflict',False) for v in views):return 'AmbiguousIdentity'
    return 'RenderLinkedMultiView' if len(valid)>=2 else 'RenderLinkedSingleView' if valid else 'UnlabeledOrUnknown'

def prediction_grade(views,has_proxy):
    if any(v.get('competing_bbox_iou',0)>=.1 for v in views):return 'AmbiguousIdentity'
    valid={v.get('camera', str(i)) for i,v in enumerate(views) if v['source_linked'] and v['retained_pixels']>=8 and
           v['instance_bbox_iou']>=.5 and v['mask_recall']>=.8}
    if len(valid)>=2:return 'RenderLinkedMultiView'
    if len(valid)==1:return 'RenderLinkedSingleView'
    return 'ProjectionProxyOnly' if has_proxy else 'UnlabeledOrUnknown'
