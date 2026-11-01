"""Explicit route-coordinate boundary; official planner and PID stay unchanged."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def load_planner(source,carla_map):
    import carla
    spec=importlib.util.spec_from_file_location('j13_original_route_planner',Path(source))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    origin=carla_map.transform_to_geolocation(carla.Location())
    return module.RoutePlanner(4.,50.,lat_ref=origin.latitude,lon_ref=origin.longitude)


def set_navigation_route(planner,route,*,coordinate_type,client_version,carla_map=None):
    """Convert GPS exactly once, then use the original set_route/run_step pair.

    World routes use CARLA Transform-like objects. GPS routes use lat/lon/alt
    dictionaries. Reject ambiguous inputs instead of guessing from values.
    0.9.15 delegates to the unchanged legacy GPS entry and configured reference.
    """
    if coordinate_type not in ('world_carla','gps'):raise ValueError('ExplicitRouteCoordinateTypeRequired')
    if client_version not in ('0.9.15','0.9.16'):raise ValueError('UnsupportedNavigationVersion')
    if not route:raise ValueError('EmptyNavigationRoute')
    converted=[]
    for point,command in route:
        if coordinate_type=='world_carla':
            if not hasattr(point,'location'):raise ValueError('WorldRouteRequiresLocation')
            xy=[point.location.x,point.location.y]
        else:
            if not isinstance(point,dict) or not {'lat','lon'}<=set(point):raise ValueError('GPSRouteRequiresLatLon')
            xy=[point['lat'],point['lon'],point.get('alt',0.)]
        if not np.isfinite(xy).all():raise ValueError('NonfiniteNavigationRoute')
    if coordinate_type=='world_carla':
        planner.set_route(route,gps=False);return 'WorldRouteUnchanged'
    if client_version=='0.9.15':
        planner.set_route(route,gps=True);return 'LegacyGPS0915'
    if carla_map is None or not callable(getattr(carla_map,'geolocation_to_transform',None)):
        raise ValueError('NativeMapInverseRequired')
    import carla
    for point,command in route:
        geo=carla.GeoLocation(latitude=float(point['lat']),longitude=float(point['lon']),altitude=float(point.get('alt',0.)))
        location=carla_map.geolocation_to_transform(geo)
        if not np.isfinite([location.x,location.y]).all():raise ValueError('NonfiniteNativeInverse')
        converted.append((SimpleNamespace(location=location),command))
    planner.set_route(converted,gps=False)
    return 'NativeMapGPS0916'


def geometry_checks(carla_map,source,version,anchors=((0.,0.),(37.,-81.))):
    """Exercise the actual planner consumer with finite, synthetic geometry."""
    import carla
    from datasets.m8j12_contract import native_target,measured_navigation_target
    cases=[]
    for ax,ay in anchors:
        for yaw in (0.,90.,-90.,180.,37.):
            tf=carla.Transform(carla.Location(x=ax,y=ay),carla.Rotation(yaw=yaw))
            for distance in (2.,10.,35.):
                for x,y in ((distance,0),(-distance,0),(0,-distance),(0,distance)):
                    p=tf.transform(carla.Location(x=x,y=y));geo=carla_map.transform_to_geolocation(p)
                    planner=load_planner(source,carla_map);command=SimpleNamespace(value=4)
                    mode=set_navigation_route(planner,[({'lat':geo.latitude,'lon':geo.longitude,'alt':geo.altitude},command)],coordinate_type='gps',client_version=version,carla_map=carla_map)
                    selected,cmd=planner.run_step(np.array([ax,ay]))
                    error=float(np.max(np.abs(selected-[p.x,p.y])))
                    assert error<.001 and cmd is command,(version,error)
                    expected=np.array([y,x]);native=native_target(tf.get_matrix(),selected)
                    row=dict(ego_world_matrix=tf.get_matrix(),target_world_xy=selected,imu=[0]*6+[(np.radians(yaw)+np.pi/2)%(2*np.pi)])
                    imu=measured_navigation_target(row)
                    assert np.allclose(native,expected,atol=.001) and np.allclose(imu,expected,atol=.001)
                    world=load_planner(source,carla_map)
                    set_navigation_route(world,[(SimpleNamespace(location=p),command)],coordinate_type='world_carla',client_version=version,carla_map=carla_map)
                    w,_=world.run_step(np.array([ax,ay]));assert np.allclose(w,selected,atol=.001)
                    cases.append(error)
    return dict(status='Passed',cases=len(cases),max_consumer_error_m=max(cases),version=version,mode=mode,
        world_route='Passed',gps_route_consumer='Passed',synthetic_compass='Passed',map_reference='NativeMap',model_forwards=0)
