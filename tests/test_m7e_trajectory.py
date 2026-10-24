import json
import numpy as np
import pytest
from dataclasses import replace
from tests.test_m7d_temporal import packet
from datasets.m7d_temporal import from_packet
from diagnostics.m7e_trajectory import state_anchored_slow,InitialStateConnectionInfeasible
from diagnostics.m7e_region import StaticRegionCache
from diagnostics.m7c_geometry import GeometryCache,evaluate_cached
from shapely.geometry import box,Polygon


def config():return json.load(open('configs/m7e/protocol.json'))


@pytest.mark.parametrize('y,yaw,v',[(0,0,0),(.3,.05,2),(-.3,-.05,2),(.7,-.1,1),(0,.15,3),(0,0,5.6)])
def test_actual_start_and_integral(y,yaw,v):
    p=packet();p['ego'].update(xy=[5.,y],yaw=yaw,signed_speed=v);r=from_packet(p).request()
    c,info=state_anchored_slow(r,config());pts=np.array(c.points)
    assert np.allclose(pts[0],r.scene.ego_xy_yaw_speed,rtol=0,atol=1e-12)
    assert c.stamp==r.scene.stamp and c.times_s[-1]==6
    ds=np.linalg.norm(np.diff(pts[:,:2],axis=0),axis=1)
    assert np.max(abs(ds-(pts[:-1,3]+pts[1:,3])*.05))<.025
    assert info['initial_acceleration_observed'] is False
    empty=replace(r,scene=replace(r.scene,boxes=()))
    ev=evaluate_cached(empty,c,box(-10,-5,110,5),5,cache=GeometryCache('synthetic'))
    assert ev.planning_feasible,ev.reasons


@pytest.mark.parametrize('y,yaw',[(1.,0),(0,.4)])
def test_declared_initial_support(y,yaw):
    p=packet();p['ego'].update(xy=[5,y],yaw=yaw)
    with pytest.raises(InitialStateConnectionInfeasible):state_anchored_slow(from_packet(p).request(),config())


def test_no_short_route_snap():
    p=packet();p['route']=[[0.,0.,0.],[5.,0.,0.]]
    with pytest.raises(InitialStateConnectionInfeasible,match='insufficient'):state_anchored_slow(from_packet(p).request(),config())


def test_cache_full_identity_and_hole():
    c=StaticRegionCache();key=(1,-1,0);p={key:Polygon([(0,0),(20,0),(20,20),(0,20)],holes=[[(8,8),(12,8),(12,12),(8,12)]])}
    a=c.region('mapA',p,{key},.03);assert not a.covers(box(9,9,10,10))
    assert c.region('mapA',p,{key},.03) is a and c.hits==1
    for m,q,ins in [('mapB',p,.03),('mapA',p,.04),('mapA',{key:box(0,0,20,20)},.03)]:c.region(m,q,{key},ins)
    assert c.misses==4
    with pytest.raises(ValueError):c.region('mapA',p,{(1,-2,0)},.03)


def test_curved_route_anchor_and_inherited_road_rejection():
    p=packet();s=np.arange(0,80,.5);k=.02
    p['route']=np.c_[np.sin(k*s)/k,(1-np.cos(k*s))/k,k*s].tolist()
    p['ego'].update(xy=[0.,.25],yaw=.03,signed_speed=2.)
    req=from_packet(p).request();c,info=state_anchored_slow(req,config());pts=np.array(c.points)
    assert np.allclose(pts[0],req.scene.ego_xy_yaw_speed,atol=1e-12,rtol=0)
    assert info['max_connection_curvature']<=.10
    result=evaluate_cached(req,c,box(-.1,-.1,.1,.1),5,cache=GeometryCache('narrow'))
    assert not result.planning_feasible and 'body_outside_allowed_road' in result.reasons


def test_excessive_curvature_rejected_without_relaxation():
    p=packet();s=np.arange(0,30,.1);k=.2
    p['route']=np.c_[np.sin(k*s)/k,(1-np.cos(k*s))/k,k*s].tolist()
    with pytest.raises(InitialStateConnectionInfeasible,match='curvature'):state_anchored_slow(from_packet(p).request(),config())


def test_closed_route_direction_and_stamp_unchanged():
    p=packet();p['ego'].update(xy=[5.,-.2],yaw=-.03,signed_speed=2.)
    req=from_packet(p).request();c,_=state_anchored_slow(req,config())
    assert c.stamp.timestamp_s==5 and c.stamp.valid_until_s==5.2
    result=evaluate_cached(req,c,box(-10,-10,110,10),5.201,cache=GeometryCache('expired'))
    assert not result.planning_feasible and result.reasons==('stale_or_future',)
