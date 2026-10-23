from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('shapely')
from shapely.geometry import box
from diagnostics.m7b_motion import ArcRoute,ObservedMotion,longitudinal,build_hypotheses,evaluate_dynamic_hypotheses,stop_witness
from datasets.m7a_contracts import Box,Stamp,WORLD,OracleSceneState,PlanningRequest,CandidateTrajectory,Unknown,reject_control_export
from diagnostics.m7a_planning import evaluate
from tools.m7b_common import read


def fixture(v=2.5,d=14.,lead=2.):
    route=ArcRoute([(float(x),0.,0.) for x in np.arange(0,100.1,.5)],[-1]*201)
    stamp=Stamp(1,100.,100.2,WORLD)
    scene=OracleSceneState(stamp,(0.,0.,0.,v),(Box('lead',(d,0.),0.,1.9,.8,(lead,0.)),))
    req=PlanningRequest(scene,tuple(map(tuple,route.points)),'unit',(100.,0.,0.))
    t=np.arange(61)*.1;c=CandidateTrajectory(stamp,'A_keep_route',tuple((v*q,0.,0.,v) for q in t),tuple(t),req.route)
    return req,c,ObservedMotion('lead',route,d)


def test_curve_arc_not_initial_tangent_and_lane_contract():
    th=np.linspace(0,.8,161);r=ArcRoute(np.c_[50*np.sin(th),50*(1-np.cos(th)),th],[-2]*len(th))
    p=r.at(30);assert p[1]>8 and p[0]<30 and r.lane_id(30)==-2
    assert np.allclose(r.tangent(30),[np.cos(p[2]),np.sin(p[2])])
    with pytest.raises(ValueError):r.at(50)
    with pytest.raises(ValueError):ArcRoute([(0,0,0),(0,0,0)])


def test_restart_is_stationary_then_finite_acceleration():
    t=np.arange(0,6.001,.025);s,v=longitudinal(t,0.,1.,2.)
    assert np.all(s[t<=2]==0) and np.all(v[t<=2]==0)
    assert max(np.diff(v)/.025)<=1.00000001
    assert np.all(np.diff(s)>=0)


def test_braking_stops_without_reverse_or_teleport():
    t=np.arange(0,6.001,.025);s,v=longitudinal(t,2.,-3.,.5)
    assert v[0]==2 and min(v)==0 and np.all(np.diff(s)>=-1e-9)
    assert abs(s[-1]-(1+2/3))<1e-8


def test_CV_miss_and_general_braking_detection():
    r,c,o=fixture();cfg=read('configs/m7b/protocol.json')
    assert evaluate(r,c,box(-5,-3,105,3),100).planning_feasible
    hs,unknown=build_hypotheses(r.scene,[o],cfg)
    result=evaluate_dynamic_hypotheses(c,r.scene,hs,unknown)
    assert result['status']=='CollisionUnderHypothesis'
    assert any('braking' in row['hypothesis'] for row in result['collisions'])


def test_unknown_route_end_not_free():
    r,c,o=fixture(d=95.)
    hs,unknown=build_hypotheses(r.scene,[o],read('configs/m7b/protocol.json'))
    assert unknown and evaluate_dynamic_hypotheses(c,r.scene,hs,unknown)['status']=='PredictionOutOfSupport'
    assert not Unknown('route_end').safety_free


def test_missing_actor_and_wrong_observed_velocity_rejected():
    r,c,o=fixture();cfg=read('configs/m7b/protocol.json')
    hs,u=build_hypotheses(r.scene,[],cfg)
    assert not evaluate_dynamic_hypotheses(c,r.scene,hs,u)['robust']
    r=replace(r,scene=replace(r.scene,boxes=(replace(r.scene.boxes[0],velocity=(0.,2.)),)))
    hs,u=build_hypotheses(r.scene,[o],cfg)
    assert u[0]['reason']=='snapshot_path_mismatch'


def test_legal_stop_and_infeasible_stop_remain_distinct():
    r,c,o=fixture(v=3,d=25);stop=stop_witness(r,3.)
    e=evaluate(r,replace(stop,name='D_stop'),box(-5,-3,105,3),100)
    assert e.planning_feasible and stop.points[-1][3]==0
    too_fast=stop_witness(r,1.)
    e=evaluate(r,replace(too_fast,name='D_stop'),box(-5,-3,105,3),100)
    assert 'acceleration_limit' in e.reasons and 'internal_jerk_limit' in e.reasons
    with pytest.raises(TypeError):reject_control_export(stop)


def test_hypotheses_only_depend_on_current_snapshot_and_map():
    import inspect
    assert 'reference' not in inspect.signature(build_hypotheses).parameters
    assert 'reference' not in inspect.signature(evaluate_dynamic_hypotheses).parameters
    r,c,o=fixture();cfg=read('configs/m7b/protocol.json')
    a=build_hypotheses(r.scene,[o],cfg);b=build_hypotheses(r.scene,[o],cfg)
    assert a==b


def test_stopped_actor_acceleration_not_mistaken_for_static_geometry():
    r,c,o=fixture(lead=0)
    hs,u=build_hypotheses(r.scene,[o],read('configs/m7b/protocol.json'))
    assert any(h.acceleration>0 for h in hs)
    static=replace(o,mode='static_geometry')
    hs,u=build_hypotheses(r.scene,[static],read('configs/m7b/protocol.json'))
    assert len(hs)==1 and hs[0].poses[-1][:2]==hs[0].poses[0][:2]


def test_batch_SAT_matches_scalar_for_rotation_and_high_speed():
    from diagnostics.m7b_motion import sat_intervals_vectorized
    from simulator.m47_motion import continuous_rectangles
    rng=np.random.default_rng(73);n=500
    e0=rng.normal(size=(n,2))*4;o0=rng.normal(size=(n,2))*4
    e1=e0+rng.normal(size=(n,2))*10;o1=o0+rng.normal(size=(n,2))*10
    ha=rng.uniform(.02,3,(n,2));hb=rng.uniform(.02,3,(n,2));a=rng.uniform(-3,3,n);b=rng.uniform(-3,3,n)
    expected=[continuous_rectangles(e0[i],o0[i],e1[i]-e0[i],o1[i]-o0[i],ha[i],hb[i],a[i],b[i],1) for i in range(n)]
    assert np.array_equal(sat_intervals_vectorized(e0,e1,o0,o1,ha,hb,a,b),expected)


def test_dynamic_checker_rejects_bad_timestamp_and_short_horizon():
    r,c,o=fixture();hs,u=build_hypotheses(r.scene,[o],read('configs/m7b/protocol.json'))
    bad=replace(c,stamp=replace(c.stamp,frame_id=2))
    assert evaluate_dynamic_hypotheses(bad,r.scene,hs,u)['status']=='InvalidInput'
    bad=replace(c,points=c.points[:5],times_s=c.times_s[:5])
    assert not evaluate_dynamic_hypotheses(bad,r.scene,hs,u)['robust']
