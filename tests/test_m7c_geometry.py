from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('shapely')
from shapely.geometry import box
from diagnostics.m7c_geometry import GeometryCache,evaluate_cached
from diagnostics.m7a_planning import evaluate
from tests.test_m7a_planning import fixture
from tools.m7c_common import equal


@pytest.mark.parametrize('angle',[0,.03,.4,1.5])
def test_cached_rotation_hole_and_curb_exact(angle):
 r,c=fixture();p=np.array(c.points);p[:,2]+=angle;c=replace(c,points=tuple(map(tuple,p)))
 for region in [box(-5,-3,40,3),box(-5,-1,40,1),box(-5,-3,40,3).difference(box(5,-.1,5.2,.1))]:
  cache=GeometryCache('mapA');a=evaluate(r,c,region,1.)
  for _ in range(2):assert equal(a.__dict__,evaluate_cached(r,c,region,1.,cache=cache).__dict__)


def test_exact_border_and_query_inputs_not_quantized():
 cache=GeometryCache('A');calls=[]
 query=cache.exact([(1,-1)],0,lambda x,y:calls.append((x,y)) or x<=1.)
 assert query(1.,0.) and not query(1.+1e-12,0.)
 assert query(1.,0.) and len(calls)==2


def test_lane_map_z_and_body_cache_invalidation():
 cache=GeometryCache('A');q=cache.exact([(1,-1)],0,lambda x,y:True);assert q(0.,0.)
 assert not cache.exact([(1,-2)],0,lambda x,y:False)(0.,0.)
 assert not cache.exact([(1,-1)],1,lambda x,y:False)(0.,0.)
 other=GeometryCache('B');assert not other.exact([(1,-1)],0,lambda x,y:False)(0.,0.)
 r,c=fixture();a=cache.prepare(r,c,((1,-1),));b=cache.prepare(replace(r,scene=replace(r.scene,body_half_extent=(3.,1.))),c,((1,-1),))
 assert a is not b and a.body_half_extent!=b.body_half_extent
 assert a.trajectory_sha256==b.trajectory_sha256


def test_no_cached_age_or_dynamic_decision():
 from datasets.m7a_contracts import Box
 r,c=fixture();cache=GeometryCache('A');road=box(-5,-3,40,3)
 assert evaluate_cached(r,c,road,1.,cache=cache).planning_feasible
 assert not evaluate_cached(r,c,road,1.201,cache=cache).planning_feasible
 r=replace(r,scene=replace(r.scene,boxes=(Box('cross',(1.,-30.),0.,.1,.1,(0.,120.)),)))
 assert 'predicted_collision' in evaluate_cached(r,c,road,1.,cache=cache).reasons
 assert not evaluate_cached(r,None,road,1.,cache=cache).planning_feasible


def test_lru_bound_and_same_geometry_distinct_stamp():
 r,c=fixture();cache=GeometryCache('A',capacity=2)
 a=cache.prepare(r,c,((1,-1),));b=cache.prepare(r,replace(c,stamp=replace(c.stamp,frame_id=11)),((1,-1),))
 assert a.trajectory_sha256!=b.trajectory_sha256
 assert len(cache.sweeps)<=2


class Waypoint:
    lane_type='Driving'
    lane_width=3.5
    road_id=1
    lane_id=-1
    section_id=0
    is_junction=False
    def __init__(self,x,y=0,yaw=0,road=1,lane=-1):
        from types import SimpleNamespace as N
        self.road_id=road;self.lane_id=lane;self.s=x;self.successors=[]
        self.transform=N(location=N(x=x,y=-y),rotation=N(yaw=-np.degrees(yaw)))
    def next(self,step):return self.successors


def test_unique_connection_ambiguity_and_intent():
    from diagnostics.m7c_routes import continue_waypoints
    a,b,c,d=Waypoint(0),Waypoint(.5),Waypoint(1,road=2),Waypoint(1,.2,.1,road=3)
    a.successors=[b];b.successors=[c,d]
    r=continue_waypoints(a,required_length=1)
    assert r.stop_reason=='AmbiguousSuccessors' and r.length==.5
    r=continue_waypoints(a,required_length=1,intent={(1,-1):(2,-1)})
    assert r.status=='Supported' and r.transitions[-1]['selected']==(2,-1)
    r=continue_waypoints(a,required_length=1,intent={(1,-1):(4,-1)})
    assert r.stop_reason=='IllegalOrAmbiguousIntent'


def test_dead_end_reverse_and_discontinuity_unknown():
    from diagnostics.m7c_routes import continue_waypoints
    for c in [Waypoint(-.1),Waypoint(2),Waypoint(1,yaw=.4)]:
        a,b=Waypoint(0),Waypoint(.5);a.successors=[b];b.successors=[c]
        r=continue_waypoints(a,required_length=5)
        assert r.stop_reason=='ConnectionDiscontinuityOrReverse'
    a,b=Waypoint(0),Waypoint(.5);a.successors=[b]
    assert continue_waypoints(a,required_length=5).stop_reason=='NoLegalSuccessor'


def test_analytic_slow_integral_and_constraints():
    from diagnostics.m7c_slow import slow_candidate
    r,_=fixture();r=replace(r,scene=replace(r.scene,ego_xy_yaw_speed=(0.,0.,0.,4.)))
    c=slow_candidate(r);p=np.array(c.points);dt=.1
    assert p[0,3]==4 and p[-1,3]==.5
    assert np.max(abs(np.diff(p[:,0])-(p[:-1,3]+p[1:,3])*dt/2))<.001
    a=np.diff(p[:,3])/dt;j=np.diff(a)/dt
    assert min(a)>=-1.8 and max(a)<=1.5 and max(abs(j))<=2.5


def test_random_boundary_kinematic_and_collision_equivalence():
    from datasets.m7a_contracts import Box
    rng=np.random.default_rng(746)
    r,c=fixture()
    cache=GeometryCache('random')
    for _ in range(30):
        y=float(rng.uniform(-1.5,1.5));p=np.array(c.points);p[:,1]=y
        q=replace(r,scene=replace(r.scene,ego_xy_yaw_speed=(0.,y,0.,2.),boxes=(Box('ob',(float(rng.uniform(0,20)),float(rng.uniform(-4,4))),float(rng.uniform(-3,3)),.3,.4,(0.,float(rng.uniform(-3,3)))),)))
        cc=replace(c,points=tuple(map(tuple,p)))
        road=box(-5,-3,35,3).difference(box(6,-.1,6.1,.1)) if rng.random()<.5 else box(-5,-3,35,3)
        old=evaluate(q,cc,road,1.)
        assert equal(old.__dict__,evaluate_cached(q,cc,road,1.,cache=cache).__dict__)
        assert equal(old.__dict__,evaluate_cached(q,cc,road,1.,cache=cache).__dict__)
