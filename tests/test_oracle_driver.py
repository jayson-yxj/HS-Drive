import numpy as np
from simulator.oracle_driver import *


def state():return OracleState(0,0,0,2,1.)

def traj():return Trajectory(np.array([[i,0,0,2] for i in np.arange(1,9)],dtype=float),1.)


def test_trajectory_timeout_missing_invalid_speed_and_footprint():
    s=state();assert safety_check(None,s,[])=='no_trajectory'
    t=traj();t.timestamp_s=0.;assert safety_check(t,s,[])=='trajectory_timeout'
    t=traj();t.points[0,0]=np.nan;assert safety_check(t,s,[])=='invalid_trajectory'
    for dt in [0.,-1.,np.nan]:
        t=traj();t.dt_s=dt;assert safety_check(t,s,[])=='invalid_trajectory'
    t=traj();t.points[:,3]=10;assert safety_check(t,s,[])=='speed_limit'
    obstacle={'xy':[5,1.6],'yaw':0.,'half_length':1.,'half_width':.7}
    assert safety_check(traj(),s,[obstacle])=='obstacle_footprint' # centerline alone misses it
    assert safety_check(traj(),s,[],road_predicate=lambda x,y:abs(y)<1.)=='road_boundary'


def test_control_sign_limits_stop_timeout():
    c=PurePursuitPID();t=traj();t.points[:,1]=1;u=c.command(t,state());assert u['steer']<0 and 0<=u['throttle']<=.05
    u=c.command(t,state(),'emergency');assert u['brake']==1 and u['throttle']==0
    s=state();s.timestamp_s=2;assert c.command(t,s)['reason']=='controller_timeout'


def test_planner_contract_curvature_and_stop():
    route=np.c_[np.arange(50),np.zeros(50),np.zeros(50)];t,i=local_trajectory(route,state());assert t.points.shape==(8,4) and (t.points[:,3]<=4).all()
    stopped,_=local_trajectory(route,state(),stop_distance=2);assert (stopped.points[:,3]==0).all()
    assert polygons_intersect(footprint(0,0,0,2,1),footprint(1,0,.4,2,1))
    assert not polygons_intersect(footprint(0,0,0,2,1),footprint(9,0,.4,2,1))
