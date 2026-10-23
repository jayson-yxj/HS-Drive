"""Synthetic contract tests; never claim CARLA execution from these."""
import copy
from dataclasses import replace, fields
import pytest
from datasets.m7d_temporal import from_packet, result_clock, TemporalPlanningInput
from datasets.m7a_contracts import Box


def packet():
    return dict(episode_id='synthetic',frame=100,sim_time=5.,input_available_wall_s=1000.,source='Privileged_ActualCARLAActorSnapshot',valid=True,missing=False,
        ego=dict(xy=[0.,0.],z=0.,yaw=0.,signed_speed=1.),body=dict(half_extent=[2.,1.],offset=[0.,0.]),
        actors=[dict(actor_id=1,xy=[15.,0.],yaw=0.,half_length=2.,half_width=1.,velocity=[1.,0.],frame=100,timestamp=5.,category='Vehicle',blueprint='a',role='lead')],
        route=[[0.,0.,0.],[100.,0.,0.]],history_stamps=[[99,4.95]])


def test_no_future_or_actor_reference_fields():
    assert not {'future','future_path','actor','episode_path'} & {f.name for f in fields(TemporalPlanningInput)}
    p=packet();p['future_reference']='evaluator.json'
    with pytest.raises(ValueError,match='forbidden'):from_packet(p)


@pytest.mark.parametrize('fault',['missing','bad_time','duplicate_id','actor_stamp','nan','reverse','history_future','identity','jump','time_frame'])
def test_rejected_faults(fault):
    p=packet();old=from_packet(p);old.validate();p=copy.deepcopy(p)
    if fault=='missing':p['missing']=True
    elif fault=='bad_time':p['sim_time']=4.9;p['actors'][0]['timestamp']=4.9
    elif fault=='duplicate_id':p['actors']*=2
    elif fault=='actor_stamp':p['actors'][0]['frame']=101
    elif fault=='nan':p['ego']['yaw']=float('nan')
    elif fault=='reverse':p['ego']['signed_speed']=-.2
    elif fault=='history_future':p['history_stamps']=[[101,5.05]]
    else:
        p['frame']=104;p['sim_time']=5.2;p['actors'][0].update(frame=104,timestamp=5.2)
        if fault=='identity':p['actors'][0]['blueprint']='changed'
        if fault=='jump':p['actors'][0]['xy']=[150.,0.]
        if fault=='time_frame':p['frame']=105;p['actors'][0]['frame']=105
    with pytest.raises(ValueError):from_packet(p).validate(old if fault in ['bad_time','identity','jump','time_frame'] else None)


def test_clock_never_restamps_and_expires():
    inp=from_packet(packet());r=result_clock(inp,1e6,1e6+.501,None)
    assert r.expired_for_online and r.input_timestamp_s==5. and r.input_available_wall_s==1000.
    assert r.modeled_result_sim_time_s==pytest.approx(5.501)
    assert r.result_available_wall_s==1e6+.501
    assert not result_clock(inp,10,10.1,'D_stop').expired_for_online
    assert result_clock(inp,10,10.1,'D_stop',.11).expired_for_online


def test_no_feasible_is_not_reuse():
    inp=from_packet(packet());r=result_clock(inp,1,1.1,None)
    assert r.selected is None and r.source.endswith('NotControl')


def test_optimization_exact_and_snapshot_scoped():
    pytest.importorskip('shapely')
    from diagnostics.m7d_geometry import SnapshotPolygons,stopping_reserve_reused
    from diagnostics.m7c_slow import slow_candidate,stopping_reserve
    from diagnostics.m7b_motion import ArcRoute,ObservedMotion,build_hypotheses
    import json
    req=from_packet(packet()).request();cfg=json.load(open('configs/m7b/protocol.json'));route=ArcRoute(req.route)
    for distance in [5.,15.,45.]:
        b=replace(req.scene.boxes[0],xy=(distance,0.));scene=replace(req.scene,boxes=(b,));r=replace(req,scene=scene)
        hs,u=build_hypotheses(scene,[ObservedMotion('1',route,distance)],cfg);assert not u
        c=slow_candidate(r);polys=SnapshotPolygons(scene,hs)
        assert stopping_reserve_reused(c,scene,hs,route,polys)==stopping_reserve(c,scene,hs,route)
        with pytest.raises(ValueError,match='snapshot_identity'):stopping_reserve_reused(c,replace(scene),hs,route,polys)


def test_missing_actor_hypothesis_is_unknown():
    from diagnostics.m7b_motion import build_hypotheses,evaluate_dynamic_hypotheses
    from diagnostics.m7a_planning import generate
    import json
    req=from_packet(packet()).request();hs,u=build_hypotheses(req.scene,[],json.load(open('configs/m7b/protocol.json')))
    assert u[0]['reason']=='missing_motion_observation'
    assert not evaluate_dynamic_hypotheses(generate(req)[0],req.scene,hs,u)['robust']


def test_map_interruption_rejects():
    from diagnostics.m7c_geometry import evaluate_cached,GeometryCache
    from diagnostics.m7a_planning import generate
    from shapely.geometry import box
    req=from_packet(packet()).request();e=evaluate_cached(req,generate(req)[-1],box(-100,-100,200,100),5.,cache=GeometryCache('synthetic'),exact_road=lambda x,y:False)
    assert not e.planning_feasible and 'body_outside_allowed_road' in e.reasons


def test_expired_candidate_cannot_be_normal_control():
    from diagnostics.m7c_geometry import evaluate_cached,GeometryCache
    from diagnostics.m7a_planning import generate
    from shapely.geometry import box
    req=from_packet(packet()).request();e=evaluate_cached(req,generate(req)[0],box(-100,-100,200,100),5.201,cache=GeometryCache('synthetic'))
    assert not e.planning_feasible and 'stale_or_future' in e.reasons
