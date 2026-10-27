import numpy as np
from training.m8j8a_cache_diagnostics import (exact_curve, match_record, operating_points,
    select_cases, summarize_transitions, evidence, sigmoid)


def record(scores=(.9,.8,.1), xs=(0.,.2,8.)):
    gt=np.array([[0.,0.,0.,1.,1.,2.,0.,0.,0.],[8.,0.,0.,1.,1.,2.,0.,0.,0.]])
    boxes=np.repeat(gt[:1],len(xs),axis=0);boxes[:,0]=xs
    return dict(boxes=boxes,scores=np.array(scores),labels=np.zeros(len(xs),dtype=int),
                gt=gt,gt_labels=np.array([0,0]))


def test_exact_curve_matches_original_at_all_scores_and_ties():
    a=record((.9,.8,.8));b=record((.7,.7,.1))
    curve=exact_curve([a,b])
    for point in curve:
        mm=[match_record(r,point['score_threshold']) for r in (a,b)]
        assert point['TP']==sum(len(m['pairs']) for m in mm)
        assert point['FP']==sum(len(m['fp']) for m in mm)
        assert point['duplicates']==sum(len(m['duplicates']) for m in mm)
    assert len(curve)==4
    assert next(p for p in curve if p['score_threshold']==.8)['TP']==2


def test_curve_does_not_include_outside_range_or_raw_candidates():
    r=record(xs=(0.,51.2,8.));r['raw_cls']=np.ones((900,10))*10
    assert exact_curve([r])[-1]['TP']==2
    assert exact_curve([r])[-1]['FP']==0


def test_unreachable_recall_is_not_extrapolated():
    curve=exact_curve([record((.9,), (0.,))])
    p=operating_points(curve,dict(TP=2,precision=1.,recall=1.))
    assert p['recall_target_reachable'] is False
    assert p['recall_first_reaching'] is None
    assert p['maximum_cache_recall']['recall']==.5


def test_operating_point_uses_observed_breakpoint_not_interpolation():
    curve=exact_curve([record()]);ref=dict(TP=2,precision=.7,recall=1.)
    p=operating_points(curve,ref)
    assert p['recall_first_reaching'] in curve
    assert p['nearest_observed_precision'] in curve


def row(instance,frame,state,scene='a',category='car',delta=0.):
    return dict(instance_token=instance,frame=frame,state=state,scene=scene,class_name=category,
                timestamp_us=frame,sample_token=str(frame),xy_delta_m=delta)


def test_transition_records_not_independent_instances():
    r=summarize_transitions([row('x',0,'A_only'),row('x',1,'common'),row('y',2,'B_only')])
    assert r['total']['records']==3 and r['total']['unique_instances']==2
    assert sum(s['unique_instances'] for s in r['total']['states'].values())==3


def test_selection_globally_deduplicates_and_obeys_quotas():
    cfg=dict(existing_visual_frames=[0],bucket_order=['A_only','B_only','common_xy_worse'],
        budgets=dict(A_only=2,B_only=1,common_xy_worse=1),scenes=['a','b'],
        maximum_unique_instances=4,common_xy_worse_min_delta_m=.5)
    rows=[row('x',0,'A_only'),row('x',1,'A_only'),row('y',2,'A_only','b'),
          row('x',3,'B_only'),row('z',4,'B_only'),row('k',5,'common',delta=.6),
          row('j',6,'common',delta=.4)]
    selected=select_cases(rows,cfg)
    assert [r['instance_token'] for r in selected]==['x','y','z','k']
    assert selected[0]['frame']==0


def raw_record(score=.15,nearest_x=0.,retain=True):
    r=record((score,), (nearest_x,))
    raw=np.zeros((900,10));raw[:,:2]=100.;raw[0,0:2]=[nearest_x,0.];raw[:,4]=1.
    logits=np.full((900,10),-8.);logits[0,0]=np.log(score/(1-score))
    r.update(raw_bbox=raw,raw_cls=logits,query_indices=np.array([0]),flat_indices=np.array([0]))
    if not retain:
        r.update(boxes=np.empty((0,9)),scores=np.empty(0),labels=np.empty(0,dtype=int),
                 query_indices=np.empty(0,dtype=int),flat_indices=np.empty(0,dtype=int))
    return r


def test_below_threshold_is_direct_score_evidence():
    r=raw_record();e=evidence(r,0,match_record(r))
    assert 'NearbyCorrectClassDecodedScoresBelow0.2' in e['flags']
    assert 'AllNearbyValidCorrectClassRawScoresBelow0.2' in e['flags']
    assert e['matched'] is None


def test_geometry_failure_not_invented_category_change():
    r=raw_record(.8,2.01);e=evidence(r,0,match_record(r))
    assert 'NoRawQueryCenterWithin2m' in e['flags']
    assert e['best_correct_class_nearby'] is None


def test_topk_exclusion_distinguished_from_score_and_range():
    r=raw_record(.8,0.,False);r['raw_cls'][1:301,0]=4.
    e=evidence(r,0,match_record(r))
    assert e['strong_nearby_top300_exclusion_count']==1
    assert 'NearbyCorrectClassScoreAtLeast0.2ExcludedFromTop300' in e['flags']


def test_argmax_is_not_an_exclusive_label_filter():
    r=raw_record(.8);r['raw_cls'][0,1]=4.
    e=evidence(r,0,match_record(r))
    assert e['matched'] is not None
    assert e['best_correct_class_nearby']['argmax_class']==1


def test_sigmoid_finite_at_extreme_logits():
    assert np.isfinite(sigmoid(np.array([-10000.,0.,10000.]))).all()
    np.testing.assert_array_equal(sigmoid(np.array([-10000.,0.,10000.])),[0.,.5,1.])
