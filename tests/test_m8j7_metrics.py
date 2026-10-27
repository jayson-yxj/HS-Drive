import numpy as np
import pytest
from training.m8j7_metrics import match, summarize, average_precision, errors


def record(x=(0.,),scores=(.8,),labels=(0,),gx=(0.,),gl=(0,)):
    def boxes(xs):
        a=np.zeros((len(xs),9));a[:,0]=xs;a[:,3:6]=[2,4,2];return a
    return dict(boxes=boxes(x),scores=np.asarray(scores),labels=np.asarray(labels),
                gt=boxes(gx),gt_labels=np.asarray(gl))


def test_duplicate_and_wrong_class_remain_false_positives():
    r=record((0.,.1,0.),(.9,.8,.7),(0,0,1))
    m=match(**r)
    assert m['pairs']==[(0,0)] and m['fp']==[1,2] and m['duplicates']==[1]
    s=summarize([r],['car','pedestrian'])
    assert (s['TP'],s['FP'],s['FN'])==(1,2,0)
    assert s['per_class']['pedestrian']['recall'] is None


def test_threshold_distance_and_unmatched_are_not_hidden():
    r=record((0.,2.,4.),(.19,.8,.9),(0,0,0))
    assert match(**r)['pairs']==[]
    assert len(match(**r)['fp'])==2
    assert match(**r,threshold=.1)['pairs']==[(0,0)]


def test_one_to_one_score_priority():
    r=record((.3,.1),(.9,.8),(0,0))
    assert match(**r)['pairs']==[(0,0)]


def test_xy_matching_z_and_dimensions_diagnosed_separately():
    r=record();r['boxes'][0,2]=10;r['boxes'][0,3]=3
    s=summarize([r],['car'])
    assert s['TP']==1 and s['matched_errors']['z_center_m']['mean']==10
    assert s['matched_errors']['dimension_abs_mean_m']['mean']==pytest.approx(1/3)


def test_no_gt_class_na_but_false_positive_counted():
    r=record(gx=(),gl=())
    a=average_precision([r],['car'])
    assert a['per_class']['car']['AP'] is None
    assert summarize([r],['car'])['FP']==1


def test_perfect_ap_and_miss():
    r=record();assert average_precision([r],['car'])['per_class']['car']['AP']==1.
    r['boxes'][0,0]=10
    assert average_precision([r],['car'])['per_class']['car']['AP']==0.


def test_predictions_filtered_without_gt_and_outside_counted():
    r=record((52.,0.),(.9,.8),(0,0))
    s=summarize([r],['car'])
    assert s['outside_task_range_predictions']==1 and s['decoded_predictions']==2
    assert s['TP']==1


def test_two_frames_not_cross_matched():
    a=record(x=(10.,));b=record(x=(0.,),gx=(10.,))
    s=summarize([a,b],['car'])
    assert s['TP']==0 and s['FP']==2 and s['FN']==2


def test_barrier_half_turn_and_cone_yaw_not_applicable():
    r=record((0.,10.),(.9,.8),(0,1),(0.,10.),(0,1));r['boxes'][:,6]=np.pi
    e=errors(r['boxes'],r['gt'],[(0,0),(1,1)],r['gt_labels'],['barrier','traffic_cone'])
    assert e['yaw_rad']==[0.]


def test_ap_false_positive_before_true_positive():
    r=record((10.,0.),(.9,.8),(0,0))
    assert average_precision([r],['car'])['per_class']['car']['AP']==pytest.approx(.5)


def test_recall_denominator_contains_every_gt():
    r=record(gx=(0.,10.,20.),gl=(0,0,0))
    s=summarize([r],['car'])
    assert s['GT']==3 and s['FN']==2 and s['recall']==pytest.approx(1/3)


def test_empty_predictions_stable():
    r=record(x=(),scores=(),labels=())
    s=summarize([r],['car'])
    assert s['TP']==0 and s['FN']==1 and s['precision'] is None
    assert s['matched_errors']['xy_m']['mean'] is None
