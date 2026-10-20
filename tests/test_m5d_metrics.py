import numpy as np
from diagnostics.m5d_metrics import voxel_3d_matching,witness_metrics


def test_3d_height_and_missing_targets_not_discarded():
    p=np.zeros((4,160,120),bool);g=p.copy();g[2,60,60]=True
    r=voxel_3d_matching(p,g);assert r['gt_to_pred']['unmatched_1m']==1 and r['gt_to_pred']['mean_m'] is None
    p[1,60,60]=True;r=voxel_3d_matching(p,g);assert r['gt_to_pred']['unmatched_1m']==1 and r['gt_to_pred']['mean_m']==1.5


def test_wrong_direction_and_unknown_high_scores_separate():
    p=np.ones((4,160,120));q=p.copy();m=np.zeros_like(p,bool);m[2,60,60]=True;pair=dict(appeared=m,vanished=np.zeros_like(m));r=witness_metrics(p,q,pair,.5)[0]
    assert r['n']==1 and r['absence_fpr']==1 and r['presence_recall']==1 and r['both_correct']==0
