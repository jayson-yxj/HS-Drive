import numpy as np
from diagnostics.m5e_metrics import selection


def test_all_low_predictions_cannot_pass_joint_selection():
    shape=(4,160,120);m=np.zeros(shape,bool);m[2,60,60]=True
    ground=np.zeros(shape,bool);ground[1,70,60]=True
    refs=[dict(valid=m|ground,target=m,actor_masks=m[None])]*2
    rows=[dict(sample_id='a',group='g'),dict(sample_id='b',group='g')]
    pair=[(dict(a='a',b='b',group='g'),dict(appeared=m,vanished=np.zeros_like(m)))]
    best,candidates=selection(np.zeros((2,*shape)),rows,refs,pair,dict(a=0,b=1),[.3,.5])
    assert not best['feasible'] and best['worst_violation']>=1


def test_ground_fpr_and_actor_recall_have_separate_constraints():
    shape=(4,160,120);m=np.zeros(shape,bool);m[2,60,60]=True;g=np.zeros(shape,bool);g[1,70,60]=True
    refs=[dict(valid=m|g,target=m,actor_masks=m[None])]*2
    rows=[dict(sample_id='a',group='g'),dict(sample_id='b',group='g')]
    pairs=[(dict(a='a',b='b',group='g'),dict(appeared=m,vanished=np.zeros_like(m)))]
    prob=np.zeros((2,*shape));prob[1,m]=.9;prob[:,g]=.9
    best,_=selection(prob,rows,refs,pairs,dict(a=0,b=1),[.5]);assert not best['feasible'];assert best['scenes']['g']['ground_fpr']==1
