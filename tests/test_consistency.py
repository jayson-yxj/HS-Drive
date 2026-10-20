import numpy as np
from geometry.consistency import compare_depths,summarize
from geometry.reference import intrinsics_from_hfov

CFG={'min_depth_m':2,'max_depth_m':40,'max_patch_spread_m':.2,'agreement_tolerance_m':.5}


def test_equal_plane_and_occlusion_count():
    K=intrinsics_from_hfov(80,60,100);d=np.full((60,80),10,dtype=np.float32)
    r=compare_depths(d,d,K,K,np.eye(4),CFG)
    assert r['comparable']>100 and np.max(abs(r['errors_m']))<1e-5
    near=d/2
    r=compare_depths(d,near,K,K,np.eye(4),CFG)
    assert r['occluded']>100 and r['comparable']==0


def test_forward_motion_plane_alignment_and_wrong_pose():
    K=intrinsics_from_hfov(80,60,100);d=np.full((60,80),10,dtype=np.float32)
    T=np.eye(4);T[2,3]=-2
    r=compare_depths(d,d-2,K,K,T,CFG)
    assert summarize(r['errors_m'])['p95_abs_m']<1e-5
    wrong=compare_depths(d-2,d,K,K,np.eye(4),CFG)
    assert summarize(wrong['errors_m'])['median_abs_m']==2


def test_depth_step_is_excluded_not_counted_as_correct_surface():
    K=intrinsics_from_hfov(80,60,100);d=np.full((60,80),10,dtype=np.float32)
    target=d.copy();target[:,40:]=20
    r=compare_depths(d,target,K,K,np.eye(4),CFG,stride=1)
    assert r['smooth_valid']<r['projected']
    assert summarize(r['errors_m'])['p95_abs_m']>1
