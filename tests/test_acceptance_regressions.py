"""Regression tests for strict temporal identity and empty geometry results."""
import numpy as np
from geometry.consistency import compare_depths,summarize
CFG={'min_depth_m':2,'max_depth_m':40,'max_patch_spread_m':.2,'agreement_tolerance_m':.5}
from geometry.reference import intrinsics_from_hfov


def test_no_fov_overlap_is_not_a_passing_measurement():
    d=np.full((60,80),10,dtype=np.float32);K=intrinsics_from_hfov(80,60,100)
    T=np.eye(4);T[0,3]=1000
    r=compare_depths(d,d,K,K,T,CFG)
    assert r['projected']==0
    assert summarize(r['errors_m'])['count']==0
