import numpy as np
from datasets.labels import sample_depth,depth_bin_edges


def test_depth_boundaries_never_make_intermediate_surfaces():
    depth=np.full((24,48),5.,np.float32);depth[:,24:]=50
    result=sample_depth(depth,output_wh=(32,16),stride=1)
    assert set(np.unique(result['depth_m'])) <= {0.,5.,50.}
    assert (~result['depth_valid']).any()
    # Exact returned rays map back to native integer pixel centers.
    uv=result['depth_uv'];native=(uv+.5)*1.5-.5
    np.testing.assert_allclose(native,np.round(native),atol=1e-5)


def test_out_of_range_nan_and_far_plane_are_ignored_not_clipped():
    for value in [0.,1.49,80.,1000.,np.nan,np.inf]:
        result=sample_depth(np.full((16,16),value),output_wh=(16,16),stride=16)
        assert not result['depth_valid'].any()
        assert (result['depth_bin']==-100).all()
    for value,expected in [(1.5,0),(79.999,47)]:
        result=sample_depth(np.full((16,16),value),output_wh=(16,16),stride=16)
        assert result['depth_valid'].all() and result['depth_bin'][0,0]==expected
    edges=depth_bin_edges()
    assert len(edges)==49 and edges[0]==1.5 and edges[-1]==80
    assert np.all(np.diff(np.diff(edges))>0)


def test_sampled_depth_ray_matches_native_unprojection():
    from geometry.reference import intrinsics_from_hfov
    from geometry.cameras import resize_rgb_and_intrinsics
    depth=np.full((24,48),12.,np.float32)
    original=intrinsics_from_hfov(48,24,100)
    _,small=resize_rgb_and_intrinsics(np.zeros((24,48,3),np.uint8),original,(32,16))
    result=sample_depth(depth,output_wh=(32,16),stride=4)
    uv=result['depth_uv'].reshape(-1,2)
    native=(uv+.5)*1.5-.5
    ray_small=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(small).T
    ray_native=np.c_[native,np.ones(len(native))]@np.linalg.inv(original).T
    np.testing.assert_allclose(ray_small*12,ray_native*12,atol=2e-6)
