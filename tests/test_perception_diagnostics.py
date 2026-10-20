import numpy as np
import pytest

torch=pytest.importorskip('torch')
from diagnostics.distributions import distribution_metrics,GroupedMetrics,js_divergence
from diagnostics.spatial import project,compare_mass,spatial_probability_metrics
from diagnostics.experiment import input_variant,replace_encoder
from models.view_transform import HeightSplat


def metric(p,gt=10.,idx=1):
    return distribution_metrics(torch.tensor([p]),torch.tensor([2.,10.,18.]),torch.tensor([gt]),torch.tensor([idx]),torch.tensor([0.,4.,16.,20.]))


def test_delta_and_uniform_distribution():
    d=metric([0.,1.,0.])
    for k in ('expected_mae_m','mode_mae_m','median_mae_m','normalized_entropy','distribution_expected_absolute_error_m'):assert d[k][0]==pytest.approx(0)
    assert d['gt_bin_probability'][0]==1 and d['mass_within_1m'][0]==1
    u=metric([1/3]*3);assert u['normalized_entropy'][0]==pytest.approx(1)
    assert u['expected_mae_m'][0]==pytest.approx(0,abs=1e-6)
    assert u['distribution_expected_absolute_error_m'][0]==pytest.approx(16/3)


def test_bimodal_mean_is_not_distribution_accuracy_and_bias_sign():
    d=metric([.5,0.,.5]);assert d['expected_mae_m'][0]==0 and d['median_depth_m'][0]==2
    assert d['distribution_expected_absolute_error_m'][0]==8 and d['separated_two_peak_fraction'][0]==1
    d=metric([1.,0.,0.]);assert d['expected_signed_bias_m'][0]==-8
    assert d['shortfall_contribution_m'][0]==8 and d['overshoot_contribution_m'][0]==0
    with pytest.raises(ValueError):metric([.2,.2,.2])


def test_center_mass_and_within_bin_assumption_are_explicitly_different():
    d=metric([0.,1.,0.]);assert d['mass_within_1m'][0]==1
    assert d['mass_within_1m_uniform_bin_assumption'][0]==pytest.approx(1/6)


def test_mask_band_and_camera_counts_and_empty_groups():
    a=GroupedMetrics();v={'gt_depth_m':np.array([5.,10.,70.]),'expected_mae_m':np.array([1.,2.,3.])}
    a.update(v,camera=np.array(['front','front','rear']),scene='s',motion='moving',in_bev=np.array([True,True,False]))
    r=a.result();assert r['all_valid']['all']['1.5-10m']['count']==1
    assert r['gt_in_bev_3d']['all']['60-80m']['count']==0
    assert r['gt_in_bev_3d']['all']['60-80m']['expected_mae_m'] is None
    assert r['all_valid']['camera/rear']['all']['expected_mae_m']==3
    assert r['all_valid']['all']['all']['expected_mae_m']==2


def test_spatial_oracle_weighted_mass_and_half_open_bounds():
    cfg={'resolution_m':1.,'x_range_m':[0,2],'y_range_m':[0,2],'z_edges_m':[0,1,2],
         'network':{'depth_bins':2,'depth_range_m':[1.,3.]}}
    splat=HeightSplat(cfg);ctx=torch.tensor([[2.,4.],[6.,8.],[99.,99.]])
    pts=torch.tensor([[[.2,.2,.2]],[[.2,.2,.2]],[[2.,.2,.2]]]);weights=torch.ones(3,1)
    h,m=project(ctx,weights,pts,splat)
    assert m.sum()==2;torch.testing.assert_close(h[0,:,0,0,0],torch.tensor([4.,6.]))
    assert compare_mass(m,m)['normalized_spatial_tv']==0
    assert compare_mass(m,m)['mass_fraction_on_gt_support']==1
    w=torch.tensor([[.25],[.75],[1.]])
    h,m=project(ctx,w,pts,splat);assert m.sum()==1;torch.testing.assert_close(h[0,:,0,0,0],torch.tensor([5.,7.]))


def test_intervention_preserves_other_cameras_and_hook_removed_on_exception():
    x=torch.rand(1,6,3,32,64);original=x.clone();y=input_variant(x,'front_local_mask')
    assert torch.equal(x,original) and torch.equal(y[:,1:],x[:,1:]) and not torch.equal(y[:,0],x[:,0])
    model=torch.nn.Module();model.encoder=torch.nn.Identity()
    try:
        with replace_encoder(model,torch.zeros_like(x)):
            assert not model.encoder(x).any();raise RuntimeError('test')
    except RuntimeError:pass
    assert torch.equal(model.encoder(x),x)
    p=torch.tensor([[.3,.7]]);assert js_divergence(p,p).item()==0


def test_out_of_grid_rank_alias_never_counts_as_gt_voxel():
    cfg={'resolution_m':1.,'x_range_m':[0,2],'y_range_m':[0,2],'z_edges_m':[0,1,2],
         'network':{'depth_bins':2,'depth_range_m':[1.,3.]}}
    splat=HeightSplat(cfg)
    # Out-of-grid y=2 has flattened rank 2, aliasing valid x=1,y=0.
    gt=torch.tensor([[.2,2.2,.2]]);points=torch.tensor([[[1.2,.2,.2]]]);rank,inside=splat.ranks(gt)
    values=spatial_probability_metrics(torch.ones(1,1),points,gt,rank,inside,splat)
    assert not inside.any() and values['mass_on_exact_gt_voxel'][0]==0


def test_normalized_single_ray_feature_cancels_probability_but_mass_retained():
    cfg={'resolution_m':1.,'x_range_m':[0,2],'y_range_m':[0,2],'z_edges_m':[0,1,2],
         'network':{'depth_bins':2,'depth_range_m':[1.,3.]}}
    splat=HeightSplat(cfg);context=torch.tensor([[2.,4.]])
    points=torch.tensor([[[.2,.2,.2],[1.2,.2,.2]]])
    h1,m1=project(context,torch.tensor([[.01,.99]]),points,splat)
    h2,m2=project(context,torch.tensor([[.99,.01]]),points,splat)
    torch.testing.assert_close(h1,h2)
    assert not torch.equal(m1,m2)
