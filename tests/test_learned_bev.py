import pytest
import numpy as np
torch=pytest.importorskip('torch')
pytest.importorskip('torchvision')
from models.view_transform import HeightSplat,feature_uv
from models.depth_bev import DepthBEV
from training.losses import compute_losses,logits_at_target_uv
from datasets.labels import depth_bin_edges
from geometry.reference import points_to_height_bev
import yaml


def bev_config():return yaml.safe_load(open('configs/bev.yaml'))


def inputs(height=64,width=96):
    k=torch.tensor([[40.,0,(width-1)/2],[0,40.,(height-1)/2],[0,0,1.]]).expand(1,6,3,3).clone()
    e=torch.tensor([[0.,0,1,0],[-1,0,0,0],[0,-1,0,1.6],[0,0,0,1]]).expand(1,6,4,4).clone()
    return dict(images=torch.rand(1,6,3,height,width),intrinsics=k,cam_to_ego=e,
        ego_pose=torch.eye(4,dtype=torch.float64)[None],ego_state=torch.zeros(1,7),
        route_command=torch.full((1,),-1,dtype=torch.int64),timestamps_s=torch.zeros(1,dtype=torch.float64),
        valid_cameras=torch.ones(1,6,dtype=torch.bool))


def test_scatter_matches_numpy_oracle_counts_and_half_open_bounds():
    splat=HeightSplat(bev_config())
    points=np.array([[-20,-30,-2],[59.99,29.99,3.99],[60,0,0],[0,30,0],[0,0,4],
                     [0,0,-.5],[0,0,1],[0,0,2.5],[0,0,0]],np.float32)
    rank,valid=splat.ranks(torch.from_numpy(points))
    counts=torch.zeros(4*160*120).index_add_(0,rank[valid],torch.ones(int(valid.sum())))
    reference=points_to_height_bev(points,x_range_m=(-20,60),y_range_m=(-30,30),
                                   z_edges_m=(-2,-.5,1,2.5,4),resolution_m=.5)
    np.testing.assert_array_equal(counts.reshape(4,160,120).numpy(),reference)


def test_chunked_splat_conserves_probability_mass_and_gradients():
    # Split ray depths across cells: if all bins share one cell, softmax mass
    # is identically 1 and its true depth gradient is zero.
    torch.manual_seed(19)
    config=bev_config();config.update(x_range_m=[-100,100],y_range_m=[-100,100],z_edges_m=[-100,0,40,100],resolution_m=5)
    one=HeightSplat(config,chunk_points=3);two=HeightSplat(config,chunk_points=1000)
    context=torch.randn(1,1,2,2,2,requires_grad=True)
    logits=torch.randn(1,1,48,2,2,requires_grad=True);p=logits.softmax(2)
    k=torch.tensor([[10.,0,.5],[0,10.,.5],[0,0,1.]]).reshape(1,1,3,3)
    e=torch.eye(4).reshape(1,1,4,4)
    a,mass=one(context,p,k,e,(2,2));b,mass2=two(context,p,k,e,(2,2))
    torch.testing.assert_close(a,b);torch.testing.assert_close(mass,mass2)
    torch.testing.assert_close(mass.sum(),torch.tensor(4.))
    (a.square().sum()+mass.square().sum()).backward()
    assert torch.isfinite(context.grad).all() and context.grad.abs().sum()>0
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


def test_logits_sampled_at_feature_centers_preserve_values():
    logits=torch.randn(1,6,48,4,6)
    uv=feature_uv(4,6,(64,96),'cpu').expand(1,6,4,6,2)
    torch.testing.assert_close(logits_at_target_uv(logits,uv,(64,96)),logits,atol=2e-6,rtol=1e-5)


def test_full_network_shapes_label_boundary_and_backward():
    torch.set_num_threads(2);torch.manual_seed(7)
    model=DepthBEV(bev_config(),chunk_points=1024);x=inputs();out=model(**x)
    assert out['depth_logits'].shape==(1,6,48,4,6)
    assert out['height_bev'].shape==(1,64,4,160,120)
    assert out['bev_current'].shape==(1,128,160,120)
    uv=feature_uv(4,6,(64,96),'cpu').expand(1,6,4,6,2)
    depth=torch.full((1,6,4,6),10.)
    edges=torch.tensor(depth_bin_edges(),dtype=torch.float32)
    targets={'depth_m':depth,'depth_uv':uv,'depth_valid':torch.ones_like(depth,dtype=torch.bool),
             'depth_bin':torch.bucketize(depth,edges,right=True)-1}
    losses,_=compute_losses(model,out,x,targets);losses['total'].backward()
    for module in (model.encoder,model.depth_head,model.context_head,model.bev_encoder,model.surface_head):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
    with pytest.raises(TypeError):model(**x,depth_gt=depth)
    x['valid_cameras'][0,0]=False
    with pytest.raises(ValueError,match='Incomplete'):model(**x)


def test_model_prediction_does_not_use_ego_or_route_labels():
    torch.set_num_threads(2);model=DepthBEV(bev_config()).eval();x=inputs()
    with torch.no_grad():
        first=model(**x)['depth_logits']
        x['ego_state'].fill_(100);x['route_command'].fill_(5);x['ego_pose'][0,0,3]=999
        second=model(**x)['depth_logits']
    torch.testing.assert_close(first,second,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA comparison requires GPU')
def test_cuda_splat_agrees_with_cpu_reference_forward_backward():
    from training.runtime import prepare_cuda
    prepare_cuda();torch.manual_seed(11)
    cfg=bev_config();cfg.update(x_range_m=[-20,60],y_range_m=[-30,30],resolution_m=2)
    cpu=HeightSplat(cfg,chunk_points=24);gpu=HeightSplat(cfg,chunk_points=24).cuda()
    x=inputs();context=torch.randn(1,6,2,4,6,requires_grad=True)
    logits=torch.randn(1,6,48,4,6,requires_grad=True)
    cg=context.detach().cuda().requires_grad_();lg=logits.detach().cuda().requires_grad_()
    hc,mc=cpu(context,logits.softmax(2),x['intrinsics'],x['cam_to_ego'],(64,96))
    hg,mg=gpu(cg,lg.softmax(2),x['intrinsics'].cuda(),x['cam_to_ego'].cuda(),(64,96))
    torch.testing.assert_close(hg.cpu(),hc,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(mg.cpu(),mc,atol=2e-5,rtol=2e-5)
    (hc.square().mean()+mc.square().mean()).backward()
    (hg.square().mean()+mg.square().mean()).backward()
    torch.testing.assert_close(cg.grad.cpu(),context.grad,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(lg.grad.cpu(),logits.grad,atol=2e-5,rtol=2e-5)
