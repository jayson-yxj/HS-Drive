import torch
from training.m5g_losses import foreground_loss,stratified_depth_loss
from datasets.labels import depth_bin_edges


def test_unknown_hard_negative_cannot_receive_gradient():
    for mode in ['bce','hard','weighted']:
        logits=torch.tensor([0.,2.,8.],requires_grad=True);loss=foreground_loss(logits,torch.tensor([True,False,False]),torch.tensor([True,True,False]),mode,torch.ones(3,dtype=torch.bool));loss.backward();assert logits.grad[0]<0 and logits.grad[1]>0 and logits.grad[2]==0


def test_ce_spatial_excludes_unknown_and_unsupported_bins():
    edges=torch.tensor(depth_bin_edges(),dtype=torch.float32);logits=torch.zeros(1,48,1,2,requires_grad=True);dep=torch.tensor([[[20.,30.]]]);v=torch.tensor([[[True,False]]]);target=torch.ones_like(v);sp=torch.zeros_like(logits,dtype=torch.bool);sp[:,10,:,0]=True
    loss=stratified_depth_loss(logits,dep,v,target,edges,sp);loss.backward();assert torch.isfinite(loss)and logits.grad[:,:,:,1].abs().sum()==0


def test_old_balanced_bce_equivalence():
    from training.m5f_loss import pixel_loss
    x=torch.tensor([.2,-.5,1.]);t=torch.tensor([True,False,False]);v=torch.ones_like(t)
    assert torch.equal(foreground_loss(x,t,v),pixel_loss(x,t,v))


def test_dense_oracle_matches_half_open_grid_and_sparse_drops_support():
    import numpy as np
    from geometry.ray_tracing import VoxelGrid
    grid=VoxelGrid();points=np.array([[.1,.1,0],[1.1,.1,0],[60.,0,0]])
    ids,inside=grid.locate(points);assert inside.tolist()==[True,True,False]
    dense=set(ids[inside]);sparse=set(ids[:1]);assert len(dense)==2 and sparse<dense


def test_spatial_loss_zero_support_keeps_classification():
    edges=torch.tensor(depth_bin_edges(),dtype=torch.float32);x=torch.zeros(1,48,1,1);d=torch.tensor([[[30.]]]);m=torch.ones_like(d,dtype=torch.bool)
    a=stratified_depth_loss(x,d,m,m,edges);b=stratified_depth_loss(x,d,m,m,edges,torch.zeros_like(x,dtype=torch.bool));assert torch.equal(a,b)


def test_locked_evaluation_never_selects_new_workpoint():
    import numpy as np
    from tools.m5g_evaluation import summarize
    rows=[dict(group='g',configuration=i,camera_names=['front'],weather='fixed',actors=[dict(actors=[])])for i in [0,1,2]]
    labels=[dict(target=np.zeros((1,2,2),bool),valid=np.ones((1,2,2),bool),classes=np.zeros((1,2,2),int),actor=np.zeros((1,2,2),int),radial=np.ones((1,2,2)))for _ in rows]
    probs=[np.zeros((1,2,2),float)for _ in rows]
    r=summarize(rows,labels,probs,.7);assert r['threshold']==.7 and r['selection']is None and r['low_fpr_validation_workpoint']is None
