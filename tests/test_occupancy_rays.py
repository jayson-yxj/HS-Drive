import numpy as np
from geometry.ray_tracing import VoxelGrid,trace_segment
from datasets.occupancy_labels import build_observed,sampled_rays,UNKNOWN,FREE,SURFACE_OCCUPIED,GROUND,STATIC_OBSTACLE

G=VoxelGrid((0,6),(-2,2),(0,1,3),1.)

def label(orig,end,**kw):return build_observed(np.array(orig,float),np.array(end,float),G,**kw)[0]


def test_wall_free_prefix_unknown_behind_and_ground_semantics():
    d=label([[.5,.5,1.5]],[[3.5,.5,1.5]],semantics=[STATIC_OBSTACLE]);s=d['occupancy_state']
    assert (s[1,:3,2]==FREE).all() and s[1,3,2]==SURFACE_OCCUPIED and (s[1,4:,2]==UNKNOWN).all()
    assert d['obstacle_mask'][1,3,2]
    d=label([[.5,.5,2.]],[[.5,.5,.2]],semantics=[GROUND])
    assert d['occupancy_state'][0,0,2]==SURFACE_OCCUPIED and not d['obstacle_mask'].any() and d['obstacle_valid'][0,0,2]


def test_two_cameras_conflict_occupied_precedence_and_no_carving_behind():
    d=label([[.5,.5,1.5],[.5,.5,1.5],[-1,.5,1.5]],[[3.5,.5,1.5],[5.5,.5,1.5],[3.5,.5,1.5]])
    assert d['surface_count'][1,3,2]==2 and d['ray_conflict_mask'][1,3,2]
    assert d['occupancy_state'][1,4,2]==UNKNOWN
    assert d['occupancy_state'][1,3,2]==SURFACE_OCCUPIED


def test_invalid_truncated_low_confidence_never_free():
    d=label([[.5,0,1.5]]*2,[[3.5,0,1.5],[np.nan,0,1]],reliable=[False,True]);assert not d['occupancy_valid'].any()
    origins,ends,sem,stats=sampled_rays(np.full((5,5),80.),np.eye(3),np.eye(4),stride=1);assert len(ends)==0


def test_outside_camera_boundary_and_nonuniform_height():
    ranks=trace_segment([-1,0,1],[7,0,1],G);assert len(ranks)==6
    for r in ranks:assert np.unravel_index(r,G.shape)[0]==1
    assert len(trace_segment([-1,2,1],[7,2,1],G))==0
    d=label([[-1,.5,1.5]],[[2.5,.5,1.5]]);assert d['occupancy_state'][1,0,2]==FREE
    assert G.locate([6,0,1])[0]==-1 and G.locate([0,0,3])[0]==-1


def test_self_occlusion_and_depth_boundary():
    d=label([[.5,.5,1.5]],[[5.5,.5,1.5]],ego_box=(np.array([1,0,1]),np.array([2,1,2])))
    assert not d['occupancy_valid'].any()
    dep=np.ones((3,3))*4;dep[1,1]=2
    assert sampled_rays(dep,np.eye(3),np.eye(4),stride=1)[3]['valid_depth_rays']==0


def test_optical_z_uses_actual_origin_and_range():
    k=np.eye(3);pose=np.eye(4);pose[:3,3]=[1,2,3]
    o,e,_,_=sampled_rays(np.full((3,3),2.),k,pose,stride=1)
    np.testing.assert_allclose(o[0],[1,2,3]);np.testing.assert_allclose(e[-1],[5,6,5])
    assert np.linalg.norm(e[-1]-o[-1])>2


def test_semantic_conflict_is_not_an_obstacle_negative():
    d=label([[.5,.5,1.5]]*2,[[3.5,.5,1.5]]*2,semantics=[GROUND,STATIC_OBSTACLE])
    assert d['occupancy_state'][1,3,2]==SURFACE_OCCUPIED
    assert d['semantic_conflict_mask'][1,3,2]
    assert not d['surface_semantic_valid'].any() and not d['obstacle_valid'].any()


def test_unreliable_free_ray_cannot_erase_reliable_surface():
    d=label([[.5,.5,1.5]]*2,[[3.5,.5,1.5],[5.5,.5,1.5]],reliable=[True,False])
    assert d['occupancy_state'][1,3,2]==SURFACE_OCCUPIED
    assert (d['occupancy_state'][1,4:,2]==UNKNOWN).all()
