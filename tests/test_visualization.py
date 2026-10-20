import numpy as np
from geometry.visualization import overlay_boxes
from geometry.reference import intrinsics_from_hfov


def test_box_crossing_near_plane_clipped_without_nan():
    points=[[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-.5,2]]
    rgb=np.zeros((100,100,3),dtype=np.uint8)
    result=overlay_boxes(rgb,[{'corners_hs_ego_m':points}],intrinsics_from_hfov(100,100,100),np.eye(4))
    assert result.shape==rgb.shape
    assert result.sum()>0


def test_box_behind_camera_not_drawn():
    points=[[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-3,-2]]
    rgb=np.zeros((100,100,3),dtype=np.uint8)
    result=overlay_boxes(rgb,[{'corners_hs_ego_m':points}],intrinsics_from_hfov(100,100,100),np.eye(4))
    assert not result.any()
