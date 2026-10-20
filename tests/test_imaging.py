from copy import deepcopy
import numpy as np
import pytest
from simulator.imaging import load_camera_config,apply_profile,rgb_attributes


def test_geometry_is_preserved_and_postprocess_explicit():
    base=load_camera_config('configs/cameras.yaml');original=deepcopy(base)
    changed=load_camera_config('configs/cameras.yaml','configs/imaging/training_960.yaml')
    assert base==original
    assert changed['cameras']==base['cameras'] and changed['distortion']==base['distortion']
    assert changed['horizontal_fov_deg']==base['horizontal_fov_deg']
    assert (changed['image_width'],changed['image_height'])==(960,576)
    assert rgb_attributes(changed)['enable_postprocess_effects']=='true'
    assert rgb_attributes(base)['enable_postprocess_effects']=='false'


@pytest.mark.parametrize('attrs',[{'fov':120},{'sensor_tick':.2},
    {'enable_postprocess_effects':'false'},{'gamma':float('nan')}])
def test_photometry_cannot_override_geometry_or_accept_bad_values(attrs):
    with pytest.raises(ValueError):rgb_attributes({'rgb_attributes':attrs})


def test_resized_intrinsics_preserve_half_pixel_geometry():
    from geometry.cameras import resize_rgb_and_intrinsics
    from geometry.reference import intrinsics_from_hfov,project_optical_points
    image=np.zeros((576,960,3),np.uint8);K=intrinsics_from_hfov(960,576,100)
    resized,new_K=resize_rgb_and_intrinsics(image,K,(640,384))
    np.testing.assert_allclose(new_K,intrinsics_from_hfov(640,384,100),atol=1e-12)
    points=np.array([[1,2,10],[0,0,4]])
    uv=project_optical_points(points,K)
    np.testing.assert_allclose(project_optical_points(points,new_K),(uv+.5)*2/3-.5)
    assert resized.shape==(384,640,3)
