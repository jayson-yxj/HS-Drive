import numpy as np
from tools.analyze_imaging import metrics


def test_surface_mask_does_not_count_white_sky_as_saturation():
    image=np.full((10,10,3),128,np.uint8);image[:5]=255
    mask=np.zeros((10,10),bool);mask[5:]=True
    assert metrics(image)['near_white_fraction']==.5
    assert metrics(image,mask)['near_white_fraction']==0


def test_gate_rejects_bad_side_camera_even_if_front_is_good():
    from tools.validate_imaging_gate import evaluate_photometry
    limits={'max_surface_near_black_fraction':.05,'max_surface_near_white_fraction':.05,'max_static_rgb_mae_8bit':2}
    good={'depth_valid_surface':{'near_black_fraction':.01,'near_white_fraction':0},'static_temporal_rgb_mae_8bit':.5}
    bad={'depth_valid_surface':{'near_black_fraction':.06,'near_white_fraction':0},'static_temporal_rgb_mae_8bit':.5}
    from datasets.schema import CAMERA_IDS
    cameras = {name:good for name in CAMERA_IDS}
    cameras['rear_right'] = bad
    assert evaluate_photometry(cameras,limits)==['rear_right/near_black_fraction']
    assert 'camera_set' in evaluate_photometry({'front':good},limits)
    cameras['rear_right'] = {**good, 'static_temporal_rgb_mae_8bit':float('nan')}
    assert evaluate_photometry(cameras,limits)==['rear_right/static_temporal_rgb_mae_8bit']
