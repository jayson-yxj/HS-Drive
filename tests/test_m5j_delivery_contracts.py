"""Additional malformed input tests, no scientific model or score changes."""
import pytest,torch
from tests.test_m5j_evidence import packet
from models.m5j_candidate import validate_packet,CAMERAS

@pytest.mark.parametrize('field',['timestamps_s','images','intrinsics','cam_to_ego'])
def test_nonfinite_rejected(field):
    x=packet();x[field].reshape(-1)[0]=float('nan')
    with pytest.raises(ValueError):validate_packet(x,CAMERAS)

def test_missing_camera_and_bad_extrinsic_rejected():
    x=packet();x['valid_cameras'][0,0]=False
    with pytest.raises(ValueError):validate_packet(x,CAMERAS)
    x=packet();x['cam_to_ego'][0,0,0,0]=2
    with pytest.raises(ValueError):validate_packet(x,CAMERAS)
