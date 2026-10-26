import json
import hashlib
import cv2
import numpy as np
import pytest
from datasets.m8j3_rgb_clip import CAMERAS, load_rgb_packet


def fixture(root):
    root.mkdir()
    cameras = []
    for name in CAMERAS:
        p = root / (name+'.jpg')
        cv2.imwrite(str(p), np.full((4, 8, 3), [10, 30, 80], np.uint8))
        cameras.append(dict(name=name, path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            width=8, height=4, K=[[4,0,4],[0,4,2],[0,0,1]], T_hs_ego_from_optical=np.eye(4).tolist()))
    packet = dict(schema='m8j3-rgb-only-v1', source='SyntheticTest', revision='fixture',
        clip='fixture', ordinal='00000', source_frame=None, timestamp_s=None,
        temporal_valid=False, camera_order=list(CAMERAS), cameras=cameras)
    p = root/'packet.json'; p.write_text(json.dumps(packet))
    return p, packet


def test_rgb_only_and_color_order(tmp_path):
    p, packet = fixture(tmp_path/'rgb')
    out = load_rgb_packet(p.parent, p.name)
    assert out['images'].shape == (6,4,8,3)
    assert out['images'][0,0,0,0] > out['images'][0,0,0,2]
    assert out['timestamps'] is None and out['temporal_valid'] is False
    assert set(out) == {'images','intrinsics','cam_to_ego','timestamps','temporal_valid','camera_order'}


@pytest.mark.parametrize('field', ['GT', 'depth', 'bounding_boxes', 'world2cam', 'actor_state'])
def test_no_supervision_fields(tmp_path, field):
    p, packet = fixture(tmp_path/'rgb')
    packet[field] = [42]
    p.write_text(json.dumps(packet))
    with pytest.raises(ValueError): load_rgb_packet(p.parent, p.name)


def test_traversal_symlink_checksum_and_false_time(tmp_path):
    p, original = fixture(tmp_path/'rgb')
    other=tmp_path/'secret.jpg'; other.write_bytes(b'secret')
    (p.parent/'escape.jpg').symlink_to(other)
    for bad in ['../secret.jpg', 'escape.jpg', str(other)]:
        packet=json.loads(json.dumps(original));packet['cameras'][0]['path']=bad
        p.write_text(json.dumps(packet))
        with pytest.raises(ValueError):load_rgb_packet(p.parent,p.name)
    packet=json.loads(json.dumps(original));packet['cameras'][0]['sha256']='0'*64
    p.write_text(json.dumps(packet))
    with pytest.raises(ValueError):load_rgb_packet(p.parent,p.name)
    packet=json.loads(json.dumps(original));packet['timestamp_s']=0.0
    p.write_text(json.dumps(packet))
    with pytest.raises(ValueError):load_rgb_packet(p.parent,p.name)


def test_camera_permutation_and_calibration(tmp_path):
    p, packet=fixture(tmp_path/'rgb')
    packet['cameras'].reverse();p.write_text(json.dumps(packet))
    with pytest.raises(ValueError):load_rgb_packet(p.parent,p.name)
    packet['cameras'].reverse();packet['cameras'][0]['T_hs_ego_from_optical'][0][0]=-1
    p.write_text(json.dumps(packet))
    with pytest.raises(ValueError):load_rgb_packet(p.parent,p.name)


def test_basis_contract():
    from geometry.reference import camera_mount_to_hs_from_optical
    t=camera_mount_to_hs_from_optical(np.eye(4))
    assert np.allclose(t[:3,:3] @ [0,0,1], [1,0,0])
    assert np.allclose(t[:3,:3] @ [1,0,0], [0,-1,0])
    assert np.allclose(t[:3,:3] @ [0,1,0], [0,0,-1])
