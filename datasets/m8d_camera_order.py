"""Research-only camera-axis permutation; physical calibration is immutable."""
import copy
import hashlib
import time
import numpy as np
from datasets.m8b_petr_adapter import ORDER, PETRCarlaOfflineAdapter

PERMUTATION = [0, 2, 1, 5, 3, 4]
INVERSE = np.argsort(PERMUTATION).tolist()
CAMERA_FIELDS = {'camera_names', 'filename', 'rgb_sha256', 'K', 'T_cam_to_ego',
                 'intrinsics', 'extrinsics', 'lidar2img', 'img_shape', 'pad_shape',
                 'ori_shape', 'img_timestamp', 'camera_frame_ids'}


def tensor_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def permute(cpu, metadata, indices):
    if sorted(indices) != list(range(6)) or cpu.shape[1] != 6:
        raise ValueError('PermutationContractNoGo: not a six-camera permutation')
    result = copy.deepcopy(metadata)
    for key, value in metadata.items():
        if key in CAMERA_FIELDS:
            if len(value) != 6:
                raise ValueError('PermutationContractNoGo: camera field length ' + key)
            result[key] = [copy.deepcopy(value[i]) for i in indices]
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 6:
            raise ValueError('PermutationContractNoGo: unclassified camera metadata ' + key)
    return cpu[:, indices].copy(), result


def prepare(adapter, packet, variant):
    start = time.perf_counter()
    cpu, meta, audit, times = adapter.prepare(packet)
    cameras = packet['cameras']
    meta.update(camera_names=[c['name'] for c in cameras], filename=[c['path'] for c in cameras],
                rgb_sha256=[c['sha256'] for c in cameras], K=[c['K'] for c in cameras],
                T_cam_to_ego=[c['T_cam_to_ego'] for c in cameras],
                intrinsics=[np.asarray(k) for k in audit['intrinsic_aug']],
                extrinsics=[np.linalg.inv(c['T_cam_to_ego']).T for c in cameras],
                img_timestamp=[c['timestamp_s'] for c in cameras],
                camera_frame_ids=[c['frame_id'] for c in cameras],
                frame_id=packet['frame_id'], timestamp_s=packet['timestamp_s'], token=packet['token'])
    t = time.perf_counter()
    if variant == 'B':
        cpu, meta = permute(cpu, meta, PERMUTATION)
    elif variant != 'A':
        raise ValueError('Only the declared A and B are allowed')
    times['permutation_ms'] = 1000*(time.perf_counter()-t)
    times['preprocess_ms'] = 1000*(time.perf_counter()-start)
    return cpu, meta, audit, times


def check_pair(a, ma, b, mb):
    """Fail closed before model quality is examined. No GT is accessed."""
    assert ma['camera_names'] == ORDER, 'PermutationContractNoGo: A order'
    assert mb['camera_names'] == [ORDER[i] for i in PERMUTATION], 'PermutationContractNoGo: B order'
    restored, rm = permute(b, mb, INVERSE)
    assert np.array_equal(a, restored), 'PermutationContractNoGo: RGB tensor mismatch'
    assert set(ma) == set(mb)
    for key in ma:
        if key in CAMERA_FIELDS:
            for left, right in zip(ma[key], rm[key]):
                assert np.array_equal(left, right), 'PermutationContractNoGo: ' + key
        else:
            assert ma[key] == rm[key], 'PermutationContractNoGo: scalar ' + key
    records = []
    for i, name in enumerate(ORDER):
        j = mb['camera_names'].index(name)
        camera_points = np.array([[0,0,5,1], [1,-1,15,1], [-2,1,30,1], [1,2,-5,1]],float)
        ego = camera_points @ np.asarray(ma['T_cam_to_ego'][i]).T
        pa = ego @ np.asarray(ma['lidar2img'][i]).T
        pb = ego @ np.asarray(mb['lidar2img'][j]).T
        np.testing.assert_array_equal(pa, pb)
        pixel_residual = float(np.max(abs(pa[:,:2]/pa[:,2:3]-pb[:,:2]/pb[:,2:3])))
        intrinsic = np.asarray(ma['intrinsics'][i])
        expected = intrinsic @ np.asarray(ma['extrinsics'][i]).T
        np.testing.assert_array_equal(expected, ma['lidar2img'][i])
        records.append(dict(name=name, A_index=i, B_index=j, rgb_sha256=ma['rgb_sha256'][i],
              prepared_tensor_sha256=tensor_sha(a[:,i]), K=ma['K'][i], T_cam_to_ego=ma['T_cam_to_ego'][i],
              intrinsic_aug=intrinsic.tolist(), lidar2img=np.asarray(ma['lidar2img'][i]).tolist(),
              pixel_projection_residual=pixel_residual))
    return dict(status='PermutationContractPassed', frame_id=ma['frame_id'], timestamp_s=ma['timestamp_s'],
                token=ma['token'], inverse_exact=True, camera_fields=sorted(CAMERA_FIELDS), cameras=records,
                image_SHA_multiset_equal=sorted(ma['rgb_sha256'])==sorted(mb['rgb_sha256']))
