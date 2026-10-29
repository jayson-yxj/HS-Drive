import numpy as np
import pytest
from pyquaternion import Quaternion
from datasets.m8j11_contract import to_legacy_boxes,from_legacy_boxes,matrix,check_sequence,INPUT_FIELDS,CAMERAS

def test_roundtrip_and_size_yaw():
    b=np.array([[1,2,-1,4,2,1.5,.7,3,-2]],dtype=np.float32)
    q=to_legacy_boxes(b)
    np.testing.assert_allclose(from_legacy_boxes(q),b,atol=1e-6)
    assert q[0,3:6].tolist()==[2,4,1.5]
    np.testing.assert_allclose(Quaternion(axis=[0,0,1],angle=-float(q[0,6])-np.pi/2).rotation_matrix,
                               Quaternion(axis=[0,0,1],angle=float(b[0,6])).rotation_matrix,atol=1e-6)
    np.testing.assert_array_equal(q[:,[0,1,2,5,7,8]],b[:,[0,1,2,5,7,8]])

def test_transform_composition():
    a=matrix(Quaternion(axis=[0,0,1],angle=.5).elements,[3,4,5]);b=matrix([1,0,0,0],[1,2,3])
    p=np.array([10,20,30,1.]);np.testing.assert_allclose(np.linalg.inv(a@b)@a@b@p,p)

def sequence():
    f=[dict(sample_token='a',scene_token='s',timestamp_us=10),dict(sample_token='b',scene_token='s',timestamp_us=20)]
    t={'sample':{'a':dict(timestamp=10,scene_token='s',prev='',next='b'),'b':dict(timestamp=20,scene_token='s',prev='a',next='')}}
    return f,t

def test_full_sequence():
    f,t=sequence();check_sequence(f,t)

@pytest.mark.parametrize('kind',['cold_drop','future_order','duplicate','bad_time'])
def test_sequence_rejects(kind):
    f,t=sequence()
    if kind=='cold_drop':f=f[1:]
    elif kind=='future_order':f.reverse()
    elif kind=='duplicate':f[1]=f[0].copy()
    else:f[1]['timestamp_us']=0
    with pytest.raises(AssertionError):check_sequence(f,t)

def test_input_contract_no_gt():
    assert not any(s in ' '.join(INPUT_FIELDS) for s in ('gt','label','points','radar'))
    assert len(CAMERAS)==6 and len(set(CAMERAS))==6

def test_bad_pred_rejected():
    with pytest.raises(AssertionError):to_legacy_boxes(np.zeros((3,7)))
    a=np.zeros((1,9));a[0,0]=np.nan
    with pytest.raises(AssertionError):to_legacy_boxes(a)
