"""SDK coordinate cross-check; real server snapshots separately audited offline."""
import math
import numpy as np
import pytest
carla=pytest.importorskip('carla')
from simulator.m6d_state import signed_from_measurement

@pytest.mark.parametrize('yaw,pitch,vel',[(35,8,(1,-2,.1)),(-90,-2,(1,2,0)),(180,0,(1,0,0))])
def test_carla_forward_vector_xy(yaw,pitch,vel):
    tf=carla.Transform(carla.Location(2,3,.4),carla.Rotation(yaw=yaw,pitch=pitch))
    f=tf.get_forward_vector();axis=np.array([f.x,f.y]);axis/=np.linalg.norm(axis)
    raw=dict(velocity=vel,rotation=[0,pitch,yaw],position=[2,3,.4],applied=dict(gear=1,reverse=False,hand_brake=True,brake=0),t=10,frame=3)
    s=signed_from_measurement(raw,[[0,0,0],[20,0,0]],1,.01)
    assert s.v_forward_mps==pytest.approx(np.dot(vel[:2],axis),abs=1e-6)
    assert s.v_lateral_mps==pytest.approx(np.dot(vel[:2],[axis[1],-axis[0]]),abs=1e-6)
