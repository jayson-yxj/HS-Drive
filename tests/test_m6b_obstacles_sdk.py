"""CARLA SDK pure geometry: offset, local rotation, tilt, timestamp preservation."""
from types import SimpleNamespace
import numpy as np
import pytest
carla=pytest.importorskip('carla')
from simulator.m6b_obstacles import snapshot_obstacle


def test_offset_rotation_tilt_obb_enclosure():
    bb=carla.BoundingBox(carla.Location(x=.4,y=.1,z=.8),carla.Vector3D(2,1,.7));bb.rotation=carla.Rotation(yaw=13)
    tf=carla.Transform(carla.Location(10,20,1),carla.Rotation(yaw=43,pitch=8,roll=4))
    state=SimpleNamespace(get_transform=lambda:tf,get_velocity=lambda:carla.Vector3D(1,2,0))
    snap=SimpleNamespace(find=lambda _:state,frame=12,timestamp=SimpleNamespace(elapsed_seconds=3.5))
    o=snapshot_obstacle(snap,SimpleNamespace(id=7,bounding_box=bb))
    xyz=np.array([[p.x,-p.y,p.z] for p in bb.get_world_vertices(tf)])
    np.testing.assert_allclose(o['xy'],xyz[:,:2].mean(0))
    yaw=o['yaw'];basis=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    local=(xyz[:,:2]-o['xy'])@basis
    assert np.all(abs(local)<=np.array([o['half_length'],o['half_width']])+1e-6)
    assert o['frame']==12 and o['timestamp']==3.5 and o['source']=='Privileged'


def test_new_actor_without_snapshot_is_rejected():
    snap=SimpleNamespace(find=lambda _:None)
    with pytest.raises(ValueError,match='absent from snapshot'):
        snapshot_obstacle(snap,SimpleNamespace(id=7))
