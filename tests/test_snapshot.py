from types import SimpleNamespace
import numpy as np
import pytest
from simulator.snapshot import extract_snapshot


def test_snapshot_pose_used_for_ego_and_boxes():
    carla=pytest.importorskip('carla')
    class ActorSnapshot:
        def __init__(self,x): self.x=x
        def get_transform(self): return carla.Transform(carla.Location(x=self.x))
        def get_velocity(self): return carla.Vector3D(1,2,3)
        def get_angular_velocity(self): return carla.Vector3D(0,0,90)
    snapshots={1:ActorSnapshot(10),2:ActorSnapshot(15)}
    world=SimpleNamespace(find=snapshots.get)
    # Actor handle intentionally exposes no get_transform method.
    ego=SimpleNamespace(id=1)
    bbox=carla.BoundingBox(carla.Location(),carla.Vector3D(1,1,1))
    state,objects=extract_snapshot(world,ego,{2:('vehicle.test',bbox)})
    np.testing.assert_allclose(np.asarray(objects[0]['corners_hs_ego_m']).mean(axis=0),[5,0,0])
    np.testing.assert_allclose(state['velocity_world_mps'],[1,-2,3])
    assert state['T_world_from_ego'][0][3]==10
