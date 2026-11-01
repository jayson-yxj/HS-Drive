from pathlib import Path
from types import SimpleNamespace as S
import ast
import json
import numpy as np
import pytest
from datasets.m8j12_contract import (CAMERAS,official_camera_contract,native_global_pose,native_target,planning_to_hs,DryRunController,assert_observation)
ROOT=Path('/root/gpufree-data/hs-drive-experiments/m8j12-sparsedrivev2/official-source')

@pytest.mark.parametrize('right,sign',[(0,0),(1,-1),(-1,1)])
def test_native_to_hs_straight_and_turn(right,sign):
    native=np.column_stack([np.arange(1,7)*right,np.arange(1,7)*2.])
    out=dict(traj_final=native,lon_reg_final=np.ones(6)*4,lat_reg_final=np.ones((15,2)))
    t,meta=planning_to_hs(out,123.45)
    np.testing.assert_array_equal(t.points[:,:2],np.column_stack([native[:,1],-native[:,0]]))
    assert np.sign(t.points[0,2])==sign and t.dt_s==.5 and len(t.points)==6 and meta['horizon_s']==3
    assert t.timestamp_s==123.45 and t.source.endswith('OfflineDryRun')

def test_no_fake_tail_or_nonfinite_output():
    o=dict(traj_final=np.zeros((6,2)),lon_reg_final=np.zeros(6),lat_reg_final=np.zeros((15,2)))
    o['traj_final'][0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):planning_to_hs(o,1)
    o['traj_final']=np.zeros((12,2))
    with pytest.raises(ValueError,match='Shape'):planning_to_hs(o,1)

def test_ego_pose_virtual_reference_without_lidar_payload():
    pose=np.eye(4);pose[:3,3]=[10,20,1]
    native=native_global_pose(pose)
    np.testing.assert_allclose(native[:3,3],[9.61,-20,2.84])
    np.testing.assert_allclose(native_target(pose,[15,22]),[2,5])

def test_official_sensor_extraction_and_crop():
    if not ROOT.exists():pytest.skip('Official source required for contract comparison')
    c=official_camera_contract(ROOT/'leaderboard/team_code/sparsedrive_b2d_agent.py')
    assert set(c['sensors'])==set(CAMERAS)
    assert c['capture_wh']==(704,396) and c['network_wh']==(704,384)
    assert c['crop']==(0,12,704,396)
    assert c['sensors']['CAM_BACK']['fov']==110
    assert c['sensors']['CAM_FRONT_RIGHT']['yaw']==55
    np.testing.assert_allclose(c['K']['CAM_FRONT'][0,0],352/np.tan(np.deg2rad(35)),atol=1e-5)

class Tensor:
    def __init__(self,a):self.a=np.asarray(a)
    def numpy(self):return self.a
    def __getitem__(self,k):return Tensor(self.a[k])

def test_controller_numeric_only_and_repeat_reset():
    if not ROOT.exists():pytest.skip('Official PID source needed')
    out=dict(lat_reg_final=Tensor(np.column_stack([np.zeros(15),np.arange(1,16)])),lon_reg_final=Tensor(np.ones(6)*2))
    args=(out,1.,[0.,10.])
    a=DryRunController(ROOT/'leaderboard/team_code/pid_controller.py');b=DryRunController(ROOT/'leaderboard/team_code/pid_controller.py')
    for _ in range(4):
        x=a.command(*args);y=b.command(*args);assert x==y and x['applied_to_vehicle'] is False
        assert -1<=x['native_agent_control']['steer']<=1
    # Structural guarantee: the output boundary has no actor/client/execution call.
    tree=ast.parse(Path('datasets/m8j12_contract.py').read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('apply_control','apply_batch','apply_batch_sync') for n in ast.walk(tree))
    assert not hasattr(a,'vehicle') and not hasattr(a,'client')

def observation():
    r=dict(frame_id=123,timestamp_s=12.3,sequence_id='S0',ego_world_matrix=np.eye(4),speed_mps=1.,imu=np.zeros(7),navigation_command=4,target_world_xy=[20.,0.],camera_frames={c:123 for c in CAMERAS},camera_timestamps_s={c:12.3 for c in CAMERAS})
    return r,{c:np.zeros((396,704,3),np.uint8) for c in CAMERAS}

def test_sync_and_exact_sensor_set():
    r,ims=observation();assert_observation(r,ims)
    r['camera_frames']['CAM_FRONT']=122
    with pytest.raises(ValueError,match='Unsynchronized'):assert_observation(r,ims)
    r,ims=observation();ims['LIDAR_TOP']=np.zeros((1,4))
    with pytest.raises(ValueError,match='CameraSet'):assert_observation(r,ims)

def test_nan_imu_and_invalid_navigation_rejected():
    r,ims=observation();r['imu'][0]=np.nan
    with pytest.raises(ValueError,match='IMU'):assert_observation(r,ims)
    r,ims=observation();r['navigation_command']=0
    with pytest.raises(ValueError,match='Navigation'):assert_observation(r,ims)
