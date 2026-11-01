import json
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
import pytest
from simulator.m8j13_shadow_contract import require_parked_command,require_completion_only,candidate_offset,parked_measurement_valid


def test_only_independent_full_brake_and_non_executable_completion_allowed():
    require_parked_command(dict(steer=0.,throttle=0.,brake=1.))
    for u in [dict(steer=0.,throttle=.1,brake=1.),dict(steer=0.,throttle=0.,brake=0.)]:
        with pytest.raises(ValueError):require_parked_command(u)
    require_completion_only(dict(status='ShadowComplete',model_forward=True,finite=True,result_expired=True))
    with pytest.raises(ValueError):require_completion_only(dict(status='ShadowComplete',model_forward=True,throttle=.5))
    assert parked_measurement_valid(.1) and not parked_measurement_valid(.1001)
    assert not parked_measurement_valid(float('nan'))


def test_fixed_candidates_and_all_inclusive_budget():
    c=json.loads(Path('configs/m8j13_parked_shadow.json').read_text())
    assert [candidate_offset(i) for i in range(80)]==list(range(0,320,4))
    with pytest.raises(ValueError):candidate_offset(80)
    assert sum(c[k] for k in ['settings_tick_charge','initialization_ticks','candidate_window_ticks','pending_drain_max_ticks','exit_observation_ticks'])<=600
    assert c['max_starts']==1 and c['max_forward_attempts']==c['planned_forwards']==16
    assert c['input_wait_wall_s']==.25 and c['max_plan_age_sim_s']==c['max_plan_age_wall_s']==.3


def test_actual_input_row_uses_navigation_consumer(monkeypatch,tmp_path):
    import sys
    from tools.m8j13_compatibility import Inputs
    from datasets.m8j12_contract import CAMERAS
    monkeypatch.setitem(sys.modules,'carla',S(Location=lambda **kw:S(**kw),Rotation=lambda **kw:S(**kw),Transform=lambda *a:S(get_matrix=lambda:np.eye(4))))
    rig=Inputs.__new__(Inputs);rig.folder=tmp_path;rig.identity=('run',1,'generation')
    rig.owner=S(cfg={'input_wait_wall_s':.25,'pose_tolerance':.0001},target=S(x=999.,y=999.))
    rig.states={7:dict(frame=7,sim_s=2.,input_wall_s=10.,pose=np.eye(4),position=[0.,0.,0.],speed=0.)}
    rig.groups={7:[dict(run='run',episode=1,sensor_generation='generation',camera=n,actor_id=i,frame=7,timestamp_s=2.,transform=np.eye(4)) for i,n in enumerate(CAMERAS)]}
    rig.sensor_map={n:S(id=i) for i,n in enumerate(CAMERAS)}
    rig.imu={7:dict(values=[0]*6+[np.pi/2],timestamp=2.)}
    rig.contract={'sensors':{n:dict(x=0,y=0,z=0,pitch=0,yaw=0,roll=0) for n in CAMERAS}}
    consumed=[]
    def step(x):consumed.append(x);return np.array([10.,0.]),S(value=4)
    rig.navigation=S(run_step=step)
    row,reason=rig.row(7,10.1)
    assert reason is None and row['target_world_xy']==[10.,0.] and row['navigation_command']==4
    assert len(consumed)==1 and row['frame_id']==7 and row['timestamp_s']==2.
    assert row['input_capture_wall_s']==10.


def test_parked_owner_cannot_be_used_to_apply_model_control():
    from tools.m8j13_parked_shadow import ParkedOwner
    owner=ParkedOwner.__new__(ParkedOwner)
    with pytest.raises(ValueError,match='ParkedOnlyControlViolation'):
        owner.apply(None,dict(steer=0.,throttle=.1,brake=0.),'Model',None)
    with pytest.raises(RuntimeError,match='ParkedControlSourceViolation'):
        owner.apply(None,dict(steer=0.,throttle=0.,brake=1.),'Model',None)
