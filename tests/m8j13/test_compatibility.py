import ast
import json
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
import pytest
from datasets.m8j12_contract import native_target,measured_navigation_target
from simulator.m8j13_safety import SafetyGate,watchdog_reason


def test_compass_geometry_front_back_left_right():
    for yaw in [0,np.pi/2,-np.pi/2,np.pi]:
        c,s=np.cos(yaw),np.sin(yaw);pose=np.eye(4);pose[:2,:2]=[[c,-s],[s,c]];pose[:2,3]=[10,20]
        for x,y in [(10,0),(-10,0),(0,-10),(0,10)]:
            world=(pose@np.array([x,y,0,1]))[:2]
            row=dict(ego_world_matrix=pose,target_world_xy=world,imu=[0]*6+[(yaw+np.pi/2)%(2*np.pi)])
            np.testing.assert_allclose(native_target(pose,world),[y,x],atol=1e-5)
            np.testing.assert_allclose(measured_navigation_target(row),[y,x],atol=1e-5)


def test_blocked_planner_cannot_renew_old_command():
    heartbeat=dict(wall_s=10.,command_valid_until_wall_s=10.3)
    assert watchdog_reason(heartbeat,10.31)=='ControlWatchdogExpired'
    gate=SafetyGate({});stop=gate.stop('ControlWatchdogExpired',1.,10.31)
    assert gate.command(dict(steer=0.,throttle=.25,brake=0.),speed=0.,sim_s=2.,wall_s=12.)==stop


def test_model_service_has_no_actuator_and_responses_do_not_expose_control():
    source=Path('tools/m8j13_shadow_service.py').read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ['apply_control','tick','set_transform'] for n in ast.walk(tree))
    assert 'import carla' not in source
    for line in source.splitlines():
        if "root/'responses'" in line:assert 'control' not in line and 'throttle' not in line


def test_compatibility_budget_covers_all_phases_without_reset():
    c=json.loads(Path('configs/m8j13_compatibility.json').read_text())
    assert c['shadow_max_ticks']+4*c['round_max_ticks']+2*c['settings_tick_charge']==1600
    assert c['planned_forwards']==16 and c['max_forward_attempts']==32 and c['max_fault_rounds']==4
    assert c['round_max_ticks']*.05==15 and c['stop_hold_sim_s']==1


def test_initialization_failure_still_restores_world(tmp_path,monkeypatch):
    from tools import m8j13_compatibility as probe
    events=[]
    class FailedOwner:
        def __init__(self,root,cfg):
            self.tick_count=0
            events.append('settings_applied')
            raise RuntimeError('GeometryContractConflict')
        def close(self):
            events.append('settings_restored')
            return {'restored':True}
    monkeypatch.setattr(probe,'Owner',FailedOwner)
    probe.worker(tmp_path,'shadow')
    result=json.loads((tmp_path/'result.json').read_text())
    assert events==['settings_applied','settings_restored']
    assert result['status']=='Partial' and result['cleanup']['restored']
    assert result['explicit_ticks']==0 and result['model_controls']==0


def test_gps_conflict_blocks_shadow_before_actor_or_model_creation(tmp_path):
    from tools.m8j13_compatibility import shadow
    (tmp_path/'geometry.json').write_text(json.dumps({'status':'GPSProjectionIncompatible'}))
    # No owner interfaces are supplied: the geometry gate must reject before
    # accessing vehicle creation, model launch, or observations.
    with pytest.raises(RuntimeError,match='ShadowBlockedByGPSCompatibility'):
        shadow(object(),tmp_path)
    assert list(tmp_path.iterdir())==[tmp_path/'geometry.json']
