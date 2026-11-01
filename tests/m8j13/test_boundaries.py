import ast,json
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
import pytest
from simulator.synchronizer import FrameSynchronizer
from datasets.schema import SensorSample
from datasets.m8j12_contract import CAMERAS,assert_observation
from simulator.m8j13_carla import Actuator


def sample(name,frame,timestamp):return SensorSample(name,frame,timestamp,1,1,bytes(4),np.eye(4))


def test_callback_order_never_substitutes_for_frame_join():
    sync=FrameSynchronizer(CAMERAS,cadence_ticks=1)
    sync.begin_interval(100,explicit_cadence=True)
    sync.add_snapshot(100,5.,{},[]);sync.add_snapshot(101,5.05,{},[])
    for n in reversed(CAMERAS):sync.callback(n)(sample(n,101,5.05))
    for n in CAMERAS[:-1]:sync.callback(n)(sample(n,100,5.))
    assert sync.drain()==[]
    sync.callback(CAMERAS[-1])(sample(CAMERAS[-1],100,5.))
    assert [b.frame_id for b in sync.drain()]==[100,101]


def test_missing_camera_is_reported_not_synthesized():
    sync=FrameSynchronizer(CAMERAS,cadence_ticks=1,timeout_ticks=2)
    sync.begin_interval(10,explicit_cadence=True);sync.add_snapshot(10,1.,{},[])
    for n in CAMERAS[:-1]:sync.callback(n)(sample(n,10,1.))
    assert sync.drain()==[]
    sync.add_snapshot(11,1.05,{},[]);sync.add_snapshot(12,1.1,{},[])
    assert sync.drain()==[] and sync.stats['dropped_frame']==1


def test_sensor_timestamp_mismatch_rejected():
    sync=FrameSynchronizer(CAMERAS,cadence_ticks=1)
    sync.begin_interval(10,explicit_cadence=True);sync.add_snapshot(10,1.,{},[])
    for i,n in enumerate(CAMERAS):sync.callback(n)(sample(n,10,1. if i else 1.05))
    assert sync.drain()==[] and sync.stats['invalid_frame']==1


def test_watchdog_has_no_model_dependency_or_tick_ownership():
    tree=ast.parse(Path('tools/m8j13_watchdog.py').read_text())
    imports=[n.name for n in ast.walk(tree) if isinstance(n,ast.alias)]
    assert not any('torch' in n or 'm8j12_runtime' in n for n in imports)
    assert not any(isinstance(n,ast.Attribute) and n.attr=='tick' for n in ast.walk(tree))
    assert any(isinstance(n,ast.Attribute) and n.attr=='apply_control' for n in ast.walk(tree))


def test_actuation_feedback_mismatch_raises():
    a=Actuator(None,S(get_control=lambda:S(steer=0.,throttle=1.,brake=0.)))
    with pytest.raises(RuntimeError,match='FeedbackMismatch'):a.verify(dict(steer=0.,throttle=0.,brake=1.))


def test_camera_every_tick_and_cleanup_fresh_snapshot():
    source=Path('simulator/m8j13_carla.py').read_text()
    assert "('sensor_tick',0.)" in source
    cleanup=Path('tools/m8j13_safety_campaign.py').read_text()
    assert 'cleanup_snapshot=session.world.wait_for_tick' in cleanup
    assert 'cleanup_snapshot.find(i)' in cleanup


def test_original_pid_ast_methods_equal_source():
    c=json.loads(Path('configs/m8j13_protocol.json').read_text())
    p=Path(c['runtime_root'])/'official-source/leaderboard/team_code/pid_controller.py'
    tree=ast.parse(p.read_text());classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    assert [n.name for n in classes]==['PID','PIDController']
    # A real numerical source equivalence regression runs in the torch environment.
    from simulator.m8j13_controller import native_controller,native_command
    from datasets.m8j12_contract import DryRunController
    original=DryRunController(p);adapted=native_controller(p)
    from simulator.m8j13_controller import ArrayTensor
    for speed in [0.,1.,2.,3.,1.]:
        path=np.column_stack([np.arange(15)*.2,np.arange(1,16)])
        speeds=np.full(6,2.)
        a=native_command(adapted,path,speeds,speed,[0,10])
        b=original.command(dict(lat_reg_final=ArrayTensor(path),lon_reg_final=ArrayTensor(speeds)),speed,[0,10])
        assert a==b
