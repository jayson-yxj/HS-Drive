"""No private observations or network needed for the intervention boundary."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_replay_has_no_vehicle_or_training_calls():
    tree=ast.parse(Path('tools/m8j12_replay.py').read_text())
    forbidden={'apply_control','apply_ackermann_control','backward','step','set_autopilot'}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]
    assert 'carla' not in [n.name for n in ast.walk(tree) if isinstance(n,ast.alias)]


def test_fixed_budget_and_capture_selection():
    p=json.loads(Path('configs/m8j12_protocol.json').read_text())
    c=json.loads(Path('configs/m8j12_capture.json').read_text())
    assert p['max_full_forwards']==160 and p['planned_primary_forwards']==128
    assert p['samples_per_sequence']==c['frames']==32
    assert (c['settle_ticks']+c['drive_warmup_ticks']+32+10)*c['physics_dt_s']<=20
    assert len(c['sequences'])==2
    for s in c['sequences']:
        import hashlib
        assert hashlib.sha256(Path(s['route_file']).read_bytes()).hexdigest()==s['route_sha256']


def test_complete_reset_reuses_native_methods():
    # This import has no official package dependency until build is explicitly used.
    pytest.importorskip('torch')
    from tools.m8j12_runtime import reset
    calls=[]
    def state(name):return SimpleNamespace(reset=lambda:calls.append(name))
    m=SimpleNamespace(head=SimpleNamespace(det_head=SimpleNamespace(instance_bank=state('det')),
        map_head=SimpleNamespace(instance_bank=state('map')),motion_plan_head=SimpleNamespace(instance_queue=state('queue'))))
    reset(m)
    assert calls==['det','map','queue']


def test_observation_allowlist_has_no_environment_truth():
    from datasets.m8j12_contract import MODEL_KEYS
    assert MODEL_KEYS=={'img','timestamp','projection_mat','image_wh','ego_status','gt_ego_fut_cmd','tp_near','img_metas'}
    # gt_ego_fut_cmd is the official name of the route command, not future states.
    source=Path('tools/m8j12_runtime.py').read_text()
    assert "row['navigation_command']-1" in source
    assert 'gt_bboxes' not in source and 'gt_map' not in source


def test_measured_compass_target_matches_native_planar_axes():
    import numpy as np
    from datasets.m8j12_contract import measured_navigation_target
    row=dict(ego_world_matrix=np.eye(4),target_world_xy=[10.,2.],imu=[0,0,0,0,0,0,np.pi/2])
    np.testing.assert_allclose(measured_navigation_target(row),[2,10],atol=1e-6)
