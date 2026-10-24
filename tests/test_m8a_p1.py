"""Safety-critical P0 harness contract tests; no pretrained model required."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import pytest
from tools.m8a_p1_common import CFG,environment,write


def test_frozen_original_spec_and_no_training():
    assert CFG['input_shape']==[1,6,3,512,1408]
    assert CFG['warmup']==3 and CFG['measured_forwards']==20
    assert CFG['training_steps']==0 and not CFG['visual_control']
    assert CFG['official_commit']=='f7525f93467a33707ef401c587a52d5e7b34de74'
    assert CFG['checkpoint_sha256']=='16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad'


def test_artifacts_append_only(tmp_path):
    target=tmp_path/'evidence.json';write(target,{'status':'Failed'})
    with pytest.raises(FileExistsError):write(target,{'status':'Passed'})
    assert json.loads(target.read_text())=={'status':'Failed'}


def test_subprocess_does_not_inherit_credentials(monkeypatch):
    monkeypatch.setenv('GITHUB_TOKEN','sentinel');monkeypatch.setenv('HTTPS_PROXY','http://private:secret@private:1234')
    env=environment(False)
    assert 'GITHUB_TOKEN' not in env and 'HTTPS_PROXY' not in env
    assert env['PIP_CONFIG_FILE']=='/dev/null'
    assert environment(True)['HTTPS_PROXY']==CFG['proxy']


def test_gpu_reader_has_no_pickle_and_strict_load():
    tree=ast.parse(Path('tools/m8a_p1_gpu_probe.py').read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not any(isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='torch' and c.func.attr=='load' for c in calls)
    loads=[c for c in calls if isinstance(c.func,ast.Attribute) and c.func.attr=='load_state_dict']
    assert len(loads)==1 and any(k.arg=='strict' and isinstance(k.value,ast.Constant) and k.value.value is True for k in loads[0].keywords)
    np_loads=[c for c in calls if isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='np' and c.func.attr=='load']
    assert len(np_loads)==1 and any(k.arg=='allow_pickle' and isinstance(k.value,ast.Constant) and k.value.value is False for k in np_loads[0].keywords)


def test_camera_check_is_not_model_inference():
    source=Path('tools/m8a_p1_camera_check.py').read_text()
    assert 'build_model' not in source and 'load_state_dict' not in source
    assert "p['ego_pose']" not in source
    assert 'carla_model_forward=False' in source
