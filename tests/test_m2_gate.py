import json,subprocess,sys
from pathlib import Path


def test_gate_fails_if_motion_is_not_verified(tmp_path):
    reports=tmp_path/'reports';reports.mkdir()
    evidence={
      'm2_town03_recording':{'m1_automated_checks_passed':True},
      'm2_town03_motion_geometry':{'motion_passed':False,'cross_view_passed':True},
      'm2_town03_slope_geometry':{'cross_view_passed':True},
      'm2_occlusion':{'passed':True},
      'm2_dynamic_actor_surfaces':{'dynamic_target_passed':True,'self_body_review_passed':True},
      'm2_interrupt':{'passed':True},'m2_render':{'passed':True},
      'depth_convention':{'convention':'optical_z'}}
    for name,data in evidence.items():(reports/(name+'.json')).write_text(json.dumps(data))
    script=Path('tools/validate_m2_gate.py').resolve()
    result=subprocess.run([sys.executable,str(script)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==1
    assert json.loads((reports/'m2_gate.json').read_text())['passed'] is False
