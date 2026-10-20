from tools.record_dataset import parse_args
from simulator.capture_config import load_capture_defaults
import pytest


def test_capture_defaults_keep_geometry_default_and_allow_cli_override():
    assert parse_args([]).camera_profile is None
    args = parse_args(['--capture-config','configs/data_capture.yaml',
                      '--frames','61','--motion','straight'])
    assert args.frames == 61 and args.motion == 'straight'
    assert args.tick_wall_delay == .5 and args.warmup_ticks == 40
    assert args.render_every_tick is True and parse_args([]).render_every_tick is False
    assert args.camera_profile.endswith('training_960_v4.yaml')
    assert parse_args(['--capture-config','configs/data_capture.yaml','--no-render-every-tick']).render_every_tick is False


def test_unknown_capture_defaults_cannot_silently_change_behavior(tmp_path):
    path = tmp_path/'defaults.yaml'
    path.write_text('schema_version: hsd-capture-defaults-v1\ncollector:\n  camera_profiel: bad.yaml\n')
    with pytest.raises(ValueError, match='Unknown collector'):
        load_capture_defaults(path)
