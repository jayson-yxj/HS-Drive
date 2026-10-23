"""Independent recorded-regression checks, read-only M6F assets."""
import json
from pathlib import Path
import pytest
from diagnostics.m6g_events import command_events,flatten,braking_metrics


def load(name):return [json.loads(s) for s in Path(name).read_text().splitlines()]


def test_recorded_delayed_brake_observation_and_request_separated():
    r=load('reports/m6f_trials_0916/responses/p126_A_402/trace.jsonl')
    e=next(e for e in command_events(r) if r[e['request_index']]['phase']=='normal_stop')
    assert e['delay_outer_ticks']==1
    assert r[e['request_index']]['body']['applied']['brake']==0
    assert r[e['applied_index']]['body']['applied']['brake']==pytest.approx(.01)


def test_recorded_sham_must_not_differentiate_cached_body_across_reset():
    root=Path('reports/m6f_trials_0916/sham/P0-sham-reapply_401');i=json.loads((root/'initialization.json').read_text())
    assert i['before_operation']['body']==i['after_operation']['body']
    assert i['before_operation']['telemetry']['data']['speed']!=i['after_operation']['telemetry']['data']['speed']
    s=flatten(load(root/'trace.jsonl'),rebuild=True)
    assert s[0]['acceleration_50ms_mps2'] is None
    assert s[1]['jerk_50ms_mps3'] is None
    assert s[2]['jerk_50ms_mps3'] is not None


def test_recorded_zero_debug_force_does_not_mean_zero_body_deceleration():
    r=load('reports/m6f_trials_0916/responses/p27_A_401/trace.jsonl');s=flatten(r)
    e=next(e for e in command_events(r) if r[e['request_index']]['phase']=='normal_stop')
    th=json.loads(Path('configs/m6g/protocol.json').read_text())['annotation_thresholds'];m=braking_metrics(s,e,th)
    assert m['peak_deceleration_mps2']>20
    assert all(abs(w['long_force'])<1 for w in m['peak_wheels'])
    assert m['status']=='Observed_outer_frames_only'


def test_all_recorded_frames_and_no_cross_gap_derivatives():
    total=0
    for p in Path('reports/m6f_trials_0916').glob('*/*/trace.jsonl'):
        config=json.loads(p.with_name('config.json').read_text());r=load(p);s=flatten(r,config['physics']!='P0-unmodified');total+=len(s)
        assert len(s)==len(r)
        assert all(x['engine_substep_timestamp']=='NotAvailable' for x in s)
    assert total==14810
