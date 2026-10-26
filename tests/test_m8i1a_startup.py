"""Real subprocess exit tests with mock RPCs; no CARLA server or traffic."""
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch
import pytest
from tools.m8i1a_common import REPO, OLD_ROOT, config, persist
from tools.m8i1a_identity import EXIT, capture_identity, reconcile
from tools.m8i1a_launch import stop_owned, authorize

CASES = [('success', 0, None), ('map_mismatch', 12, 'MapMismatch'),
         ('xodr_mismatch', 13, 'XODRMismatch'), ('map_unavailable', 14, 'MapIdentityUnavailable'),
         ('unexpected_exception', 20, 'RunnerException'), ('cleanup_failure', 23, None),
         ('signal_termination', 21, 'RunnerTerminated'), ('hard_signal', -9, None),
         ('connection_failure', 10, 'ClientConnectionFailed'), ('version_mismatch', 11, 'VersionMismatch')]


def case_dir(tmp_path, name):
    parent = Path(os.environ['M8I1A_TEST_OUTPUT']) if 'M8I1A_TEST_OUTPUT' in os.environ else tmp_path
    assert parent.resolve() != OLD_ROOT.resolve()
    assert parent.resolve() != Path(config()['runtime_root']).resolve()
    p = parent / name
    p.mkdir(parents=True, exist_ok=False)
    return p


@pytest.mark.parametrize('case,expected,failure', CASES)
def test_real_runner_exit_and_durable_identity(tmp_path, case, expected, failure):
    out = case_dir(tmp_path, case)
    command = [sys.executable, '-m', 'tests.m8i1a_fakes', '--case', case, '--output', str(out)]
    with (out / 'stdout.txt').open('xb') as log:
        p = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if 'signal' in case:
                deadline = time.monotonic() + 5
                while not (out / 'ready_for_signal.json').exists():
                    assert time.monotonic() < deadline and p.poll() is None
                    time.sleep(.01)
                p.send_signal(signal.SIGKILL if case == 'hard_signal' else signal.SIGTERM)
            rc = p.wait(timeout=8)
        finally:
            if p.poll() is None:
                p.kill(); p.wait()
    assert rc == expected
    session = json.loads((out / 'session_result.json').read_text()) if (out / 'session_result.json').exists() else None
    cleanup = json.loads((out / 'runner_cleanup.json').read_text()) if (out / 'runner_cleanup.json').exists() else None
    summary = reconcile(rc, session, cleanup, dict(ok=True, scope='NoServer_SyntheticCleanup'), case)
    assert summary['propagation'] == 'StatusPropagationGo'
    assert summary['status'] == ('StartupIdentityGo_NoTraffic' if case == 'success' else 'StartupDiagnosticNoGo')
    if session:
        assert session['failure'] == failure
        assert session['expected_runner_returncode'] == rc
        assert all(session[k] == 0 for k in ['actor_creations', 'sensor_creations', 'TM_instances', 'owner_ticks', 'world_mutations'])
        trace = json.loads((out / 'synthetic_call_trace.json').read_text())
        assert all(v['durable_before_check'] for v in trace['validations'])
        allowed = {'client.set_timeout', 'client.get_client_version', 'client.get_server_version',
                   'client.get_available_maps', 'client.get_world', 'world.get_map',
                   'world.get_snapshot', 'world.get_settings', 'world.get_weather', 'map.to_opendrive'}
        assert set(trace['calls']) <= allowed
    identities = sorted(out.glob('identity_*.json'))
    assert bool(identities) == (case != 'connection_failure')
    if identities:
        row = json.loads(identities[0].read_text())
        assert row['kind'] == 'SyntheticMockCARLA_NoServer'
        if case == 'map_unavailable':
            assert row['actual_map'] is None and 'actual_map' in row['errors']
        elif case == 'map_mismatch':
            assert row['actual_map'].endswith('UnexpectedSyntheticMap')
    persist(out / 'case_assertions.json', dict(kind='ActualPythonSubprocess_MockCARLA',
            case=case, command=command, actual_returncode=rc, passed=True,
            supervisor=summary, real_server_launches=0, identity_files=len(identities)))


@pytest.mark.parametrize('mutation', ['zero_failure', 'nonzero_success', 'cleanup_mismatch',
                                    'wrong_run', 'missing_result', 'declared_wrong_code'])
def test_propagation_contradictions_fail_closed(mutation):
    session = dict(run_id='test', business_status='Failed', failure='MapMismatch',
                   expected_runner_returncode=12, cleanup_ok=True, identity_records=[{}])
    rc = 12
    cleanup = dict(run_id='test', ok=True)
    if mutation == 'zero_failure': rc = 0
    if mutation == 'nonzero_success':
        session.update(business_status='Success', failure=None, expected_runner_returncode=0)
    if mutation == 'cleanup_mismatch': cleanup['ok'] = False
    if mutation == 'wrong_run': session['run_id'] = 'stale'
    if mutation == 'missing_result': session = None; rc = 0
    if mutation == 'declared_wrong_code': session['expected_runner_returncode'] = 0
    out = reconcile(rc, session, cleanup, dict(ok=True), 'test')
    assert out['propagation'] == 'StatusPropagationNoGo'
    assert out['supervisor_returncode'] != 0 and out['status'] == 'StartupDiagnosticNoGo'


@pytest.mark.parametrize('reason', ['ServerLaunchFailed', 'ServerExited', 'ResourceBudgetExceeded', 'ClientConnectionFailed'])
def test_supervisor_failure_never_success(reason):
    result = reconcile(None, None, None, dict(ok=True), 'test', reason)
    assert result['status'] == 'StartupDiagnosticNoGo' and reason in result['failures']


def test_server_cleanup_failure_overrides_success():
    session = dict(run_id='test', business_status='Success', failure=None,
                   expected_runner_returncode=0, cleanup_ok=True, identity_records=[{}])
    result = reconcile(0, session, dict(run_id='test', ok=True), dict(ok=False), 'test')
    assert result['status'] == 'StartupDiagnosticNoGo'
    assert result['propagation'] == 'StatusPropagationGo'
    assert 'ServerCleanupFailed' in result['failures']


def test_identity_write_error_does_not_reach_validation(tmp_path):
    from tests.m8i1a_fakes import Client
    with patch('tools.m8i1a_identity.persist', side_effect=OSError('disk write failure')):
        with pytest.raises(OSError):
            capture_identity(Client('map_mismatch', []), config(),
                             dict(kind='Synthetic', run_id='test'), tmp_path, 0)


def test_atomic_history_protection_and_authorization(tmp_path):
    p = tmp_path / 'existing.json'; persist(p, dict(original=True))
    with pytest.raises(FileExistsError): persist(p, dict(original=False))
    receipt = tmp_path / 'denied.json'; persist(receipt, dict(approved=False))
    with pytest.raises(ValueError, match='AuthorizationMissing'):
        authorize(receipt, config())
    assert json.loads(p.read_text()) == dict(original=True)


def test_no_mutating_simulator_calls():
    forbidden = {'tick', 'wait_for_tick', 'spawn_actor', 'try_spawn_actor', 'get_trafficmanager',
                 'apply_settings', 'set_weather', 'load_world', 'reload_world', 'apply_batch',
                 'apply_batch_sync', 'apply_control', 'set_autopilot', 'set_transform', 'apply_physics_control'}
    for path in REPO.glob('tools/m8i1a_*.py'):
        tree = ast.parse(path.read_text())
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not calls & forbidden, path


def test_process_group_cleanup_keeps_intermediate_failures(tmp_path):
    # This is a Python sleep subprocess, NOT a simulator.
    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        result = stop_owned([p], [], 2)
        assert result['ok'] and result['final_OS_probe']['ok']
        assert result['probes']
    finally:
        if p.poll() is None: p.kill(); p.wait()
