"""Read-only CARLA identity and durable business result, independent of traffic."""
import hashlib
import os
import signal
import time
from pathlib import Path
from tools.m8i1a_common import persist, sha

EXIT = dict(Success=0, ClientConnectionFailed=10, VersionMismatch=11,
            MapMismatch=12, XODRMismatch=13, MapIdentityUnavailable=14,
            WorldIdentityUnavailable=15, StartupIdentityUnstable=16,
            RunnerException=20, RunnerTerminated=21, ResourceBudgetExceeded=22,
            CleanupFailed=23, ServerLaunchFailed=24, ServerExited=25,
            PortOccupied=26, AuthorizationMissing=27, StatusPropagationNoGo=28)
SETTINGS = ['synchronous_mode', 'no_rendering_mode', 'fixed_delta_seconds',
            'substepping', 'max_substep_delta_time', 'max_substeps',
            'max_culling_distance', 'deterministic_ragdolls',
            'tile_stream_distance', 'actor_active_distance', 'spectator_as_ego']


class Rejected(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def terminated(signum, frame):
    raise Rejected('RunnerTerminated')


def capture_identity(client, cfg, context, output, sequence):
    """Always persist available identity/errors BEFORE version/map/XODR decisions."""
    row = dict(kind=context['kind'], run_id=context['run_id'], sequence=sequence,
               expected_map=cfg['expected_map'], actual_map=None, world_episode=None,
               snapshot=None, client_version=None, server_version=None,
               world_settings=None, world_settings_raw=None, weather_raw=None,
               xodr_sha256=None, xodr_bytes=None, available_maps=None,
               context=context, errors={}, captured_wall_time_ns=time.time_ns())

    def read(key, fn):
        try:
            row[key] = fn()
        except Exception as exc:
            if isinstance(exc, Rejected):
                raise
            row['errors'][key] = f'{type(exc).__name__}: {exc}'
        return row[key]

    read('client_version', client.get_client_version)
    read('server_version', client.get_server_version)
    world = None
    try:
        world = client.get_world()
    except Exception as exc:
        if isinstance(exc, Rejected):
            raise
        row['errors']['world'] = f'{type(exc).__name__}: {exc}'
    if world is not None:
        read('world_episode', lambda: world.id)
        def snapshot():
            snap = world.get_snapshot()
            return dict(frame=int(snap.frame), elapsed_seconds=float(snap.timestamp.elapsed_seconds),
                        delta_seconds=float(snap.timestamp.delta_seconds),
                        platform_timestamp=float(snap.timestamp.platform_timestamp))
        read('snapshot', snapshot)
        try:
            settings = world.get_settings()
            row['world_settings_raw'] = str(settings)
            row['world_settings'] = {k: getattr(settings, k) for k in SETTINGS if hasattr(settings, k)}
        except Exception as exc:
            if isinstance(exc, Rejected):
                raise
            row['errors']['world_settings'] = f'{type(exc).__name__}: {exc}'
        read('weather_raw', lambda: str(world.get_weather()))
        try:
            mp = world.get_map()
            read('actual_map', lambda: mp.name)
            def xodr():
                raw = mp.to_opendrive().encode('utf-8')
                row['xodr_bytes'] = len(raw)
                return hashlib.sha256(raw).hexdigest()
            read('xodr_sha256', xodr)
        except Exception as exc:
            if isinstance(exc, Rejected):
                raise
            row['errors']['actual_map'] = f'{type(exc).__name__}: {exc}'
    read('available_maps', client.get_available_maps)
    target = Path(output) / f'identity_{sequence:03d}.json'
    persist(target, row)
    return row, dict(path=str(target), sha256=sha(target))


def validate_identity(row, cfg):
    if row['actual_map'] is None:
        raise Rejected('MapIdentityUnavailable')
    if row['client_version'] != cfg['expected_version'] or row['server_version'] != cfg['expected_version']:
        raise Rejected('VersionMismatch')
    if row['actual_map'].split('/')[-1] != cfg['expected_map']:
        raise Rejected('MapMismatch')
    if row['xodr_sha256'] is None:
        raise Rejected('MapIdentityUnavailable')
    if row['xodr_sha256'] != cfg['expected_xodr_sha256']:
        raise Rejected('XODRMismatch')
    if row['world_episode'] is None or row['snapshot'] is None or row['world_settings'] is None:
        raise Rejected('WorldIdentityUnavailable')


def run_session(client_factory, cfg, context, output, cleanup=None,
                clock=time.monotonic, sleep=time.sleep):
    """Returns the real runner exit code, after persisting cleanup and Session Result."""
    out = Path(output)
    failure = None
    detail = None
    observations = []
    phase = 'connect'
    try:
        client = client_factory()
        client.set_timeout(cfg['rpc_timeout_seconds'])
        deadline = clock() + cfg['connect_seconds']
        attempt = 0
        while True:
            attempt += 1
            try:
                client.get_server_version()
                break
            except Exception as exc:
                if isinstance(exc, Rejected):
                    raise
                persist(out / f'connection_{attempt:03d}.json', dict(
                    kind=context['kind'], run_id=context['run_id'], error=f'{type(exc).__name__}: {exc}'))
                if clock() >= deadline:
                    raise Rejected('ClientConnectionFailed') from exc
                sleep(cfg['poll_seconds'])
        phase = 'identity'
        deadline = clock() + cfg['identity_seconds']
        previous = None
        for sequence in range(cfg['stable_match_reads']):
            if clock() >= deadline:
                raise Rejected('StartupIdentityUnstable')
            row, evidence = capture_identity(client, cfg, context, out, sequence)
            observations.append(evidence)
            # Validation happens only after fsync and returns no simulator mutation.
            validate_identity(row, cfg)
            if previous is not None:
                if any(row[k] != previous[k] for k in ['world_episode', 'actual_map', 'xodr_sha256', 'world_settings']):
                    raise Rejected('StartupIdentityUnstable')
                if (row['snapshot']['frame'] < previous['snapshot']['frame'] or
                        row['snapshot']['elapsed_seconds'] < previous['snapshot']['elapsed_seconds']):
                    raise Rejected('StartupIdentityUnstable')
            previous = row
            if sequence + 1 < cfg['stable_match_reads']:
                sleep(cfg['poll_seconds'])
    except Rejected as exc:
        failure, detail = exc.reason, str(exc)
    except Exception as exc:
        failure = 'ClientConnectionFailed' if phase == 'connect' else 'RunnerException'
        detail = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            clean = cleanup() if cleanup else dict(ok=True, owned_actor_ids=[],
                created_actors=0, created_sensors=0, world_mutations=0,
                scope='Runner owns no simulator resources; supervisor must stop owned server')
            if not isinstance(clean, dict) or clean.get('ok') is not True:
                clean = dict(ok=False, reason='CleanupReturnedFailure', original=clean)
        except Exception as exc:
            clean = dict(ok=False, reason=f'{type(exc).__name__}: {exc}')
        persist(out / 'runner_cleanup.json', dict(run_id=context['run_id'], **clean))
        business_code = EXIT[failure] if failure else 0
        code = business_code or (EXIT['CleanupFailed'] if not clean['ok'] else 0)
        result = dict(kind=context['kind'], run_id=context['run_id'],
                      business_status='Failed' if failure else 'Success', failure=failure, error=detail,
                      expected_runner_returncode=code, cleanup_ok=clean['ok'],
                      identity_records=observations, actor_creations=0, sensor_creations=0,
                      TM_instances=0, owner_ticks=0, world_mutations=0,
                      outcome='StartupIdentityGo_NoTraffic' if code == 0 else 'StartupDiagnosticNoGo')
        persist(out / 'session_result.json', result)
    return code


def reconcile(returncode, session, runner_cleanup, server_cleanup, run_id, supervisor_error=None):
    """Fail closed over process, business, AND cleanup evidence; failures stay visible."""
    contradictions, failures = [], []
    if supervisor_error:
        failures.append(supervisor_error)
    if session is None:
        failures.append('RunnerTerminated' if returncode is not None and returncode < 0 else 'SessionResultMissing')
        if returncode == 0:
            contradictions.append('ZeroExitWithoutBusinessResult')
    else:
        if session.get('run_id') != run_id:
            contradictions.append('SessionRunIDMismatch')
        success = session.get('business_status') == 'Success'
        failure = session.get('failure')
        if success == bool(failure) or session.get('business_status') not in ['Success', 'Failed']:
            contradictions.append('BusinessStatusContradiction')
        expected = EXIT.get(failure) if failure else (0 if session.get('cleanup_ok') is True else EXIT['CleanupFailed'])
        if expected is None or session.get('expected_runner_returncode') != expected:
            contradictions.append('DeclaredExitContradiction')
        if returncode != expected:
            contradictions.append('ProcessExitContradiction')
        if not success:
            failures.append(failure or 'InvalidBusinessFailure')
        if session.get('cleanup_ok') is not True:
            failures.append('RunnerCleanupFailed')
        if success and not session.get('identity_records'):
            contradictions.append('SuccessWithoutIdentityEvidence')
    if returncode not in (0, None):
        failures.append('RunnerNonzeroExit')
    if not runner_cleanup or runner_cleanup.get('ok') is not True or runner_cleanup.get('run_id') != run_id:
        failures.append('RunnerCleanupMissingOrFailed')
    if session and runner_cleanup and session.get('cleanup_ok') != runner_cleanup.get('ok'):
        contradictions.append('CleanupStatusContradiction')
    if not server_cleanup or server_cleanup.get('ok') is not True:
        failures.append('ServerCleanupFailed')
    if contradictions:
        failures.append('StatusPropagationNoGo')
    ok = not failures and not contradictions and returncode == 0
    return dict(run_id=run_id, runner_returncode=returncode,
                propagation='StatusPropagationNoGo' if contradictions else 'StatusPropagationGo',
                contradictions=contradictions, failures=failures,
                status='StartupIdentityGo_NoTraffic' if ok else 'StartupDiagnosticNoGo',
                supervisor_returncode=0 if ok else (EXIT['StatusPropagationNoGo'] if contradictions else 1))
