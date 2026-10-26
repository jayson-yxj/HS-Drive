"""Append-only closeout of the one separately authorized startup; no CARLA RPC."""
import hashlib
import json
from pathlib import Path
import subprocess
from tools.m8i1a_common import REPO, asset, sha, persist, config

EXECUTED = '2cc22439587976d87550bf411e22f625d30508d0'
RUNTIME = Path('/root/gpufree-data/hs-drive-experiments/m8i1a-startup-identity')
EVIDENCE = Path('/root/gpufree-data/hs-drive-experiments/m8i1a-approved-execution-evidence')


def read(path):
    return json.loads(Path(path).read_text())


def report(name, **fields):
    persist(REPO / f'reports/m8i1a_execution_{name}.json', dict(
        kind='ActualAuthorizedReadOnlyStartup_Closeout', executed_commit=EXECUTED,
        run_id=read(RUNTIME / 'context.json')['run_id'],
        closeout_source=asset(__file__), actor_creations=0, sensor_creations=0,
        TM_instances=0, owner_ticks=0, model_forwards=0, training_steps=0,
        visual_control=False, new_Final=0, **fields))


def main():
    cfg = config()
    context = read(RUNTIME / 'context.json')
    session = read(RUNTIME / 'session_result.json')
    supervisor = read(RUNTIME / 'supervisor_result.json')
    cleanup = read(RUNTIME / 'cleanup.json')
    post = read(EVIDENCE / 'post_shutdown_OS_probe.json')
    authorization = read(EVIDENCE / 'authorization.json')
    exit_receipt = read(EVIDENCE / 'launcher_exit_receipt.json')
    assert context['git_head'] == EXECUTED
    verified = {}
    for path, digest in context['source_sha256'].items():
        original = subprocess.check_output(['git', 'show', EXECUTED + ':' + path], cwd=REPO)
        verified[path] = hashlib.sha256(original).hexdigest() == digest == sha(REPO / path)
    assert all(verified.values())
    assert session['failure'] == 'ClientConnectionFailed'
    assert session['identity_records'] == []
    assert supervisor['runner_returncode'] == 10 and supervisor['supervisor_returncode'] == 1
    assert exit_receipt['actual_supervisor_process_returncode'] == 1
    assert supervisor['server_launches'] == 1 and supervisor['propagation'] == 'StatusPropagationGo'
    assert not supervisor['contradictions']
    assert cleanup['ok'] and post['probe']['ok'] and not post['matching_CARLA_processes']
    assert not list(RUNTIME.glob('identity_*.json')) and not list(RUNTIME.rglob('*.png'))
    connection_files = sorted(RUNTIME.glob('connection_*.json'))
    connection_errors = [read(p) for p in connection_files]
    assert len(connection_files) == 31
    assert all('time-out of 2000ms' in r['error'] for r in connection_errors)
    seal_checks = []
    for name in ['m8i1_release_manifest.json', 'm8i1a_release_manifest.json', 'm8i1a_prelaunch_freeze.json']:
        original = read(REPO / 'reports' / name)
        for row in original['files']:
            assert sha(row['path']) == row['sha256'], row['path']
        seal_checks.append(dict(manifest=asset(REPO / 'reports' / name), verified_files=len(original['files'])))
    assert not subprocess.check_output(['git', 'diff', EXECUTED, '--name-status', '--diff-filter=MDTR'], cwd=REPO, text=True).strip()
    report('preflight', status='Passed_BeforeSingleAuthorizedAttempt',
           authorization=authorization, authorization_file=asset(EVIDENCE / 'authorization.json'),
           no_message_id_invented=authorization['user_message_id'] is None,
           recorded_user_message_is_current_conversation=True,
           prelaunch=read(EVIDENCE / 'prelaunch_state.json'),
           sealed_input_checks=seal_checks, executed_source_sha_verified=verified,
           offline_test_count=25, no_parameter_or_source_changes_during_attempt=True)
    binary = read(RUNTIME / 'server_process_binary.json')
    report('map_identity', status='StartupIdentityUnknown', expected_map='Town02',
           actual_map=None, world_episode=None, snapshot_frame=None, snapshot_timestamp=None,
           client_version_RPC=None, server_version_RPC=None, actual_XODR_SHA256=None,
           expected_version=cfg['expected_version'], expected_XODR_SHA256=cfg['expected_xodr_sha256'],
           unknown_reason='Version RPC never succeeded; get_world/get_map/identity capture were not reached',
           identity_validation='NotReached_NoMapJudgmentMade',
           actual_process_identity=binary, initial_process_identity=read(RUNTIME / 'server_process_initial.json'),
           actual_process_identity_file=asset(RUNTIME / 'server_process_binary.json'),
           requested_cwd=context['requested_server_cwd'], commands=read(RUNTIME / 'execution_commands.json'),
           no_expected_field_substituted_for_actual=True)
    report('startup_readiness', status='StartupDiagnosticNoGo',
           reason='ClientConnectionFailed', connect_window_s=cfg['connect_seconds'],
           per_rpc_timeout_s=cfg['rpc_timeout_seconds'], poll_interval_s=cfg['poll_seconds'],
           failed_version_RPC_attempts=len(connection_files), successful_version_RPC_attempts=0,
           connection_errors=connection_errors, total_supervised_wall_s=supervisor['wall_elapsed_s'],
           bounded_connection_deadline_semantics='Deadline checked after a failed RPC; final 2s RPC can cross the 90s connection boundary, within 180s total limit',
           during_startup_OS=read(EVIDENCE / 'during_startup_OS_probe.json'),
           TCP_accept_is_not_application_RPC_success=True,
           matching_identity_confirmations=0, required_matching_confirmations=2,
           root_cause='Unresolved', retries_of_server_process=0)
    report('exit_status', status='StatusPropagationGo',
           actual_runner_returncode=supervisor['runner_returncode'], session=session,
           supervisor=supervisor, actual_supervisor_returncode=exit_receipt['actual_supervisor_process_returncode'],
           shell_exit_evidence=exit_receipt, cleanup=read(RUNTIME / 'runner_cleanup.json'),
           business_process_supervisor_consistent=True,
           tested_runtime_path='Actual ClientConnectionFailed -> runner10 -> supervisor1',
           map_mismatch_path='OfflineTestsOnly_NotReachedInThisRuntime',
           successful_identity_path='OfflineTestsOnly_NotReachedInThisRuntime')
    report('cleanup', status='CleanupGo', immediate_and_final_supervisor_evidence=cleanup,
           independent_post_shutdown_OS=post,
           transient_failures_preserved=sum(not p['ok'] for p in cleanup['probes']),
           final_groups_absent=not any(post['probe']['group_alive'].values()),
           final_ports_closed=not any(post['probe']['listening_ports'].values()),
           original_world_or_weather_modified=False,
           claim_scope='Own server terminated and OS resources absent; no actor/sensor creation code reached, no per-Actor RPC inventory possible',
           log_inventory=[asset(p) for p in sorted(RUNTIME.glob('*.log'))],
           log_timing_limitation='Server stdout lacks per-line timestamps; EOF/stream errors cannot alone establish pre-shutdown cause',
           runtime_bytes=sum(p.stat().st_size for p in RUNTIME.rglob('*') if p.is_file()),
           wall_s=supervisor['wall_elapsed_s'], max_wall_s=180, max_log_bytes=67108864)
    report('acceptance', status='StartupDiagnosticNoGo',
           identity='StartupIdentityUnknown', propagation='StatusPropagationGo', cleanup='CleanupGo',
           runtime='ExecutedOnce_ConnectionFailed', server_launches=1, restarts=0,
           actual_runner_returncode=10, actual_supervisor_returncode=1,
           answers=dict(A_ActualMap='Unknown_RPCConnectionFailed_NoWorldRead',
                        B_IdentityBeforeMapDecision='OfflineVerified_RuntimeStageNotReached',
                        C_NonzeroFailurePropagation='ActualConnectionFailurePassed_MapRejectOfflineOnly',
                        D_RootCause='Unresolved_ActualProcessCWDAndARGVNowRecorded',
                        E_ReadOnlyDiagnostic='OneActualStartupAttempt_IdentityGoalNotAchieved',
                        F_NoTrafficSensorTick='ConfirmedNoCreationOrOwnerTickPath',
                        G_Cleanup='CleanupGo_FinalIndependentOSProbe'),
           TM_pilot_proposal_eligible=False, TM_pilot_authorized=False,
           further_startup_authorized=False, approval_consumed=True,
           old_m8i1_status='PilotNoGo_Unchanged', old_offline_reports='Preserved_AsOfPreapproval',
           current_state_document='reports/m8i1a_execution_summary.md',
           no_main_merge=True, default_system_unchanged=True)
    print(json.dumps(dict(status='StartupDiagnosticNoGo', propagation='StatusPropagationGo',
                          cleanup='CleanupGo', attempts=1, runner_rc=10, supervisor_rc=1), indent=2))


if __name__ == '__main__':
    main()
