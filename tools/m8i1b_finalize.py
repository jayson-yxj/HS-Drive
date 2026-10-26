"""Append-only offline decision, provenance erratum and release seal."""
from tools.m8i1b_forensics import *


def main():
    approval = read(APPROVAL / 'authorization.json')
    assert sha(APPROVAL / 'authorization.json') == '6abee8f1a7ab7fbec4f9134eba616aea133d63b2556d1eda735642bc1bb39023'
    report('authorization_provenance', status='AppendOnlySourceClarification',
           original_receipt=asset(APPROVAL / 'authorization.json'), original_receipt_verbatim=approval,
           user_provided_external_transcript=asset(REPO / 'reports/m8i1b_authorization_source_transcript.md'),
           source_chain=[dict(context='ExternalChatGPT', role='user', action='AuthorizationRequest'),
                         dict(context='ExternalChatGPT', role='assistant', action='ApprovalReply'),
                         dict(context='CurrentCodexConversation', role='user', action='RelayAndExecutionInstruction')],
           external_roles_evidence='UserProvidedTranscript_NotIndependentlyVerifiedPlatformExport',
           external_message_id=None, external_exact_message_timestamp=None,
           external_platform_export=None, external_conversation_share_link=None,
           transcript_claimed_date='2026-10-26 UTC+8',
           local_receipt_creation_not_external_message_time=approval['record_created_utc'],
           original_receipt_unchanged=True, already_consumed_attempts=1,
           additional_launch_authorized=False)
    clock = read(OUT / 'clock_anchor.json')
    process = read(I1A / 'server_process_binary.json')['members']
    times = []
    for p in process:
        fields = p['stat'].split(') ', 1)[1].split()
        ticks = int(fields[19])
        # Only a clock cross-check, not proof of historical wall-clock accuracy.
        approx = dt.datetime.fromisoformat(clock['wall_utc']).timestamp() - clock['clock_boottime_s'] + ticks / clock['clk_tck']
        times.append(dict(pid=p['pid'], process_group=p['pgid'], start_clock_ticks=ticks,
                          inferred_wall_start_utc=utc(approx), actual_cwd=p['cwd'],
                          recorded_argv=p['cmdline'], quality='RecordedProcTicks_CurrentBootClockApproximation'))
    report('time_and_process_association', execution_windows=WINDOWS,
           clock_anchor=asset(OUT / 'clock_anchor.json'),
           original_artifact_times=read(OUT / 'execution_window_anchors.json'),
           I1a_proc_time_evidence=times,
           I1_process_time='NoProcStatRecorded_LaunchMonotonicAndFileMtimeOnly',
           H1_process_time='PIDInCleanupReport_ExactLaunchNotRecorded',
           limitation='Filemtime/current boot conversion are auxiliary evidence, not per-line UE event time or platform message time')
    for row in read(OUT / 'historical_freeze.json')['files']:
        assert sha(row['path']) == row['sha256']
    changed = subprocess.check_output(['git','diff',BASE,'--name-status','--diff-filter=MDTR'],cwd=REPO,text=True)
    assert not changed.strip(), changed
    report('acceptance', status='OfflineForensicsCompleted',
           decision='StartupRootCauseUnresolved', decision_option='B',
           actionable_startup_hypothesis_supported=False, environment_issue_supported=False,
           UE_log_status='NoRunMatchedUELog',
           failure_conclusions=dict(M8_I1='VersionRPCSucceeded_MapRejected_ActualMapNotRecorded',
                                    M8_I1a='31VersionRPCTimeouts_NoMapIdentityRead'),
           I1a_highest_supported_state='TCPListening',
           historical_manifests_verified=5, historical_unique_assets_verified=306,
           new_simulation_tests=0, new_model_tests=0, no_port_probes=True,
           authorization_erratum='ExternalUserRequest_ExternalAssistantApproval_UserRelay',
           external_authorization_evidence_quality='UserProvidedTranscript_NoPlatformExport',
           original_authorization_preserved=True, prior_launch_quota_unchanged=True,
           new_startup_authorized=False, next_startup_experiment_proposed=False,
           next_direction='Bench2DriveSingleSegmentLicenseAndAccessConditionsReview',
           next_direction_executed=False, default_files_modified=0, main_merged=False,
           original_I1_status='PilotNoGo',original_I1a_status='StartupDiagnosticNoGo',
           training_and_visual_safety='Unchanged_NotPassed')
    paths = set(REPO.glob('reports/m8i1b_*'))
    paths.update(REPO.glob('tools/m8i1b_*.py'))
    paths.update(REPO.glob('docs/M8I1B_*.md'))
    paths.update(p for p in OUT.rglob('*') if p.is_file())
    write(REPO / 'reports/m8i1b_release_manifest.json', dict(kind='OfflineForensicsReleaseSeal',
          base_commit=BASE, files=[asset(p) for p in sorted(paths)], self_excluded=True,
          receipt_excluded='reports/m8i1b_verification.json', server_launches=0,
          original_assets_modified=0, decision='StartupRootCauseUnresolved'))
    print(f'Sealed {len(paths)} files; decision B; no new runtime')


if __name__ == '__main__':
    main()
