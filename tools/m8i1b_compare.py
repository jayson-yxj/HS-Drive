"""Evidence-tiered comparison of existing runs; no simulator or external access."""
from tools.m8i1b_forensics import *


def fact(value, tier, sources, limitation=None):
    return dict(value=value, evidence_tier=tier, sources=sources, limitation=limitation)


def main():
    old_cmd = read(I1 / 'execution_commands.json')
    new_cmd = read(I1A / 'execution_commands.json')
    assert old_cmd['server'] == new_cmd['server']
    h1_pre = read(REPO / 'reports/m8h1_logs/preflight.json')
    h1_contract = read(REPO / 'reports/m8h1_shifted_contract.json')
    h1_environment = read(H1 / 'environment_inventory.json')
    old_super = read(I1 / 'supervisor_result.json')
    new_super = read(I1A / 'supervisor_result.json')
    process = read(I1A / 'server_process_binary.json')['members']
    during = read(APPROVAL / 'during_startup_OS_probe.json')
    config = read(REPO / 'configs/m8i1a/protocol.json')
    h1source = ['tools/m8h1_preflight.py', 'scripts/start_carla.sh', 'docs/M8H1_IMPLEMENTATION.md']
    h1_start = ['runuser', '-u', 'ubuntu', '--', 'bash', 'scripts/start_carla.sh', str(INSTALL), '2100', 'Epic']
    h1_effective = [str(INSTALL / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'),
                    'CarlaUE4', '-RenderOffScreen', '-quality-level=Epic', '-nosound',
                    '-carla-rpc-port=2100', '-stdout', '-FullStdOutLogOutput']
    common_env = dict(DISPLAY='unset', VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    fields = {}
    def add(key, h, i, a): fields[key] = {'M8-H1': h, 'M8-I1': i, 'M8-I1a': a}
    add('full_argv',
        fact(dict(documented_start=h1_start, reconstructed_binary=h1_effective, actual_proc_argv=None),
             'DocumentedCommandPlusSourceReconstruction', h1source, 'No per-process argv snapshot; not a recorded exact command execution'),
        fact(old_cmd, 'ActualRecordedInvocation', [str(I1 / 'execution_commands.json')], 'Shipping /proc argv not captured'),
        fact(dict(invocations=new_cmd, actual_processes=[{k:p[k] for k in ['pid','pgid','cmdline']} for p in process]),
             'ActualRecordedInvocationAndProc', [str(I1A / 'execution_commands.json'),str(I1A / 'server_process_binary.json')]))
    add('cwd',
        fact(dict(script_cwd=str(INSTALL), actual_binary_cwd=None), 'SourceInference', ['scripts/start_carla.sh'], 'Documented launch cd; historical binary cwd unavailable'),
        fact(None, 'NotRecorded', ['tools/m8i1_launch.py'], 'Popen inherited caller cwd; current cwd cannot fill historical actual'),
        fact([dict(pid=p['pid'],cwd=p['cwd']) for p in process], 'ActualProcReadback', [str(I1A / 'server_process_binary.json')]))
    add('execution_user',
        fact('ubuntu', 'DocumentedCommand', ['docs/M8H1_IMPLEMENTATION.md'], 'No per-process UID snapshot'),
        fact('runuser -u ubuntu', 'ActualRecordedInvocation', [str(I1 / 'execution_commands.json')], 'No actual process UID snapshot'),
        fact('runuser -u ubuntu', 'ActualRecordedInvocationAndProc', [str(I1A / 'execution_commands.json')], 'Process UID not captured; do not claim full credentials evidence'))
    add('environment',
        fact(common_env, 'SourceInference', ['scripts/start_carla.sh'], 'VK export conditional on file existence; full original environ absent'),
        fact(common_env, 'ActualExplicitARGVEnvironment', [str(I1 / 'execution_commands.json')], 'Inherited environment not snapshotted'),
        fact(common_env, 'ActualExplicitARGVEnvironment', [str(I1A / 'execution_commands.json')], 'Inherited environment not snapshotted'))
    add('map_parameter_and_loading',
        fact(dict(cli_map=None, conditional_load_world='Town02', final_observed_map=h1_environment['map'],
                  initial_map=None, actual_load_world_branch_taken=None),
             'SourcePolicyPlusActualFinalReadback', ['tools/m8h1_preflight.py',str(H1/'environment_inventory.json')],
             'H1 is not a no-reload startup control; final Town02 cannot prove initial Town02'),
        fact(dict(cli_map='/Game/Carla/Maps/Town02', reload_forbidden=True, actual_map=None, map_check='Rejected'),
             'ActualFailureAndExecutedSource', [str(I1/'session_result.json'),'simulator/m8i1_session.py']),
        fact(dict(cli_map='/Game/Carla/Maps/Town02', reload_forbidden=True, actual_map=None, map_check='NotReached'),
             'ActualFailureAndExecutedSource', [str(I1A/'session_result.json'),'tools/m8i1a_identity.py']))
    add('server_start_complete_criterion',
        fact('Preflight reads world, optionally loads Town02, asserts versions and XODR before downstream work',
             'SourcePolicyAndSuccessfulPreflight', ['tools/m8h1_preflight.py','reports/m8h1_logs/preflight.json'], 'Initial server-ready wait not recorded'),
        fact('Supervisor first successful version RPC, then new Runner checks versions/world/map',
             'SourcePolicyWithActualVersionCheckReached', ['tools/m8i1_launch.py','simulator/m8i1_session.py',str(I1/'session_result.json')]),
        fact('One persistent Client attempts version RPC; only then capture/validate identity twice',
             'SourcePolicyWithActualVersionFailure', ['tools/m8i1a_identity.py',str(I1A/'session_result.json')]))
    add('python_client_and_server_version',
        fact(dict(executable='.venv/bin/python', client='0.9.15',server=h1_contract['server']),
             'ActualPreflightSuccessThroughVersionAssertion', ['reports/m8h1_logs/preflight.json','tools/m8h1_preflight.py','reports/m8h1_shifted_contract.json']),
        fact(dict(executable=old_cmd['runner'][0],client='0.9.15',server='0.9.15'),
             'VersionAssertionReachedBeforeActualMapRejection', ['simulator/m8i1_session.py',str(I1/'session_result.json')]),
        fact(dict(executable=new_cmd['runner'][0],expected='0.9.15',actual_client_rpc_record=None,actual_server_rpc_record=None),
             'ActualTimeout_NoRecordedVersion', [str(I1A/'session_result.json')], 'Installed version or shared path cannot substitute for missing per-run RPC result'))
    add('RPC_port',
        fact(2100,'DocumentedCommandAndExecutedSource',h1source),
        fact(2100,'ActualARGVAndExecutedSource',[str(I1/'execution_commands.json')]),
        fact(2100,'ActualARGVProcAndTCPProbe',[str(I1A/'execution_commands.json'),str(APPROVAL/'during_startup_OS_probe.json')]))
    add('single_RPC_timeout_seconds',
        fact(60,'ExecutedSource',['tools/m8h1_preflight.py'], 'Source setting, not measured latency'),
        fact(dict(supervisor=2,runner=30),'ExecutedSource',['tools/m8i1_launch.py','simulator/m8i1_session.py']),
        fact(2,'ExecutedSourceAnd31TimeoutErrors',['tools/m8i1a_identity.py',str(I1A/'connection_001.json'),str(I1A/'connection_031.json')], '31 failures do not prove timeout length is the cause'))
    add('client_retry_object_lifecycle',
        fact('One Client per preflight; no readiness retry loop in preflight', 'ExecutedSource', ['tools/m8h1_preflight.py']),
        fact('New Client inside each supervisor connection iteration; independent Runner Client', 'ExecutedSource', ['tools/m8i1_launch.py']),
        fact('One Client reused across all31 attempts', 'ExecutedSourceWithActualAttemptCount', ['tools/m8i1a_identity.py'], 'No client internal connection state trace; cannot infer poisoned retries'))
    add('total_readiness_wait_and_actual_elapsed',
        fact(dict(startup_wait=None,preflight_elapsed_s=h1_pre['elapsed_s']),
             'UnknownStartupWait_ActualPreflightDuration',['reports/m8h1_logs/preflight.json'], '6.28s preflight is not server boot time'),
        fact(dict(startup_cap_s=180,total_cap_s=1800,actual_supervised_wall_s=old_super['elapsed_s']),
             'SourceCapsAndActualSupervisorTimer',['tools/m8i1_launch.py',str(I1/'supervisor_result.json')]),
        fact(dict(connect_cap_s=90,identity_cap_s=30,total_cap_s=180,actual_supervised_wall_s=new_super['wall_elapsed_s']),
             'SourceCapsAndActualSupervisorTimer',['configs/m8i1a/protocol.json',str(I1A/'supervisor_result.json')], 'Identity window never reached; connection deadline checked after RPC returns'))
    add('GPU_and_process_resource_evidence',
        fact(dict(own_pid=157737,per_run_gpu=None), 'ActualCleanupReport_NoResourceTrace',['reports/m8h1_runtime_cleanup.json']),
        fact(dict(server_pid=35544,runner_pid=36560,sampled_gpu_max_MiB=5840),
             'ActualSupervisorAndSampledGPU',[str(I1/'supervisor_result.json'),str(I1/'resource_samples.jsonl')], 'Not complete GPU peak; no CPU trace'),
        fact(dict(processes=[dict(pid=p['pid'],pgid=p['pgid']) for p in process],gpu_sample=during['gpu'],
                  tcp=during['ports']['listening_ports']), 'ActualProcAndOSProbe',[str(APPROVAL/'during_startup_OS_probe.json')], 'GPU99% does not prove deadlock, OOM or render error'))
    add('shutdown',
        fact('Cleanup readback then own PID157737 SIGTERM', 'HistoricalRecordedClaim', ['reports/m8h1_final_validation.json','reports/m8h1_runtime_cleanup.json'], 'No raw post-exit process-group trace; final map Town04 is negative-control map, not startup map'),
        fact(dict(supervisor=old_super['server'],post='groups absent, ports closed'),
             'ActualSupervisorAndLaterOSProbe',[str(I1/'supervisor_result.json'),str(I1/'post_shutdown_process_probe.json')]),
        fact(dict(server_rc=-15,runner_rc=10,supervisor_rc=1,cleanup='21 probes, final group/port absence'),
             'ActualSupervisorAndIndependentOSProbe',[str(I1A/'supervisor_result.json'),str(I1A/'cleanup.json'),str(APPROVAL/'post_shutdown_OS_probe.json')]))
    evidence_sources=set()
    for values in fields.values():
        for f in values.values():
            evidence_sources.update(f['sources'])
    report('success_failure_launch_diff', comparison=fields,
           evidence_legend=['ActualRecorded', 'ExecutedSource', 'DocumentedCommand', 'SourceInference', 'NotRecorded'],
           I1_I1a_server_argv_identical=True,
           I1_I1a_have_different_failure_stages=True,
           observed_differences_are_not_causal_isolation=True,
           evidence_files=[asset(Path(p) if Path(p).is_absolute() else REPO/p) for p in sorted(evidence_sources)],
           unsupported_claims=['H1 booted directly into Town02','I1 actually loaded Town10HD_Opt',
              'I1a failed because timeout2s is too short','I1 and I1a share a proven map-load root cause'])
    chains={
      'M8-H1': [
        fact('Supported', 'HistoricalActualPID', ['reports/m8h1_runtime_cleanup.json']),
        fact('InferredFromActualCARLAWorldOperations', 'FunctionalEvidence_NoProcSnapshot', ['reports/m8h1_shifted_contract.json']),
        fact('InferredFromSuccessfulRPC', 'FunctionalEvidence_NoTCPProbe', ['reports/m8h1_shifted_contract.json']),
        fact('Supported', 'ActualPreflightSuccessThroughVersionAssertion', ['tools/m8h1_preflight.py','reports/m8h1_logs/preflight.json']),
        fact(dict(status='SupportedAfterOptionalMapLoad',map=h1_environment['map'],episode=h1_environment['world_episode'],frame=h1_environment['frame']),
             'ActualInventory',[str(H1/'environment_inventory.json')]),
        fact(dict(status='SupportedAfterOptionalMapLoad',sha256=h1_contract['map_xodr']['sha256']),
             'ActualAPIXODR',['reports/m8h1_shifted_contract.json',str(H1/'Town02.xodr')])],
      'M8-I1': [
        fact('Supported','ActualSupervisor',[str(I1/'supervisor_result.json')]),
        fact('InferredFromSuccessfulVersionAndWorldRPC','FunctionalEvidence_NoProcSnapshot',[str(I1/'session_result.json')]),
        fact('InferredFromSuccessfulRPC','FunctionalEvidence_NoTCPReadyProbe',[str(I1/'session_result.json')]),
        fact('Supported','ActualCodePathBeforeMapRejection',['simulator/m8i1_session.py',str(I1/'session_result.json')]),
        fact('RPCReadReached_ButActualNameNotPersisted','ActualMapRejectWithObservabilityGap',[str(I1/'session_result.json')]),
        fact('NotReached','ExecutedSourceAfterMapCheck',['simulator/m8i1_session.py'])],
      'M8-I1a': [
        fact('Supported','ActualProc',[str(I1A/'server_process_initial.json')]),
        fact('Supported','ActualProc',[str(I1A/'server_process_binary.json')]),
        fact('Supported','ActualTCPProbe',[str(APPROVAL/'during_startup_OS_probe.json')]),
        fact('NotAchieved_31Timeouts','ActualRPCErrors',[str(I1A/'session_result.json')]),
        fact('NotReached','ActualBusinessFailure',[str(I1A/'session_result.json')]),
        fact('NotReached','ActualBusinessFailure',[str(I1A/'session_result.json')])]
    }
    states=['ServerProcessCreated','ShippingProcessAlive','TCPListening','CARLAVersionRPCReady',
            'WorldMapIdentityAvailable','MapXODRVerified']
    report('rpc_readiness_evidence',states=states,chains={k:dict(zip(states,v)) for k,v in chains.items()},
           I1a_highest_supported_state='TCPListening', no_TCP_to_RPC_promotion=True,
           two_failure_conclusions=dict(M8_I1='VersionRPCSucceeded_MapRejected_ActualNameMissing',
                                        M8_I1a='31VersionRPCTimeouts_NoWorldMapRead'),
           no_new_runtime_evidence=True, forbidden_unification='Both failures are map-load failures')
    print('Wrote evidence-tiered comparison and readiness chains; no simulator calls')


if __name__ == '__main__':
    main()
