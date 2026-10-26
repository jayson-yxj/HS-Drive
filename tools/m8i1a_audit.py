"""Static evidence plus bounded mock subprocess tests. No CARLA launch/connection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from tools.m8i1a_common import REPO, BASE, OLD_ROOT, PROTOCOL, config, asset, sha, persist, provenance

OUTPUT = Path('/root/gpufree-data/hs-drive-experiments/m8i1a-offline-evidence')


def report(name, kind, **fields):
    persist(REPO / f'reports/m8i1a_{name}.json', {**provenance(kind), **fields})


def main():
    OUTPUT.mkdir(exist_ok=False)
    cfg = config()
    install = Path(cfg['install'])
    historic = subprocess.run([sys.executable, '-m', 'tools.m8i1_verify'], cwd=REPO,
                              capture_output=True, text=True, check=True)
    persist(OUTPUT / 'historical_verification.json', json.loads(historic.stdout))
    commands = json.loads((OLD_ROOT / 'execution_commands.json').read_text())
    old_files = ['reports/m8i1_summary.md', 'reports/m8i1_acceptance.json',
                 'reports/m8i1_session_and_tick_audit.json', 'reports/m8i1_resource_and_cleanup.json',
                 'simulator/m8i1_session.py', 'tools/m8i1_launch.py', 'tools/m8i1_run_session.py',
                 'configs/m8i1/protocol.json', 'reports/m8i1_release_manifest.json']
    seal = [asset(REPO / p) for p in old_files]
    seal += [asset(p) for p in sorted(OLD_ROOT.rglob('*')) if p.is_file()]
    resources = []
    for item in json.loads((REPO / 'reports/m8i1_preflight_freeze.json').read_text())['files']:
        if item['path'].startswith(str(install)):
            assert sha(item['path']) == item['sha256']
            resources.append(asset(item['path']))
    launcher = install / 'CarlaUE4.sh'
    default = install / 'CarlaUE4/Config/DefaultEngine.ini'
    game = install / 'CarlaUE4/Config/DefaultGame.ini'
    old_script = REPO / 'scripts/start_carla.sh'
    user_config = Path('/home/ubuntu/.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor')
    ini = [default, game, *sorted(user_config.glob('*.ini'))]
    map_lines = []
    for path in ini:
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if any(token in line for token in ['GameDefaultMap=', 'ServerDefaultMap=', 'EditorStartupMap=', 'TransitionMap=', 'Town02']):
                map_lines.append(dict(path=str(path), line=number, text=line))
    map_arg_index = commands['server'].index('/Game/Carla/Maps/Town02')
    xodr = install / 'CarlaUE4/Content/Carla/Maps/OpenDrive/Town02.xodr'
    assert sha(xodr) == cfg['expected_xodr_sha256']
    report('static_launch_audit', 'ActualLocalReadOnlyFileAudit_NoServer', status='StaticReviewCompleted_RootCauseUnresolved',
           historical_files=seal, historical_verification=json.loads(historic.stdout),
           preserved_facts=dict(launches=1, client_server='0.9.15_CheckReached',
               failure='MapMismatch_NoReload', actual_map='NotRecorded', owner_ticks=0, actors=0,
               sensors=0, RGB=0, runner_returncode=0, final_cleanup='ConfirmedByOldOSProbe'),
           prior_server_argv=commands['server'], prior_runner_argv=commands['runner'],
           historical_actual_process_cwd=None,
           historical_cwd_evidence='Old Popen has no cwd argument and inherits caller CWD; no /proc record preserved. Not recoverable.',
           current_review_cwd=os.getcwd(), proposed_explicit_server_cwd=str(REPO),
           launcher=dict(**asset(launcher), text=launcher.read_text()),
           old_start_script=dict(**asset(old_script), text=old_script.read_text()),
           argument_position=dict(top_level_map_argv_index=map_arg_index,
               immediately_after=commands['server'][map_arg_index - 1],
               wrapper='Shipping binary receives CarlaUE4 followed by all supplied arguments; parser behavior untested'),
           resource_files=resources, expected_xodr=asset(xodr),
           inspected_configuration=[asset(p) for p in ini], relevant_configuration_lines=map_lines,
           log_map_load_evidence='NoneInPreservedServerLog',
           hypotheses=[dict(name='Positional map argument not applied/default config wins', status='Unproven',
                 evidence='Packaged defaults name Town10HD_Opt; this is NOT the old actual map'),
               dict(name='Working-directory-dependent behavior', status='Unproven',
                 evidence='Legacy script cd install; old pilot inherits unknown caller CWD'),
               dict(name='RPC version available before final world identity ready', status='Unproven',
                 evidence='Old supervisor waited only for version RPC; no old identity timeline')],
           planned_single_variable=cfg['single_variable'], no_map_argument_or_resource_change=True)
    trace_root = OUTPUT / 'synthetic_cases'
    env = os.environ.copy(); env['M8I1A_TEST_OUTPUT'] = str(trace_root)
    command = [sys.executable, '-m', 'pytest', 'tests/test_m8i1a_startup.py', '-q',
               '--junitxml=' + str(OUTPUT / 'tests.xml')]
    run = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    with (OUTPUT / 'pytest.txt').open('x') as f:
        f.write(run.stdout + run.stderr)
    suites = ET.parse(OUTPUT / 'tests.xml').getroot()
    cases = [dict(name=n.attrib['name'], classname=n.attrib.get('classname'),
                  seconds=float(n.attrib.get('time', 0)), passed=not any(n.find(tag) is not None for tag in ['failure','error','skipped']))
             for n in suites.iter('testcase')]
    rows = [json.loads(p.read_text()) for p in sorted(trace_root.glob('*/case_assertions.json'))]
    passed = run.returncode == 0 and all(c['passed'] for c in cases) and len(rows) == 10
    report('failure_injection_tests', 'ActualPythonSubprocessTests_MockCARLAOnly',
           status='OfflineInstrumentationPassed' if passed else 'OfflineInstrumentationFailed',
           command=command, returncode=run.returncode, test_count=len(cases), tests=cases,
           subprocess_cases=rows, files=[asset(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()],
           mocked_identity='All fake map/frame/episode values are synthetic; not actual server evidence',
           first_development_test_run='25 passed; final logged suite includes latest launcher source',
           actual_server_launches=0, actual_CARLA_connections=0)
    if not passed:
        raise RuntimeError('OfflineGateNotPassed')
    from tools.m8i1a_identity import EXIT
    report('exit_status_contract', 'OfflineContractValidatedWithRealPythonSubprocesses',
           status='StatusPropagationGo', runtime_validation='NotRun', exit_codes=EXIT,
           success_requires=['Runner rc=0', 'Session business success with same run_id',
              'Nonempty durable identity evidence', 'Runner cleanup ok', 'Final server OS cleanup ok',
              'No supervisor error or contradictory status'],
           contradiction='StatusPropagationNoGo plus nonzero supervisor exit',
           missing_signal_result='NoGo; keep already durable identity; never fabricate missing result',
           tested_propagation='Mock CARLA -> real Python runner exit -> reconciliation contract',
           untested='Full CARLA supervisor process orchestration awaits separate authorization')
    report('map_identity', 'NotRun_NoCARLAServer', status='StartupIdentityUnknown',
           expected_map=cfg['expected_map'], actual_map=None, world_episode=None,
           snapshot_frame=None, snapshot_timestamp=None, actual_xodr_sha256=None,
           packaged_xodr_sha256=sha(xodr), packaged_hash_is_not_runtime_identity=True,
           identity_persistence='OfflineValidated: exclusive write, file fsync and directory fsync before validation')
    report('startup_readiness', 'ProposedBoundedProtocol_NotRuntimeEvidence', status='NotRun',
           policy=cfg, matching_confirmation='Two passive matching identity observations with same episode/map/XODR/settings',
           mismatch='Persist actual identity and XODR, immediately reject; do not wait it away',
           unavailable='Persist null actual fields plus RPC errors; reject',
           no_automatic_load_or_reload=True, old_root_cause='Unresolved')
    report('runtime_probe', 'NotRun_AwaitingSeparateUserAuthorization', status='NotRun',
           server_launches=0, CARLA_connections=0, process_ids=[], actual_map=None,
           runtime_root_exists=Path(cfg['runtime_root']).exists(), authorization_received=False,
           output_is_offline_only=True)
    report('cleanup', 'NotRun_NoOwnedCARLAServer', status='NotRun',
           owned_server_processes=[], runtime_cleanup='NotRun',
           synthetic_process_cleanup='Passed_PythonSleepProcessOnly',
           historical_m8i1_cleanup='Preserved_ConfirmedByPreviousOSProbe',
           world_or_weather_mutations=0, no_claim_of_new_CleanupGo=True)
    report('acceptance', 'OfflineAcceptance_RuntimeAwaitingAuthorization',
           status='OfflineInstrumentationPassed', failure_propagation='StatusPropagationGo_Offline',
           actual_startup='NotRun', actual_map='StartupIdentityUnknown',
           root_cause='Unresolved', runtime_cleanup='NotRun',
           answers=dict(A_ActualMap='NotRun_NoActualIdentity',
                        B_PersistBeforeDecision='Passed_Offline_DurableWrite',
                        C_NonzeroPropagation='Passed_Offline_RealPythonSubprocess',
                        D_RootCause='Unresolved_ThreeUnprovenHypotheses',
                        E_ReadOnlyStartup='NotRun_SeparateApprovalRequired',
                        F_NoActorSensorTick='Confirmed_CodeAndOfflineExecution',
                        G_OwnCARLACleanup='NotRun_NoNewServer'),
           tests=len(cases), passed_tests=sum(c['passed'] for c in cases),
           runtime_authorized=False, next_TM_pilot_proposal_eligible=False,
           m8i1_status='PilotNoGo_Unchanged', training_readiness='NoGo_Unchanged',
           default_files_modified=0, main_merged=False)
    print(json.dumps(dict(status='OfflineInstrumentationPassed', tests=len(cases),
                          runtime='NotRun', output=str(OUTPUT)), indent=2))


if __name__ == '__main__':
    main()
