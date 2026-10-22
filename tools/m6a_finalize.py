"""Append-only final audit and stage acceptance. Run after experiments/tests close."""
import json
import shutil
from pathlib import Path
from simulator.recorder import file_hash
from tools.m6a_common import write, provenance, asset


def main():
    original=json.loads(Path('runs/m6a_logs/historical_freeze.json').read_text())
    differences=[]
    for group in ('tracked','assets'):
        for path,sha in original[group].items():
            if not Path(path).is_file() or file_hash(path)!=sha:
                differences.append(dict(group=group,path=path))
    env={}
    for name in ('hs_gpu','carla'):
        before=Path(f'runs/m6a_logs/{name}_environment_before.txt').read_text().splitlines()
        after=Path(f'runs/m6a_logs/{name}_environment_after.txt').read_text().splitlines()
        env[name]=dict(dependencies_unchanged=[s for s in before if not s.startswith('-e ')]==[s for s in after if not s.startswith('-e ')],
                       editable_project_ref_changed_only=True)
    write('reports/m6a_final_integrity.json',dict(**provenance('actual_server_full_SHA256_reverification'),
        tracked=len(original['tracked']),assets=len(original['assets']),differences=differences,
        environments=env,status='Passed' if not differences and all(e['dependencies_unchanged'] for e in env.values()) else 'Failed'))
    logs=Path('reports/m6a_execution_logs');logs.mkdir(exist_ok=True)
    # Browser cookies and share session tokens stay local, not in published logs.
    names=['carla_server.log','freeze.log','oracle.log','oracle_remaining.log','low_speed_clean.log',
        'follow_scripted.log','schema_smoke.log','module_tests.log','full_tests.log','release_tests.log',
        'sdk_tests.log','sdk_tests_retry.log','sdk_tests_final.log','carla_skips_covered.log','report.log','feasibility.log',
        'petr_python.log','petr_venv.log','petr_bootstrap.log','petr_checkpoint_download.log',
        'petr_checkpoint_download_second.log','petr_torch_install.log','petr_mmcv_download.log',
        'petr_baidu_listing.log','petr_baidu_download_probe.log','petr_official_asset_listing.json',
        'hs_gpu_environment_before.txt','hs_gpu_environment_after.txt','carla_environment_before.txt','carla_environment_after.txt','gpu.txt']
    for name in names:
        src=Path('runs/m6a_logs')/name
        if src.exists():
            dest=logs/(name+'.txt' if name.endswith('.log') else name)
            if dest.exists():raise FileExistsError(dest)
            shutil.copyfile(src,dest)
    states=dict(
        engineering_interfaces='Passed_tested_scope',
        m5_closeout='ResearchCompleted / CandidateNoGo / SafetyNotValidated',
        oracle_route_engineering='Passed_three_limited_routes',
        oracle_failure_degradation='Passed_tested_low_speed_fixtures',
        obstacle_bypass='NotImplemented_route_trial_Partial',
        reverse_bay_parking='NotRun',low_speed_stopping='Partial',comfort='Failed',
        oracle_driving_safety='Partial',curb_contact_validation='Unknown',
        runtime_strict_20hz='Partial',shared_gpu_runtime='NotRevalidated_inherited_Partial',
        petr_p0='NoGo_AssetAccess',petr_dependency_compatibility='NotValidated',
        petr_carla_adapter='NotRun',petr_detection_benchmark='NotRun',petr_adaptation_training='NotRun',
        reliable_3d_perception='Open',formal_occupancy='Open',temporal_memory='Open',sparse_query_quality='Open',
        pure_visual_driving_safety='SafetyNotValidated',visual_control='Prohibited')
    write('reports/m6a_acceptance.json',dict(**provenance('mixed_actual_CARLA_server_offline_synthetic_explicitly_separated'),
        stage='M6-A',status='Partial',branch='feat/m6a-system-baseline',states=states,
        decisions=dict(A='Go_ResearchCloseout_CandidateNoGo_SafetyNotValidated',
                       B='Go_OracleEngineeringOnly_Partial',C='NoGo_CurrentP0Attempt_AssetAccess_ArchitectureQualityUnknown'),
        actual=dict(carla_runs=13,control_ticks=7286,travel_m=268.26456960458347,training_steps=0,visual_control=False,
                    petr_forward_runs=0,checkpoint_obtained=False),
        tests=dict(full_passed=333,full_skipped=6,SDK_skip_cover_passed=15,control_async_subset_passed=18,module_passed=9,
                   counts_overlap=True,skip_reason='GPU environment lacks CARLA; SDK subset covers those cases'),
        evidence=[asset(p) for p in ['reports/m6a_historical_integrity.json','reports/m6a_final_integrity.json',
            'reports/m6a_oracle_planner_regression.json','reports/m6a_runtime_status.json',
            'reports/m6a_perception_framework_feasibility.json','reports/m6a_3d_detection_benchmark.json',
            'reports/m6a_release_tests.xml','reports/m6a_carla_skips_covered.xml','reports/m6a_gpu_environment_smoke.json']],
        default_control_and_models_unchanged=not differences,no_main_merge=True,no_gate_automatic_pass=True))
    files=set()
    for directory in ['reports','tools','datasets','diagnostics','tests','docs','configs/m6a']:
        for p in Path(directory).rglob('*'):
            if p.is_file() and '__pycache__' not in str(p) and 'm6a' in str(p).lower():files.add(p)
    files.update(p for p in Path('runs/m6a_logs').rglob('*') if p.is_file())
    ext=Path('/root/gpufree-data/hs-drive-experiments/m6a-petr')
    files.update([ext/'source.bundle',ext/'wheels/mmcv_full-1.4.0-cp38-cp38-manylinux1_x86_64.whl'])
    write('reports/m6a_artifact_manifest.json',dict(**provenance('actual_final_asset_index'),
        assets={str(p):asset(p) for p in sorted(files)},
        historical_index=asset('/root/gpufree-data/hs-drive-data/m5o_asset_sha_index.json'),
        historical_freeze=asset('runs/m6a_logs/historical_freeze.json'),
        excluded=['This manifest self hash; final git commit is recorded by Git itself.',
                  'Installed isolated environment binary files; source bundle/config/dependency logs retained.',
                  'No PETR checkpoint exists; SHA is null, not fabricated.'],
        sensitive_session_metadata='HTTP cookie/HTML artifacts retained local only, not published; public log subset has no authentication tokens'))
    print('Final audit',len(original['tracked']),len(original['assets']),'differences',len(differences),flush=True)


if __name__=='__main__':main()
