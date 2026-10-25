"""Seal independent decisions and preserve failed execution evidence."""
import json
from pathlib import Path
from tools.m8d_common import *

def main():
    baseline=[]
    for name in ['m8b_release_manifest','m8b_evaluation_reference_audit','m8c_release_manifest']:
        path=Path('reports')/(name+'.json');data=json.loads(path.read_text())
        assert all(sha(a['path'])==a['sha256'] for a in data['files'])
        baseline.append(dict(manifest=asset(path),files=len(data['files']),unchanged=True))
    tests={}
    for name,count in [('final_new_tests',16),('m8b_regression',18),('m8c_regression',13)]:
        content=(LOG/(name+'.txt')).read_text()
        assert 'Ran %d tests'%count in content and '\nOK\n' in content
        tests[name]=dict(count=count,log=asset(LOG/(name+'.txt')))
    logs=[json.loads(p.read_text()) for p in LOG.glob('*.json')]
    failures=[r for r in logs if r.get('returncode',0)!=0]
    write('reports/m8d_failure_ledger.json',dict(**provenance('ExecutedFailureAndScopeLedger'),
        process_failures=failures,
        fixes=[dict(id='small12_shape_mutation',reason='Official extract_feat calls squeeze_ in place. Warmup input reused at wrong rank.',
                    fix='Every model.extract_feat call receives explicit view(1,6,3,512,1408). Official source and parameters unchanged.',
                    retained_source=asset(LOG/'gpu_failed_v1.py.txt'),retained_output=str(ROOT/'small_failed_v1'),
                    rerun=asset(LOG/'small12_v2.json'),quality_metrics_from_failed_run=False)],
        unresolved=['TrainingPKLOrderNotVerified','FarFieldIdentityInsufficient','AnnotationSupportInsufficient',
                    'CrossSessionRuntimeDifferenceUnattributed','NoIndependentFinalLocations'],
        not_run=['training','fine_tuning','second_B_order','height_search','crop_search','risk_threshold_search',
                 'new_CARLA_collection','new_vehicle_closed_loop','visual_control','shared_GPU_runtime'],
        historical_failures_preserved=True))
    reports=['ablation_protocol','source_evidence','permutation_contract','gpu_output_difference',
             'paired_detection_quality','paired_identity_support','paired_localization','score_candidate_burden',
             'latency_resource','paired_supplement','visualization_manifest','failure_ledger']
    full=json.loads((ROOT/'full/run.json').read_text())
    assert all(x['box_exact'] and x['score_exact'] and x['labels_exact'] for x in full['replay_A'])
    quality=json.loads(Path('reports/m8d_paired_detection_quality.json').read_text())
    assert quality['natural']['A']['TP']==45 and quality['natural']['B']['TP']==45
    write('reports/m8d_acceptance.json',dict(**provenance('ActualGPU_OfflineCARLARecordedEvaluation_SyntheticContracts'),
        overall='DiagnosticCompleted_QualityPartial',
        decisions=dict(A_TrainingInputOrder='TrainingPKLOrderNotVerified',
                       B_Permutation='PermutationContractPassed',
                       C_Encoding='OrderEncodingChanged',
                       C_ModelResponse='OrderEffectOnPredictionsSupported',
                       D_Quality='OrderChangeNoQualityGain',
                       E_Consistency='NoPositiveDetectionTrendAcross38Positions',
                       F_Identity='FarFieldIdentityInsufficient',
                       F_Annotations='AnnotationSupportInsufficient',
                       G_Training='TrainingReadinessNoGo',
                       ModelAdoption='NotApproved_DefaultAUnchanged',
                       Runtime='OnlineNoGo_HistoricalPartialUnchanged',
                       Occupancy='NotValidated',Perception='NotValidated',Safety='NotValidated'),
        execution=dict(actual_GPU='RTX4090',contract_frames=192,contract_cameras=1152,
                       hook_pairs=12,full_pairs=192,successful_forward_calls=412,
                       full_raw_candidates_per_variant=57600,all_A_replay_bitwise=True,
                       weights_unchanged=full['weights_unchanged'],training_steps_executed=0,
                       new_CARLA_closed_loops=0,failed_trial='Retained separately; not counted as successful'),
        quality=dict(natural2m_A=45,natural2m_B=45,GT=155,groups=38,
                     transitions=quality['strata']['all']['transitions'],far_A=1,far_B=1,far_GT=83,
                     strict_identity_A=6,strict_identity_B=5,common_identity=5,
                     common_XY_P95_A=5.08250143760131,common_XY_P95_B=6.034157372830014,
                     new_independent_final_groups=0),
        tests=tests,total_unit_and_regression_tests=47,baseline_verified=baseline,
        reports={name:asset('reports/m8d_'+name+'.json') for name in reports},
        inference_source='OfficialPETR_Frozen_OfflineShadow; GT only in separate evaluator',
        next_priority='Propose bounded identity/background/data support improvement; no order search or training authorization'))
    paths=[*Path('configs/m8d').rglob('*'),*Path('reports').glob('m8d_*.json'),
           *Path('reports').glob('m8d_*.md'),*Path('reports/m8d_logs').rglob('*'),
           *Path('reports/m8d_figures').rglob('*'),*Path('tools').glob('m8d_*.py'),
           *Path('datasets').glob('m8d_*.py'),*Path('diagnostics').glob('m8d_*.py'),
           *Path('tests').glob('test_m8d*.py'),*Path('docs').glob('M8D*.md'),*ROOT.rglob('*')]
    files=[asset(p) for p in sorted(set(paths)) if p.is_file() and p.name!='m8d_release_manifest.json']
    write('reports/m8d_release_manifest.json',dict(**provenance('FinalSourceAndEvidenceManifest'),files=files,
        historical_manifests=baseline,weights=asset(RUNTIME/'artifacts/safe_extract/model_tensors.npz'),
        checkpoint_sha256_original=CFG['checkpoint_sha256'],
        commit_contract='Final Git commit seals this manifest; execution parent plus exact source SHA recorded in each run. Manifest excludes own hash.'))
    print('Sealed',len(files),'new assets; history unchanged')
if __name__=='__main__':main()
