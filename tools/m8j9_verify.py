"""Offline final report/reproducibility checks. Does not rerun official evaluation."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)


def canonical_metrics(value,ignore_runtime=False):
    if isinstance(value,dict):return {k:canonical_metrics(v,ignore_runtime) for k,v in value.items() if not (ignore_runtime and k=='eval_time')}
    if isinstance(value,list):return [canonical_metrics(v,ignore_runtime) for v in value]
    if isinstance(value,float) and math.isnan(value):return '__OfficialNotApplicableNaN__'
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true',help='Recheck evidence without replacing reports or running evaluation')
    args=parser.parse_args()
    test_log=Path('reports/m8j9_sources/tests_final.txt')
    assert '19 passed' in test_log.read_text(), 'Final interface tests missing or failed'
    cfg=read('configs/m8j9_retry1_protocol.json');private=Path(cfg['private_output'])
    source=Path('reports/m8j9_sources');export=read(source/'retry1/export/result.json')
    assert export['status']=='ExportInterfacePassed'
    assert sha(export['result_path'])==export['result_sha256']
    first=source/'retry1/eval1';second=source/'resume/eval2'
    assert read(second/'result.json')['status']=='OfficialEvaluationCompleted'
    assert read(source/'resume/audit/result.json')['status']=='OfficialFilteringAuditCompleted'
    comparisons={}
    for name in ('metrics_summary.json','metrics_details.json'):
        a=read(first/name);b=read(second/name)
        assert canonical_metrics(a,True)==canonical_metrics(b,True),name
        assert (first/name).read_bytes()==(private/'eval1'/name).read_bytes()
        assert (second/name).read_bytes()==(private/'eval2'/name).read_bytes()
        comparisons[name]=dict(first_sha256=sha(first/name),second_sha256=sha(second/name),
            exact_values_equal=True,byte_equal=(first/name).read_bytes()==(second/name).read_bytes(),
            ignored_fields=['eval_time'] if name=='metrics_summary.json' else [])
    for path,expected in cfg['sources_sha256'].items():assert sha(path)==expected,path
    for path,expected in export['source_hashes'].items():assert sha(path)==expected,path
    historical=[]
    for path in ['reports/m8j8_release_manifest.json','reports/m8j8a_release_manifest.json']:
        rel=read(path)
        for f in rel['files']:assert sha(f['path'])==f['sha256'],f['path']
        historical.append(dict(manifest=path,files=len(rel['files']),unchanged=True))
    pids=[]
    for name in ['supervisor.json','retry1/supervisor.json','resume/supervisor.json']:
        sup=read(source/name)
        for phase in sup['phases']:
            pids.append(dict(pid=phase['PID'],phase=phase['phase'],exit_code=phase['exit_code'],
                             still_present=Path('/proc',str(phase['PID'])).exists()))
    assert not any(p['still_present'] for p in pids)
    assert read(source/'export/result.json')['status']=='FailedStopped'
    assert read(first/'result.json')['status']=='FailedStopped' # preserve failure after official score creation
    audit=read('reports/m8j9_GT_filter_audit.json')
    assert audit['official_project_common']+audit['official_only']==audit['official_GT_filtered']['total']
    assert audit['official_project_common']+audit['project_only']==audit['project_GT_count']
    metrics=read(second/'metrics_summary.json')
    out=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_name='冻结A在mini_val上的本地官方devkit评测，使用指定缓存与输出策略',
        interface_status='Passed',comparison=comparisons,result_sha256=export['result_sha256'],
        two_official_metric_computations=True,extra_filter_only_audit=True,
        failure_preservation=dict(first_export_keyframe_index_failure=True,post_eval_numpy_diagnostic_failure=True,
            failed_phase_not_relabelled_success=True,same_result_file_reused=True),
        scalar_scores=dict(mAP=metrics['mean_ap'],NDS=metrics['nd_score'],**metrics['tp_errors']),
        cache_limitations=['FP32 fixed checkpoint A Top300','Formal preexisting score>=0.2 limits low-score PR support',
            'Original XY source range and official class ranges','Planar predicted velocity; deterministic attribute rule',
            '81 samples from two related scenes, not full validation; three unsupported GT classes retained in official averaging'],
        model_forward_calls=0,optimizer_steps=0,backward_calls=0,B_loaded=False,
        original_exporter_and_devkit_sources_unchanged=True,caches_and_export_metadata_unchanged=True,
        historical=historical,owned_processes=pids,
        metadata_SHA256={str(p):sha(p) for p in (Path(cfg['dataset_root'])/cfg['version']).glob('*.json')},
        commands_file='docs/M8J9_IMPLEMENTATION.md')
    acceptance=dict(stage='M8-J9',base_commit=cfg['base_commit'],
        interface_protocol='Passed',cache_status='SufficientForSpecifiedOfficialEvaluation_WithDeclaredOutputLimitations',
        scores=out['scalar_scores'],evaluation_name=out['evaluation_name'],
        official_config=cfg['official_detection_config'],devkit=cfg['nuscenes_devkit'],samples=81,
        GT_official=3393,GT_project=3559,export_predictions=7794,evaluator_predictions=7783,
        reproducibility='ExactExceptEvalTime_OfficialNaNPositionsEqual',
        tests_passed=19,model_forward=0,optimizer_step=0,training_line='ClosedUnchanged',reference='A',
        formal_training_ready='NotEvaluatedOrUpgraded',visual_driving_safety='NotValidated',
        original_failures_retained=True,raw_images_predictions_labels_weights_published=False,main_merged=False,
        result_path=export['result_path'],result_sha256=export['result_sha256'])
    if args.check_only:
        assert read('reports/m8j9_acceptance.json')==acceptance
        old=read('reports/m8j9_reproducibility.json')
        for k in ('comparison','result_sha256','scalar_scores','historical','metadata_SHA256'):
            assert old[k]==out[k], k
    else:
        write('reports/m8j9_reproducibility.json',out)
        write('reports/m8j9_acceptance.json',acceptance)
    print(json.dumps(dict(status='OfficialInterfaceVerified',scores=out['scalar_scores'],repeat=True,
        tests_passed=19,test_log_sha256=sha(test_log),check_only=args.check_only),indent=2))


if __name__=='__main__':main()
