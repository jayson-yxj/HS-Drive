"""Seal all M8-E evidence, including intermediate reports and failed sources."""
import json
import shutil
from pathlib import Path
from tools.m8e_common import ROOT,LOG,provenance,asset,write,sha
from tools.m8e_verify import verify


def main():
    assert not Path('reports/m8e_release_manifest.json').exists()
    baseline=[]
    for name in ['m8b_release_manifest','m8b_evaluation_reference_audit',
                 'm8c_release_manifest','m8d_release_manifest']:
        p=Path('reports')/(name+'.json')
        n=verify(p)
        baseline.append(dict(manifest=asset(p),files=n,unchanged=True))
        print(name,n,'unchanged',flush=True)
    # Preserve in-session provisional decisions before adding final findings.
    draft=ROOT/'provisional_reports';draft.mkdir(exist_ok=False)
    def update(name,mutator):
        p=Path('reports')/('m8e_'+name+'.json')
        shutil.copy2(p,draft/p.name)
        d=json.loads(p.read_text());mutator(d)
        p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    def acceptance(d):
        log=LOG/'source_time_tests.txt'
        assert 'Ran 18 tests' in log.read_text() and '\nOK\n' in log.read_text()
        d['test_results'].pop('final_tests',None)
        d['test_results']['source_time_tests']=dict(count=18,log=asset(log))
        d['total_unit_and_regression_tests']=52
        d['historical_manifests_reverified']=baseline
        d['finalization']=provenance('FinalEvidenceSeal')
    update('acceptance',acceptance)
    def readiness(d):
        d['unresolved'].extend([
            'Reserved final target only9.9007m from historical target; zero valid independent final groups.',
            'Implausible off-road/floating controlled placements; no naturalistic training subset qualified.',
            'Far front-left support0 and no >40m target; only21/30 unique far render links.'])
        d['independent_final_groups']=0
        d['annotation_review']=asset('reports/m8e_annotation_review.json')
        d['independence_exception']=asset('reports/m8e_heldout_independence_exception.json')
    update('training_readiness',readiness)
    def failures(d):
        d['implementation_issues'].append(
            'Final defensive source-frame guard added after review: all32 actual sidecars passed, no grades altered. '
            'Executed prior source copies hash-match logs/audit_executed_v2.py.txt and reference_executed_v2.py.txt.')
    update('failure_ledger',failures)
    write('reports/m8e_final_validation.json',dict(**provenance('ActualFinalArtifactValidation'),
        tests_passed=52,baseline_reverified=baseline,
        source_snapshot_reconstruction='Executed source copies independently matched recorded SHA.',
        stage_outputs_required=[
            str(p) for p in sorted(Path('reports').glob('m8e_*.json'))],
        active_model_calls=0,
        cleanup=asset('reports/m8e_runtime_cleanup.json'),
        carla_shutdown='Own CARLA PID57945 terminated with SIGTERM after restored async world and zero sensors/vehicles/walkers. Server process exited143.',
        new_GPU_compute='CARLA rendering only. Input contract and evaluator run on CPU. No PETR model execution.',
        intermediate_report_copies=str(draft)))
    paths=[*Path('configs/m8e').rglob('*'),*Path('reports').glob('m8e_*.json'),
           *Path('reports').glob('m8e_*.md'),*LOG.rglob('*'),
           *Path('reports/m8e_figures').rglob('*'),*Path('tools').glob('m8e_*.py'),
           *Path('datasets').glob('m8e_*.py'),*Path('tests').glob('test_m8e*.py'),
           *Path('docs').glob('M8E*.md'),*ROOT.rglob('*')]
    files=[asset(p) for p in sorted(set(paths)) if p.is_file()
           and p.name!='m8e_release_manifest.json' and '__pycache__' not in p.parts]
    write('reports/m8e_release_manifest.json',dict(**provenance('FinalSourceAndEvidenceManifest'),
        files=files,historical_manifests=baseline,
        total_bytes=sum(a['bytes'] for a in files),
        contract='Final Git commit seals this manifest. Execution parent plus per-file source SHA recorded. Manifest excludes own hash. Raw data external; retain server artifact root.',
        checkpoint='No checkpoint loaded, created or modified; M8-B/C/D manifests reverified.'))
    print('SEALED',len(files),'files',sum(a['bytes'] for a in files),'bytes',flush=True)


if __name__=='__main__':main()
