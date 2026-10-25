"""Finalize M8-E decisions using immutable actual evidence; no CARLA/model calls."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tools.m8e_common import ROOT, LOG, provenance, write, asset


def read(name):
    return json.loads(Path('reports/m8e_' + name + '.json').read_text())


def main():
    source = provenance('ActualCARLAData_OfflineAudits_AssistantReview_NoModelRun')
    isolation = json.loads((LOG/'input_isolation.txt').read_text())
    assert isolation['packets'] == 32 and isolation['images'] == 192
    assert len(isolation['blocked_canaries']) == 3
    write('reports/m8e_input_isolation.json', dict(**source, result=isolation,
          execution=asset(LOG/'input_isolation.json'), model_inference_test='NotRun',
          guarantee='Strict packet whitelist, exact RGB path allowlist, Python open guard, network namespace. Not hostile native-code sandbox.'))
    figures = read('visualization_manifest')['files']
    review_notes = {
        'm8e_development_00': 'Road barriers/terrain occlude side targets; sparse far pixels. No complete background identity.',
        'm8e_development_01': 'Guardrail and vegetation obscure side targets; green boxes mean accepted render link only.',
        'm8e_development_02': 'Visible car and cone over water/off-road; apparent suspension. Ground support not established.',
        'm8e_validation_00': 'Covered parking walls heavily occlude views; some placed vehicles appear unsupported. Limited natural-road relevance.',
        'm8e_validation_01': 'Open junction and small cone/pedestrian pixels; low pixel support is not safety quality.',
        'm8e_calibration_00': 'Road barrier occlusion; far cone appears elevated, no audited ground-contact evidence.',
        'm8e_calibration_01': 'Close car spans front/front-left; far actors behind roadside structures.',
        'm8e_held_out_final_00': 'Urban objects visible, background furniture unlinked. Annotation only; historical target proximity invalidates independent final status.'}
    review = []
    for a in figures:
        stem = Path(a['path']).stem
        review.append(dict(figure=a, review_type='AssistantVisualReview',
            initial_choice='Retain render-link labels; do not approve naturalistic training support',
            observation=review_notes.get(stem, 'Four-phase evidence for every qualified far object; fixed present ROI can miss shifted target. Small pixels and implausible placements remain visible.'),
            adjudication='No independent human reviewer; no identity-grade override',
            model_predictions_shown=False))
    write('reports/m8e_annotation_review.json', dict(**source, records=review,
        figures_inspected=12, far_objects_in_figures=21, independent_human_reviewers=0,
        human_double_verified=False, manual_identity_changes=0,
        initial_conflicts_retained=True, raw_development_disagreements=asset('reports/m8e_development_raw_probe.json'),
        historical_154_review='Machine re-audit of frozen evidence, retaining previous visual records; no claim of fresh double review of all154.',
        disputed_scene_examples=[dict(group='m8e_development_02',entity='Actor:717',issue='Apparent floating car over water',figure='far_controls_01.png row0'),
            dict(group='m8e_development_02',entity='Actor:715',issue='Cone over water/off-road',figure='far_controls_01.png row1'),
            dict(group='m8e_calibration_00',entity='Actor:1022',issue='Apparently elevated cone; physical support unverified',figure='far_controls_02.png row1')],
        naturalistic_training_support='NotQualified; ray-based ground placement/fallback lacked per-target contact logs. No contact/road validity guarantee.',
        counterfactual_identity=asset('reports/m8e_counterfactual_identity.json')))
    hist = read('historical_evaluator')
    def count(rows):
        linked = sum(r['grade'].startswith('RenderLinked') for r in rows)
        return dict(recorded_GT=len(rows), supported_associations=linked,
                    unsupported_localization=len(rows)-linked,
                    support_fraction=linked/len(rows) if rows else None,
                    grades=dict(Counter(r['grade'] for r in rows)),
                    meaning='Conditional association support, not complete physical detection recall')
    strata = {'all': count(hist['rows'])}
    for lo, hi in [(0,10),(10,20),(20,40),(40,1000)]:
        strata['range_%d_%d'%(lo,hi)] = count([r for r in hist['rows'] if lo<=r['range_m']<hi])
    for cam in ['front','front_left','front_right','rear_left','rear_right','rear']:
        strata[cam] = count([r for r in hist['rows'] if any(v['camera']==cam for v in r['views'])])
    errors = [r['metric_error'] for r in hist['rows'] if r['metric_error'] is not None]
    absolute = {}
    for key in ['optical_depth_signed_m','optical_lateral_signed_m','radial_signed_m','center_Z_signed_m']:
        a=np.abs([e[key] for e in errors])
        absolute[key.replace('signed','absolute')]=dict(n=len(a),mean=float(a.mean()),
            p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),p95=float(np.percentile(a,95)),max=float(a.max()))
    write('reports/m8e_evaluator_denominators.json',dict(**source, strata=strata,
        localization_absolute_supplement=absolute, supported_count=12,excluded_localization_count=143,
        all_output_burden=dict(raw_historical=57600,above_frozen_score=11090,
            historical_natural_matches=45,historical_unmatched_to_incomplete_GT=11045,
            historical_natural_misses=110,historical_duplicates=9,
            fixed_diagnostic_background_sample=154,confirmed_physical_FP=0,
            physical_precision=None,class_confusion='NotEstablished_AnnotationsIncomplete'),
        new_data_model_metrics='NotRun; identity support is not detector recall',
        historical_semantic_caveat='Inherited background views retain old semantic histogram fields; old barrier fraction is not reliable taxonomy evidence. M8-C correction and M8-E Unknown policy take precedence.',
        camera_denominator='Recorded GT views; targets can appear in multiple cameras. Do not sum camera denominators as independent objects.',
        confidence='Descriptive correlated observations; no independent-sample confidence intervals.'))
    all_rows=[]
    for a in read('actor_environment_gt')['audit_assets']:
        all_rows.extend(r for f in json.loads(Path(a['path']).read_text())['frames'] for r in f['objects'])
    bad=[dict(group=r['group'],entity=r['entity_key'],phase=r['phase']) for r in all_rows
         if r['identity_grade'].startswith('RenderLinked') and not r['frame_consistent']]
    assert not bad
    write('reports/m8e_source_frame_verification.json', dict(**source,
        linked_records=sum(r['identity_grade'].startswith('RenderLinked') for r in all_rows),
        source_frame_inconsistencies=bad, qualification='Static EnvironmentObject inventory is episode-scoped. Actor transforms checked against synchronized snapshot.'))
    logs=[json.loads(p.read_text()) for p in LOG.glob('*.json')]
    write('reports/m8e_failure_ledger.json',dict(**source,
        failed_processes=[r for r in logs if r.get('returncode',0)!=0],
        implementation_issues=[
            'Initial .venv audit lacked PIL; .venv-gpu historical audit lacked Shapely. Existing compatible environments used; no dependency installs.',
            'Boost actor-wrapper equality produced false controlled flags. Controlled_entity_ids retained; audit derives membership by ID. Initial raw data and failed sources preserved.',
            'Static prop semantic_tags empty. Development-only amendment permits controlled asset + unique geometry + measured semantic; does not globally map Dynamic to barrier.',
            'Region count corrected to distinct camera count; actual grades unchanged in all frames. No post-validation threshold change.',
            'Shell python alias unavailable during isolation launch; python3 used, no experiment executed in failed shell invocation.',
            'Two official raw-source downloads failed; browser official-source evidence and successful third source retained. TLS not disabled.'],
        unresolved=[
            '46/80 controlled present source links; incomplete static background.',
            '21/30 far identity target; front-left far0; no >40m examples.',
            'Reserved final target9.9007m from historical target; zero valid independent final groups.',
            'Implausible off-road/floating placements; ground support not logged or certified.',
            'Semantic camera disagreement mean7.50%, causes not isolated; original0.95 per-region agreement retained.',
            'No exact Mesh or independent human double verification.'],
        not_run=['new PETR forward','training','hyperparameter search','new driving loop','visual control','shared GPU runtime'],
        original_assets_preserved=True))
    tests={}
    for name,n in [('source_time_tests',18),('m8d_regression',16),('m8b_regression',18)]:
        log=LOG/(name+'.txt');s=log.read_text()
        assert 'Ran %d tests'%n in s and '\nOK\n' in s
        tests[name]=dict(count=n,log=asset(log))
    write('reports/m8e_acceptance.json',dict(**source,overall='DataFoundationImplemented_QualityPartial',
        decisions=dict(A_Data='DataContractGo',B_Identity='IdentitySupportPartial',
            B_Annotation='AnnotationCompletenessPartial',C_Far='FarFieldSupportInsufficient',
            D_Denominators='SingleMultiProxySeparated',E_Taxonomy='ExplicitClassOrUnknown_NoPhysicalPrecision',
            F_Independence='NoGo_FinalTargetHistoricalOverlap',G_Training='TrainingReadinessNoGo',
            Evaluator='LimitedEvaluatorContractReady_FullEvaluationBaselineNotReady',
            Perception='NotValidated',Safety='NotValidated',Occupancy='NotValidated',Runtime='HistoricalPartialUnchanged'),
        execution=dict(carla_version='0.9.15',capture='ActualStaticCARLA_NoDriving',groups=8,frames=32,
            sensor_payloads=768,rgb_images=192,controlled_present_targets=80,
            single_view=28,multi_view=18,unknown=34,unique_far_render_linked=21,far_groups=8,
            far_identity_support_target=30,naturalistic_training_targets_qualified=0,
            independent_final_groups=0,new_model_forwards=0,heldout_model_forwards=0,training_steps=0,driving_loops=0),
        test_results=tests,total_unit_and_regression_tests=52,
        additional_actual_contract_checks=dict(rgb_packets=32,gt_injections_rejected=32,forbidden_reads_rejected=3,
            counterfactual_objects=46,counterfactual_failures=0),
        default_model_order='A',score=.1,default_code_changed=False,
        stop_reason='Fixed allocation exhausted and data quality insufficient; no resampling or rule relaxation to meet quota.',
        superseding_assessments=['m8e_heldout_independence_exception.json','m8e_annotation_review.json'],
        next_priority='Bounded scene-generation/identity QA and historical-target-footprint exclusion; no PETR adaptation request yet.'))


if __name__=='__main__':
    main()
