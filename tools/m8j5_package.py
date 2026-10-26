"""Aggregate-only publication; original images, labels and reviews stay private."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False)


def main():
    cfg=read('configs/m8j5_protocol.json');root=Path(cfg['private_root'])
    common=dict(stage='M8-J5',base_commit=cfg['base_commit'],checked_at_utc=datetime.now(timezone.utc).isoformat(),
                branch='feat/m8j5-ignore-aware-petr-supervision')
    gpu=read('reports/m8j5_sources/gradient_release.json')
    run=read(root/'forward_run2/run.json')
    for layer in run['query_counts']:
        for sample in layer['samples']:sample.pop('state')
    run.update(comparison_with_M8J4=read(root/'forward_comparison.json'))
    write('reports/m8j5_frozen_forward_regression.json',dict(**common,**run,
        evidence_type='ActualFrozenRGBGPUForward_NoBackward_NoUpdates',
        latency_scope='Single cold inference plus decode; not a performance benchmark',
        model_loss_wrapper_does_not_patch_forward=True))
    write('reports/m8j5_query_supervision.json',dict(**common,
        real_evidence='Actual00107FrozenForward_UnsignedSupervision',real_layers=run['query_counts'],
        real_training_batches=0,current_golden=0,current_safe_regions=0,
        real_all_ignore_reason='Absent human signoff and all-class safe-negative certificates, not a blanket training strategy',
        synthetic_cases=[dict(name=r['name'],audit=r.get('audit')) for r in gpu['records'] if 'audit' in r],
        future_signed_positive_only='Positives remain active; unverified unmatched remain Ignore; not readiness for balanced detection training'))
    write('reports/m8j5_golden_review_status.json',dict(**common,evidence_type='ActualPrivateReviewMaterial_NoHumanSignoff',
        candidates=36,candidate_frames=27,geometry_status='CandidateOnly',golden=0,reviewer='NotAvailable',review_timestamp='NotAvailable',
        human_review='Pending',private_html=str(root/'review_index.html'),
        private_json=str(root/'golden_candidates.json'),template=str(root/'review_responses_TEMPLATE.json'),
        signoff_validator='tools.m8j5_review_gate',authentication='JSON validates structure and hashes, cannot authenticate human identity',
        unmodified_candidate_sha_required=True,manual_request_presented_in_conversation=True,
        real_updates='NotRun_HumanSignoffMissing',raw_or_review_images_published=False))
    write('reports/m8j5_loss_validation.json',dict(**common,**gpu,
        tests_passed=16,tests_skipped=0,full_loss_and_gradient_parity_tolerance=1e-6,
        official_head_and_assigner_unmodified=True,
        normalization='max(P+bg_cls_weight*N,1); regression max(reduce_mean(P),1); bg_cls_weight=0 in this config',
        all_layers='Six independent matching and identical policy, 12 supervised loss terms',
        spatial_policy='Detached yaw-independent full box envelope+0.5m; full containment in one reviewed safe AABB; touching Unknown/Positive veto; Positive always wins',
        limitation='Envelope is predicted geometry, not physical ground truth; real safe-region quality and localization uncertainty are not certified',
        negative_class_scope='All ten classes required; partial car/person visibility does not certify no other classes',
        zero_positive_bug='Legacy sampler returned [0,4] vs [0,9]; wrapper constructs targets from unchanged official Hungarian result',
        tested_two_stage=False,tested_distributed=False,tested_precision='FP32',
        smoke='NotRun_NoHumanGolden'))
    acceptance=dict(**common,overall='Partial',decision='IgnoreLossEngineeringGo_HumanGoldenAndRealUpdateNoGo',
        trust_axes='Implemented_3741Records_214Groups',difference_712_711='Explained_OneYawOnlyTiltFilter',
        human_golden='Pending_0of36CandidatesSigned',negative_certification='Pending_0Regions',
        synthetic_gradient='Passed_16IsolatedTests',CPU_contracts='Passed_24Tests',
        frozen_inference='Passed_ExactDecodedOutputEqualityToM8J4_OneForward',
        real_optimizer_smoke='NotRun_HumanSignoffMissing',training_readiness='TrainingReadinessNoGo',
        generalization='NotEvaluated_SingleClip',visual_control='NotAuthorized',
        default_model_modified=False,old_reports_modified=False,new_dataset_downloads=0,CARLA_launches=0,
        real_backward_calls=0,optimizer_steps=0,maintainer_contact='NotSent',main_merged=False,
        source_and_evidence_manifest='reports/m8j5_release_manifest.json',
        private_assets_published=False,
        next_stage_gates=['Real reviewer-supplied immutable target signoffs; verify evidence source, not JSON alone',
            'Positive geometry/class/corner roundtrip and model range/crop approved; no velocity loss',
            'Ignore-aware exact-parity/gradient tests remain passing for every decoder',
            'At most two signed frames and three updates, <=8GiB allocated and <=180s; separate real-chain validation before J6',
            'For small detection training, reviewed all-class negative support or explicitly bounded positive-only feasibility scope; no automatic negative certification',
            'J6 budget and objectives declared, no generalization from random split of this clip'])
    write('reports/m8j5_acceptance.json',acceptance)
    assets=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    write(root/'asset_inventory.json',dict(files=assets))
    write('reports/m8j5_private_artifacts.json',dict(**common,root=str(root),file_count=len(assets),
        bytes=sum(r['bytes'] for r in assets),inventory_sha256=sha(root/'asset_inventory.json'),
        candidates_sha256=sha(root/'golden_candidates.json'),review_template_sha256=sha(root/'review_responses_TEMPLATE.json'),
        originals_or_per_object_labels_committed=False))
    print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
