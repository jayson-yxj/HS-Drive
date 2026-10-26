"""Package aggregate M8-J4 results; keep all per-object/image assets private."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tools.m8j4_sidecar import OUT, save


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checked = datetime.now(timezone.utc).isoformat()
    cfg = load('configs/m8j4_protocol.json')
    reviewed = load(OUT / 'reviewed_summary.json')
    support = load(OUT / 'support_audit.json')
    review = load(OUT / 'review_manifest.json')
    initial = load(OUT / 'sidecar_summary.json')
    run = load(OUT / 'forward/run.json')
    model_source = Path(cfg['model']['runtime']) / 'sources/PETR/projects/mmdet3d_plugin/models/detectors/petr3d.py'
    prefix = dict(stage='M8-J4', checked_at_utc=checked, base_commit=cfg['base_commit'],
                  branch='feat/m8j4-single-frame-gt-forward')
    save(Path('reports/m8j4_gt_sidecar.json'), dict(**prefix, **reviewed, support=support,
        evidence_type='ActualExistingPrivatePayload_OfflineCPU',
        source_clip='DynamicObjectCrossing_Town02_Route13_Weather6',
        source_revision='ce56cfbc26edf7796586e26fd44644f0ac2f6bcc',
        source_archive_sha256='b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2',
        records_are_actor_frames_not_independent_targets=True,
        calibration='ActualBench2DriveK_T_NoHSRigSubstitution',
        source_frame=None, timestamp_s=None, synchronization='NotIndependentlyVerified',
        valid_does_not_mean_certified=True, unsupported_traffic_sign_light='Unknown_NoEightCornerReference',
        velocity_supervision='NotValidated', depth_supervision='Excluded_Uint8',
        expert_assessment='Quarantined_OrdinalOffset',
        historical_reports_modified=False))
    denominator = 1276
    association = dict(**prefix, evidence_type='OfficialSourcePlusActualPayload_OfflineCPU',
        encoding='Raw BGRA: R semantic; G low byte and B high byte of UE GetUniqueID; truncated render token, not CARLA registry Actor ID',
        source_evidence=[load('reports/m8j4_sources/tagger.cpp.json'), load('reports/m8j4_sources/actor_registry.cpp.json')],
        captured_executable_identity='Unavailable_OfficialVersionSourceIsNotExactCaptureBinaryProof',
        association_method=cfg['association'], thresholds_unchanged=True,
        supported_actor_frame_denominator=denominator,
        status_counts={'Valid':712, 'Unknown':541, 'Invalid':23},
        status_fractions={'Valid':712/denominator, 'Unknown':541/denominator, 'Invalid':23/denominator},
        eligible_projected_actor_view_denominator=1322,
        eligible_view_status_counts={'Valid':758, 'Unknown':564, 'Invalid':0},
        eligible_view_valid_fraction=758/1322, eligible_view_unknown_fraction=564/1322,
        all_actor_views={'Valid':758, 'Unknown':564, 'Invalid':6334, 'denominator':7656},
        ambiguous_render_token_groups=len(initial['render_token_cross_actor_conflicts']),
        withdrawn_view_associations=39, active_ambiguous_associations=0,
        quarantine_is_not_root_cause_resolution=True,
        actual_association_precision='NotMeasured_NoIndependentPixelIdentityTruth',
        failed_identity_proof='No proof of absence for unmatched projection; use Unknown, not false negative/background',
        camera_support=support['view_support'],
        no_positive_support_cameras=['CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT'],
        confidence_calibrated=False, human_certification='Pending')
    save(Path('reports/m8j4_instance_association.json'), association)

    # Human-readable observations recorded by the assistant after opening private images.
    observed = sorted(set(review['uniform'] + review['anomalies']))
    notes = [
        '00000–00022: six-view overview inspected; underground actor not visibly supported. Pixel-class absence was separately measured.',
        '00027–00074: distant pedestrian near kiosk and partly occluded objects remain uncertain; projection is not visibility proof.',
        '00083,00092,00101,00111,00120: pedestrian passes front/right/left views; close truncation and occlusion require object-level review.',
        '00055,00064,00138,00175,00203: near vehicles cross image boundaries; clipped boxes appear broadly consistent in overview.',
        '00185–00213: foreground vehicle and multiple distant rear objects; unresolved far/overlapping boxes must retain Unknown.',
        'Three inspected instance mosaics support coarse silhouette checks only; pseudo-color tokens do not establish Actor identity.',
        '00107 private BEV comparison inspected: some centers align, many extra predictions have incomplete reference coverage; no FP certification.'
    ]
    visual = dict(**prefix, evidence_type='AssistantVisualInspectionPlusCPUOverlays',
        generated_six_camera_RGB_and_instance_frames=53, actually_opened_RGB_frames=observed,
        actually_opened_RGB_frame_count=len(observed), uniformly_sampled_RGB_frame_count=24,
        anomaly_RGB_frame_count=23, actually_opened_instance_ordinals=['00083','00092','00111'],
        actually_opened_BEV_ordinals=['00107'], observations=notes,
        independent_human_signoff='Pending', all_automatically_supported_objects_require_review=True,
        pixel_level_certification=False, full_resolution_every_object_review=False,
        private_visual_directory=str(OUT/'reviewed_visuals'), original_before_quarantine_visuals_retained=True)
    save(OUT/'visual_inspection_ledger.json', visual)
    save(Path('reports/m8j4_visual_review.json'), visual)

    # Preserve the raw GPU record; append an explicit shape interpretation instead of rewriting it.
    forward = dict(**prefix, **run, evidence_type='ActualRTX4090_FP32_FrozenForward',
        model=cfg['model'], execution=load('reports/m8j4_sources/gpu_execution.json'),
        model_entry_shape=[1,6,3,512,1408],
        recorded_post_call_shape=[6,3,512,1408],
        input_shape_note='Original PETR extract_img_feat line67 calls img.squeeze_() in-place for B=1. Raw log shape is post-call; prepare() asserted 5D entry. No calibration/model modification or extra GPU run.',
        original_petr3d_sha256=sha(model_source),
        geometry_projection_max_residual_px=max(load(OUT/'forward'/(s+'.json'))['calibration']['projection_residual_px'] for s in cfg['model']['ordinals']),
        peak_allocated_GiB=max(r['peak_allocated_bytes'] for r in run['records'])/1024**3,
        peak_reserved_GiB=max(r['peak_reserved_bytes'] for r in run['records'])/1024**3,
        memory_scope='PyTorch CUDA allocator peak, not total device/process VRAM',
        latency_scope='Synchronized FP32 inference; first call cold; 3 samples insufficient for P95/runtime qualification',
        total_scope='Per-frame total excludes model construction/loading and result serialization; includes checks and CPU conversion',
        GT_isolation='Numeric RGB/K/T only, Python-open audit canary and disabled network. Not a hostile-native-code filesystem sandbox.',
        accepted_label_gate='Basic sidecar geometry and isolation plus ambiguity quarantine; not complete label certification')
    save(Path('reports/m8j4_model_forward.json'), forward)
    acceptance = dict(**prefix, overall='Partial',
        decision='SingleFrameGTInterfaceAndFrozenForwardGo_LabelQualityPartial',
        gt_sidecar_interface='Passed_214Groups_GeometryAndStatusContract',
        instance_identity='Partial_ConservativeAssociation_NotActorIDEquality',
        underground_pedestrian_audit='Passed_All23ExcludedWith138ViewEvidence_CauseUnproven',
        visual_review='Completed_24UniformPlus23Anomaly_44UniqueRGBFrames_HumanSignoffPending',
        supervision_isolation='Passed_AllowlistedRGBK_T_NumericBundle_Canary',
        model_forward='Passed_3RealSixCameraFrozenPETRForwards',
        label_quality='Partial_NoIndependentIdentityOrExhaustiveNegativeTruth',
        generalization='NotEvaluated_OneCorrelatedClip', runtime='NotQualified_ThreeSamples',
        training='TrainingReadinessNoGo', visual_driving='NotValidated',
        tests={'passed':18,'skipped':0,'new_M8J4':9,'reused_M8J3_RGB_contract':9,
               'log':'reports/m8j4_sources/tests_final.log','log_sha256':sha('reports/m8j4_sources/tests_final.log')},
        private_artifacts_only=True, published_images=False, published_per_object_labels=False,
        additional_dataset_downloads=0, CARLA_launches=0, training_steps=0,
        inference_calls=3, new_dependencies=0, default_model_changes=False,
        old_reports_modified=False, main_merged=False,
        license='HistoricalConflictRetained_NoNewAnalysis',
        next_minimal_training_conditions=[
            'Human verify a few visible car/pedestrian targets and review all targets in their supervision support',
            'Mask Unknown at the loss/assignment level; ordinary PETR unmatched-query background loss is not yet safe with incomplete labels',
            'Match box-center/lwh/yaw, model range/crop; exclude velocity until separately validated',
            'Predeclare 1–4 frame frozen-backbone overfit budget and gradient/leakage/collapse checks; not generalization',
            'Do not random-split this clip to claim independence; obtain separately approved independent data for future generalization'
        ])
    save(Path('reports/m8j4_acceptance.json'), acceptance)
    # Detailed file inventory stays private; publish only its digest and aggregate sizes.
    assets=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file():assets.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    save(OUT/'artifact_inventory.json',dict(created_at_utc=checked,files=assets))
    save(Path('reports/m8j4_private_artifacts.json'),dict(**prefix,
        root=str(OUT), file_count=len(assets), bytes=sum(p['bytes'] for p in assets),
        private_inventory=str(OUT/'artifact_inventory.json'), inventory_sha256=sha(OUT/'artifact_inventory.json'),
        sidecar_summary_sha256=sha(OUT/'sidecar_summary.json'),
        reviewed_summary_sha256=sha(OUT/'reviewed_summary.json'),
        forward_run_sha256=sha(OUT/'forward/run.json'),
        input_manifest_sha256=sha(OUT/'inference_input/manifest.json'),
        includes='Original and quarantined sidecars, image mosaics, RGB input bundles, all prediction candidates, private audits',
        uploaded_to_git=False))
    print(json.dumps(acceptance,indent=2))


if __name__ == '__main__':
    main()
