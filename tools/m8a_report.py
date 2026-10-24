"""Close a failed asset gate; no installs, GPU work, or downstream experiments."""
import argparse
import json
import subprocess
from pathlib import Path

from tools.m8a_common import BASE, PROTOCOL, asset, provenance, sha, write
from tools.m8a_assets import asset_gate


REPORTS = (
    "m8a_acceptance.json", "m8a_official_assets.json",
    "m8a_checkpoint_validation.json", "m8a_dependency_matrix.json",
    "m8a_cuda_operator_tests.json", "m8a_model_load.json",
    "m8a_4090_forward.json", "m8a_gpu_memory_latency.json",
    "m8a_camera_projection_contract.json",
)


def not_run_payload(kind, reason, **fields):
    return dict(**provenance(kind), status="NotRun", reason=reason,
                prerequisite="AssetAccessPassed", **fields)


def preserve():
    """Verify inherited tracked files, sealed M7 artifacts, models and packages."""
    changed = subprocess.check_output(
        ["git", "diff", "--name-status", "--diff-filter=MDRT", BASE, "--"],
        text=True).splitlines()
    if changed:
        raise ValueError("historical_tracked_files_changed:" + repr(changed))
    manifest = json.loads(Path("reports/m7e_release_manifest.json").read_text())
    mismatches = []
    for item in manifest["files"]:
        p = Path(item["path"])
        if not p.exists() or p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]:
            mismatches.append(str(p))
    before = json.loads(Path("reports/m8a_execution_logs/environment_before.json").read_text())
    command = ("import importlib.metadata as m,json,sys;"
               "print(json.dumps({'python':sys.version,"
               "'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))")
    environments = {}
    for exe in before["environments"]:
        environments[exe] = json.loads(subprocess.check_output([exe, "-c", command], text=True))
    env_equal = environments == before["environments"]
    history = [asset(item["path"]) for item in before["history"]]
    history_equal = history == before["history"]
    expected_models = {
        "runs/m5i_learning/m5i_mobile_gt_101_step2400.pt":
            "e2f0d0d85a03544186cd8c92095c95a3e2baf256fee0d9ab8f591e1c715d5c33",
        "runs/m5o_training/m5o_direct_101_f100_step100.pt":
            "9d407af4c0eefb31e85fd7b2a0d9d2bb82948d9a320447ce8df3b49145174ebb",
    }
    models = [asset(p) for p in expected_models]
    model_equal = all(x["sha256"] == expected_models[x["path"]] for x in models)
    passed = not mismatches and env_equal and history_equal and model_equal
    result = dict(**provenance("ActualReadOnlyPreservationCheck"),
        status="Passed" if passed else "Failed", modified_inherited_tracked=changed,
        m7e_manifest_count=len(manifest["files"]), m7e_mismatches=mismatches,
        environment_package_versions_unchanged=env_equal, environments=environments,
        historical_manifest_and_failure_reports_unchanged=history_equal, history=history,
        representative_models_unchanged=model_equal, models=models,
        scope="Version/package inventory and listed SHA, not all external assets or every installed package byte.")
    write("reports/m8a_execution_logs/preservation.json", result)
    if not passed:
        raise ValueError("preservation_failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    root = Path(args.attempt)
    outcome = json.loads((root/"outcome.json").read_text())
    if outcome["status"] != "AssetNoGo" or outcome["checkpoint"] is not None:
        raise ValueError("This closeout only supports an observed asset failure")
    for name in REPORTS:
        if (Path("reports")/name).exists():
            raise FileExistsError(name)
    config = json.loads(PROTOCOL.read_text())
    preserved = preserve()
    source = Path(config["historical_source"])
    checks = dict(official_source=True, usage_scope_confirmed=False,
                  complete_bytes=False, sha256_recorded=False,
                  readable_isolated=False, detection_head_present=False,
                  key_shapes_match=False)
    assert not asset_gate(checks)
    source_assets = outcome["official_files"]
    listing = json.loads((root/"baidu_listing.json").read_text())
    legal = dict(
        code_license="Apache-2.0", license=asset(source/"LICENSE"),
        model_source="Pinned official PETR README original R50-C5 result row",
        permitted_intent="Official authors offer trained weights for evaluation/research",
        separate_checkpoint_terms="Not independently established; no bytes/embedded notice obtained",
        redistribution_authorized=False, commercial_rights_asserted=False,
        no_third_party_weights=True, no_access_restriction_bypass=True)
    write("reports/m8a_official_assets.json", dict(
        **provenance("ActualServerOfficialAssetGate"), status="AssetNoGo",
        acquisition_git_head=outcome["git_head"], acquisition=asset(root/"outcome.json"),
        official_files=source_assets, official_repository=config["official_repository"],
        official_commit=config["official_commit"], official_config=config["official_config"],
        official_google_view=config["network"]["google_view"],
        official_google_download=config["network"]["google_download"],
        official_baidu_share=config["network"]["baidu_share"], legal=legal,
        observations=outcome["events"], elapsed_s=outcome["elapsed_s"],
        actual_checkpoint_bytes=0, complete_file_bytes=None, checkpoint_sha256=None,
        historical_listing_only=dict(bytes=435218261, filename="epoch_24.pth",
            source=asset("reports/m6a_execution_logs/petr_official_asset_listing.json"),
            not_obtained_file=True, not_independent_hash=True),
        current_baidu_listing=listing, asset_checks=checks, gate_passed=False,
        failure_attribution="Google connection timeouts; Baidu verify errno=0 but directory errno=-7. "
            "Service response does not distinguish endpoint/session/client incompatibility; "
            "not proof the official asset is permanently unavailable.",
        bounded_stop="One frozen Google download and one Baidu directory attempt; no retry/bypass.",
        access_budget_person_days_max=1, dependency_install_started=False))
    reason = "AssetNoGo: no official checkpoint bytes acquired in the bounded attempt."
    write("reports/m8a_checkpoint_validation.json", not_run_payload(
        "ConditionalCheckpointInspection_NotRun", reason,
        checkpoint_path=None, checkpoint_bytes=None, checkpoint_sha256=None,
        checks=checks, isolated_deserialization=False, detection_head_present=None,
        tensor_shapes=None, missing_keys=None, unexpected_keys=None,
        components={
            "DetectionCheckpoint": {"status":"NotObtained", "expected":"original PETR R50-C5 epoch_24.pth"},
            "BackboneInitializationWeights": {"status":"NotUsed",
                "config_reference":"ckpts/resnet50_msra-5891d200.pth",
                "substitute_for_detection_checkpoint":False},
            "OptimizerState":{"status":"NotInspected","used_for_inference":False},
            "Metadata":{"status":"NotInspected"}},
        not_inferred_from_filename_or_remote_size=True))
    write("reports/m8a_dependency_matrix.json", not_run_payload(
        "ConditionalIsolatedDeployment_NotRun", reason,
        new_environment_created=False, install_commands_executed=0,
        experiment_root=config["experiment_root"],
        installed_python=None, installed_torch=None, installed_cuda_runtime=None,
        toolkit=None, mmcv=None, mmdet=None, mmdet3d=None, plugin=None,
        historical_reference_only={
            "python_readme":"3.6.8", "python_m6a_attempt":"3.8.20",
            "torch":"1.9.0", "CUDA_readme":"11.2", "MMCV_wheel_CUDA":"11.1",
            "mmcv_full":"1.4.0", "mmdet":"2.24.1", "mmseg":"0.20.2", "mmdet3d":"0.17.1"},
        actual_host_inventory=asset("reports/m8a_execution_logs/environment_before.json"),
        compatibility_experiments_executed=0, new_wheel_or_source_installations=[],
        environment_lock_status="NotCreated_NoNewEnvironment",
        historical_environment_preservation=asset("reports/m8a_execution_logs/preservation.json")))
    write("reports/m8a_cuda_operator_tests.json", not_run_payload(
        "ConditionalRTX4090OperatorTests_NotRun", reason,
        gpu_name_observed="NVIDIA GeForce RTX 4090", compute_capability_measured=None,
        tests={k: {"status":"NotRun","result":None} for k in (
            "torch_cuda_is_available", "device_compute_capability", "cuda_tensor",
            "mmcv_cpp_import", "mmdet_import", "dcnv2_cuda_forward",
            "transformer_cuda_path", "petr_plugin_full_import")},
        import_torch_is_not_a_pass=True))
    write("reports/m8a_model_load.json", not_run_payload(
        "ConditionalOfficialModelBuildAndLoad_NotRun", reason,
        model_build=None, checkpoint_sha256=None, missing_keys=None,
        unexpected_keys=None, backbone_coverage=None, neck_coverage=None,
        head_coverage=None, parameter_count=None, device=None, dtype=None,
        expected_config_only={"classes":10,"queries":900,"decoder_layers":6,
                              "backbone":"ResNet50_DCNv2_C5"},
        random_initialized_detector_used=False))
    write("reports/m8a_4090_forward.json", not_run_payload(
        "ConditionalOriginalSpecificationForward_NotRun", reason,
        input_kind=None, input_files=[], output_files=[], actual_input_shape=None,
        required_input_shape=[1,6,3,512,1408], actual_forward_count=0,
        warmup_count=0, measured_count=0, output_shapes=None,
        finite_boxes=None, labels=None, score_range=None,
        official_nuscenes_replication=False, carla_quality_evaluation=False,
        synthetic_input_smoke_executed=False, reduced_specification_executed=False))
    write("reports/m8a_gpu_memory_latency.json", not_run_payload(
        "ConditionalModelResourceMeasurement_NotRun", reason,
        cold_ms=None, p50_ms=None, p95_ms=None, max_ms=None,
        cuda_synchronized_measurement=False, allocated_peak_mib=None,
        reserved_peak_mib=None, gpu_process_peak_mib=None,
        utilization=None, memory_curve=None, oom=None, cuda_error=None,
        measurements=0, host_device_memory_mib=24564,
        host_inventory_is_not_model_resource_evidence=True,
        shared_carla_gpu_test=False, strict20Hz="NotValidated"))
    write("reports/m8a_camera_projection_contract.json", not_run_payload(
        "ConditionalCARLAProjectionContract_NotRun", reason,
        additional_prerequisite="OfficialModelLoadAndOriginalSpecificationGPUForwardPassed",
        new_analytic_point_tests=0, new_non_square_box_tests=0, existing_carla_KT_tests=0,
        rgb_inference_executed=False, carla_adapter_developed=False,
        official_camera_order=None, verified_yaw_convention=None,
        projection_closure_residual=None, all_camera_depth_signs=None,
        historical_review_reference=asset("reports/m8_0_perception_readiness.md"),
        historical_review_not_promoted_to_current_test=True,
        forbidden_inputs=["GT_actor_transform","instance_mask","GT_depth","future_trajectory"],
        control_connection=False))
    write("reports/m8a_acceptance.json", dict(
        **provenance("ActualServerBoundedP0Exit"), overall="AssetNoGo",
        A_AssetAccess={"status":"AssetNoGo","complete_trained_checkpoint_obtained":False},
        B_Dependency={"status":"NotRun","reason":reason},
        C_ModelLoad={"status":"NotRun","reason":reason},
        D_Forward={"status":"NotRun","reason":reason},
        E_Resource={"status":"NotRun","reason":reason},
        F_Contract={"status":"NotRun","reason":reason},
        upstream_failure_stopped_downstream=True,
        architecture_quality="NotValidated_NotAnArchitectureFailure",
        inherited_gates={
            "M5O":"ResearchCompleted / CandidateNoGo",
            "M7":"ResearchCompleted / OfflinePlannerResearchGo / OnlinePlannerNoGo",
            "vehicle_control":"ControlNoGo", "runtime":"RuntimePartial",
            "occupancy":"NotPassed", "visual_driving":"SafetyNotValidated"},
        preservation_status=preserved["status"],
        tests=asset("reports/m8a_execution_logs/asset_gate_tests.json"),
        next_stage="NoAutomaticContinuation_NoTraining_NoAlternativeFramework",
        no_default_changes=True, no_new_carla=True, no_shared_gpu_tests=True,
        acquisition_source_commit=outcome["git_head"]))
    print(json.dumps({"reports":len(REPORTS),"overall":"AssetNoGo",
                      "downstream":"NotRun"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
