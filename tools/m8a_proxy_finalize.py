"""Append-only supplemental network/asset decision, without changing old gates."""
import json
from pathlib import Path
import subprocess

from tools.m8a_common import asset,write
from tools.m8a_proxy_audit import CONFIG,provenance

ROOT=Path("reports/m8a_proxy_retry_logs")


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    config=json.loads(CONFIG.read_text())
    discovery=read("discovery.json")
    curl=read("curl_probes.json")
    python=read("python_probes.json")
    download=read("gdown_result.json")
    archive=read("checkpoint_archive_v2.json")
    execution=read("isolated_load_execution_v2.json")
    inspected=read("checkpoint_structure_v2.json")["inspection"]
    by_name={r["name"]:r for r in curl["results"]}
    proxy_ready=all(by_name[n]["http_code"]==200 and by_name[n]["https_verified"]
                    and by_name[n]["connected_to_configured_loopback"]
                    for n in ["github_explicit_http","google_drive_explicit_http","google_drive_socks5h"])
    python_ready=all(x.get("status")==200 and x.get("https_verified") and
                     x.get("connected_to_configured_loopback") for x in python["results"])
    file=asset(config["checkpoint_output"])
    complete=download["completed"] and archive["all_crc_passed"] and archive["file"]==file
    checkpoint_ready=(execution["returncode"]==0 and
        inspected["status"]=="BasicCheckpointContractPassed" and
        inspected["contract"]["passed"] and not inspected["nonfinite_tensors"])
    assert proxy_ready and python_ready and complete and checkpoint_ready, "No silent partial Go"
    original=json.loads(Path("reports/m8a_release_manifest.json").read_text())
    assert all(asset(x["path"])==x for x in original["files"]), "Historical evidence changed"
    source=Path("/root/gpufree-data/hs-drive-experiments/m6a-petr/source")
    sources=[asset(source/p) for p in ["README.md","LICENSE",config["official_config"],
        "projects/mmdet3d_plugin/models/dense_heads/petr_head.py"]]
    causal=dict(
        same_server_paired_control=True,
        inherited_github_http=by_name["github_inherited"]["http_code"],
        inherited_drive_http=by_name["google_drive_inherited"]["http_code"],
        inherited_drive_curl_exit=by_name["google_drive_inherited"]["returncode"],
        explicit_drive_http=by_name["google_drive_explicit_http"]["http_code"],
        conclusion="Current download environment did not automatically use the running proxy. "
                   "Explicit proxy changes Google Drive from connection timeout to verified HTTP200 "
                   "and successful official checkpoint download.",
        historical_scope="Strong evidence for the same Google connectivity failure mode; not proof "
                         "that every M6-A failure had the same cause. Baidu errno=-7/2 remains separate.",
        no_public_exit_identity_claim=True,
        limitations="Same configured local proxy and explicit socket/CONNECT evidence; rule-based "
                    "routing may use different upstream exits by hostname. No private exit IP collected.")
    write("reports/m8a_network_proxy_audit.json",dict(
        **provenance("ActualGPUHostPairedProxyAndTLSAudit"),status="ProxyVerified",
        audit_host="HS-Drive RTX4090 server, not user workstation",
        gpu="NVIDIA GeForce RTX 4090",discovery=discovery,
        effective_type="Explicit HTTP CONNECT selected; same mixed port supports SOCKS5h",
        effective_endpoint=config["discovered_proxy"],
        system_tun_enabled_observed=False,inherited_proxy_env_present=False,
        upstream_transparent_routing="NotClaimed_not_exhaustively_audited",
        curl=curl["results"],python=python["results"],tls_certificate_checks_enabled=True,
        same_local_proxy_verified=True,same_public_exit_identity_verified=False,
        causal_analysis=causal,secrets_and_exit_ips_logged=False,
        reproduction=[
            "curl --disable -IL --connect-timeout 10 --max-time 30 https://github.com/megvii-research/PETR",
            "curl --disable -IL --proxy http://127.0.0.1:7897 --connect-timeout 10 --max-time 30 "
            "https://drive.google.com/file/d/1c5rgTpHA98dFKmQ9BJN0zZbSuBFT8_Bt/view"],
        note="Published logs contain allowlisted fields, not response cookies or private upstream URLs."))
    write("reports/m8a_official_checkpoint_download.json",dict(
        **provenance("ActualOfficialDetectionCheckpointDownload"),
        status="DownloadCompleted_IntegrityInspectedSeparately",
        acquisition_git_head=download["git_head"],network_code_commit=curl["git_head"],
        official_repository="https://github.com/megvii-research/PETR",
        official_commit=config["official_commit"],official_config=config["official_config"],
        official_entry=config["sites"]["google_drive"],official_source_files=sources,
        downloader=dict(name="gdown",version=download["tool_version"],isolated_environment=config["download_env"],
            environment_lock=asset(ROOT/"download_env_lock.json"),
            explicit_proxy=config["discovered_proxy"],resume_enabled=True,retries_max=1,
            retries_actual="NotSeparatelyInstrumented_NoExceptionInFinalResult",
            inactivity_timeout_s=30,outer_wall_timeout_s=900,tls_verify=True,
            browser_or_user_cookies=False,permission_bypass=False),
        elapsed_s=download["elapsed_s"],file=file,completed=download["completed"],
        captured_redacted_log=download["captured_log"],error=download["error"],
        baidu_retry="NotRun_GoogleSucceeded",
        source_scope=dict(trust_basis="Pinned official authors' model row + same Google file ID + "
            "validated TLS via explicit proxy + successful trained detector structure inspection",
            official_expected_sha256=None,local_sha_is_independent_authenticity_proof=False,
            code_license="Apache-2.0",weights_offered_by_authors="Evaluation/research",
            separate_checkpoint_license="NotFound_in_reviewed_materials; code license not automatically extended to all third-party rights",
            redistribution_or_commercial_rights_asserted=False),
        historical_failure_reports_unchanged=True,checkpoint_uploaded_to_git=False,
        raw_log=asset(ROOT/"gdown_result.json")))
    write("reports/m8a_checkpoint_integrity.json",dict(
        **provenance("ActualArchiveAndIsolatedCPUCheckpointAssurance"),
        status="AssetAccessGo",file=file,archive=archive,
        isolated_execution=asset(ROOT/"isolated_load_execution_v2.json"),
        initial_tool_failure=asset(ROOT/"isolated_load_execution.json"),
        initial_failure_reason="Sandbox script named inspect.py shadowed Python standard inspect; "
            "failed before torch import/checkpoint load. Renamed only; failure retained.",
        successful_inspection_commit=execution["git_head"],
        isolation=inspected["isolation"],load=dict(torch=inspected["torch_version"],
            weights_only=inspected["weights_only"],cpu_only=True,
            torch_cuda_initialized=inspected["torch_cuda_initialized"],
            arbitrary_pickle_fallback=False,environment="Separate namespace and inspect-env; "
            "trusted existing installed torch runtime read-only mounted at /runtime; "
            "HS project and credentials not mounted, no NVIDIA devices"),
        components=inspected["components"],metadata=inspected["metadata"],
        tensor_count=inspected["tensor_count"],tensor_elements=inspected["tensor_elements"],
        finite_tensors=len(inspected["nonfinite_tensors"])==0,
        basic_config_shape_contract=inspected["contract"],
        full_tensor_shape_manifest=asset(ROOT/"checkpoint_structure_v2.json"),
        independent_expected_hash_available=False,
        authentic_source_basis="Official provenance chain, not local SHA alone",
        full_model_load_state_dict="NotRun",cuda_operator_tests="NotRun",
        six_camera_forward="NotRun",gpu_memory_latency="NotRun",
        carla_adapter="NotRun",petr_overall_P0="NotPassed_AssetOnly",
        gates_preserved={"M5O":"ResearchCompleted / CandidateNoGo",
            "M7":"ResearchCompleted / OfflinePlannerResearchGo / OnlinePlannerNoGo",
            "control":"ControlNoGo","runtime":"RuntimePartial",
            "occupancy":"NotPassed","visual_driving":"SafetyNotValidated"},
        next_stage="Original M8-A dependency/operator/model/forward work only after this asset "
                   "gate; not executed in this network-and-asset supplement."))
    print(json.dumps(dict(network="ProxyVerified",asset="AssetAccessGo",petr_P0="NotPassed_AssetOnly")))


if __name__=="__main__":
    main()
