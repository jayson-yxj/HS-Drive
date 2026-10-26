"""Verify sealed M8-I0 review and unchanged historical assets; no runtime/network."""
import ast
import json
import re
import subprocess
from pathlib import Path
from tools.m8i0_audit import BASE, sha


def main():
    manifest = json.loads(Path("reports/m8i0_release_manifest.json").read_text())
    verified = []
    for item in manifest["files"]:
        path = Path(item["path"])
        assert path.is_file(), str(path)
        assert sha(path) == item["sha256"], str(path)
        assert path.stat().st_size == item["bytes"], str(path)
        if path.suffix == ".json":
            json.loads(path.read_text())
        if path.suffix == ".py":
            ast.parse(path.read_text())
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if "://" not in target and not target.startswith("#"):
                    assert (path.parent / target.split("#")[0]).exists(), (str(path), target)
        verified.append(str(path))
    required = ["summary.md", "acceptance.json", "traffic_manager_review.md",
                "scenariorunner_compatibility.md", "scenariorunner_scenario_catalog.json",
                "bench2drive_data_contract.md", "external_dataset_access.md",
                "groundtruth_compatibility_matrix.json", "compute_and_maintenance_cost.md",
                "source_comparison.md", "next_pilot_proposal.md"]
    assert all(f"reports/m8i0_{name}" in verified for name in required)
    assert "docs/M8I0_ARCHITECTURE_DECISION.md" in verified
    acc = json.loads(Path("reports/m8i0_acceptance.json").read_text())
    assert acc["base_commit"] == BASE
    assert not acc["pilot_authorized"] and acc["pilot_execution"] == "NotRun"
    assert all(value == 0 for value in acc["execution"].values())
    assert acc["qualified_data_source_status"] == "NoSourceReady"
    assert acc["history"]["m8h1_status"] == "DevelopmentSmokeNoGo"
    assert acc["new_independent_final"] == 0
    for path, digest in acc["source_code_hashes"].items():
        assert sha(path) == digest, path
    matrix = json.loads(Path("reports/m8i0_groundtruth_compatibility_matrix.json").read_text())
    assert len(matrix["fields"]) == 28
    assert not any(matrix["state_implications"].values())
    assert not any(x["payload_verified"] for x in matrix["fields"])
    assert not set(matrix["model_input_allowlist"]) & set(matrix["sidecar_only"])
    catalog = json.loads(Path("reports/m8i0_scenariorunner_scenario_catalog.json").read_text())
    assert len(catalog["scenarios"]) == 3
    assert sum(x["count"] for x in catalog["scenarios"]) == 26
    historical = 0
    for stage in ["m8g", "m8h0", "m8h1"]:
        for item in json.loads(Path(f"reports/{stage}_release_manifest.json").read_text())["files"]:
            assert sha(item["path"]) == item["sha256"], item["path"]
            historical += 1
    changes = subprocess.check_output(
        ["git", "diff", "--name-status", BASE, "--diff-filter=MDTR"], text=True).strip()
    assert not changes, changes
    print(json.dumps(dict(kind="ActualOfflineArtifactVerification_NotSimulation",
                          status="Passed", review_files=len(verified),
                          historical_manifest_entries=historical,
                          required_deliverables=len(required) + 1,
                          base_tracked_modifications=0, source_hashes_verified=True,
                          JSON_and_source_syntax="Passed", local_document_links="Passed",
                          stage_boundary_consistency="Passed", new_CARLA_or_GPU_tests="NotRun",
                          command="python3 -m tools.m8i0_verify"), indent=2))


if __name__ == "__main__":
    main()
