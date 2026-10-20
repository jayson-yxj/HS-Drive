"""Final report assembly from measured artifacts; no quality gate relaxation."""

import json
from pathlib import Path
from simulator.recorder import atomic_json, file_hash


def read(p):
    return json.loads(Path(p).read_text())


def main():
    out = Path("reports/m5c_acceptance.json")
    if out.exists():
        raise FileExistsError(out)
    test = read("reports/m5c_absolute_obstacle_detection.json")
    training = read("reports/m5c_model_ablation.json")
    runtime = read("reports/m5c_runtime_stress.json")
    data = read("reports/m5c_dataset_quality.json")
    freeze = read("reports/m5c_pretest_decision.json")
    assets = read("reports/m5c_assets.json")
    perf = {
        f"{r['name']}_{r['seed']}": dict(
            absolute=r["summary"],
            paired=r["paired_summary"],
            threshold=r["threshold"],
            change_threshold=r["change_threshold"],
            checkpoint_sha256=r["checkpoint_sha256"],
        )
        for r in test["records"]
    }
    unchanged = [p for p, h in freeze["code_sha256"].items() if file_hash(p) != h]
    assert unchanged == ["tools/m5c_visualize.py"], unchanged
    states = dict(
        engineering_interfaces="Passed_in_tested_scope",
        training_learnability="Passed_research_only",
        spatial_generalization="Partial",
        runtime_stability="Partial",
        oracle_driving_safety="Partial",
        pure_visual_driving_safety="NotValidated",
    )
    result = dict(
        stage="M5-C",
        overall="Partial",
        base_git_sha=assets["base_git_sha"],
        branch="feat/m5c-obstacle-localization",
        states=states,
        metrics=perf,
        dataset_summary=data["summary"],
        runtime=dict(
            runs=len(runtime["records"]),
            ticks=sum(r["ticks"] for r in runtime["records"]),
            collisions=sum(r["collisions"] for r in runtime["records"]),
            lane_invasions=sum(r["lane_invasions"] for r in runtime["records"]),
            road_boundary_frames=sum(
                r["road_boundary_frames"] for r in runtime["records"]
            ),
            expired_normal_control=sum(
                r["normal_control_expired"] for r in runtime["records"]
            ),
            max_control_interval_ms=max(
                r["control_interval_ms"]["max"] for r in runtime["records"]
            ),
            strict_cycle_status=runtime["cycle_status"],
            fault_to_brake_s=[
                dict(
                    report=r["report"], scene=r["scene"], seconds=r["fault_to_brake_s"]
                )
                for r in runtime["records"]
                if r["fault_to_brake_s"] is not None
            ],
            incomplete_routes=[
                dict(
                    report=r["report"],
                    scene=r["scene"],
                    completion=r["route_completion"],
                )
                for r in runtime["records"]
                if r["scene"] in ["straight", "mild_curve", "junction_turn"]
                and not r["route_passed"]
            ],
        ),
        training=dict(
            models=len(training["records"]),
            seeds=[71, 72],
            steps_each=400,
            gpu=training["gpu"],
            max_allocated_mib=max(r["peak_allocated_mib"] for r in training["records"]),
            all_gradients_finite=all(
                r["finite_gradients"] for r in training["records"]
            ),
            frozen_gradients_absent=all(
                r["frozen_gradients_absent"] for r in training["records"]
            ),
            precompute=training["precompute"],
        ),
        tests=dict(
            full_passed=219,
            full_skipped=5,
            full_xml="reports/m5c_full_tests.xml",
            carla_interface_passed=18,
            carla_skipped_coverage_passed=14,
            skip_reason="GPU env has no CARLA module; all5 skipped API cases covered in CARLA env",
            types="synthetic/unit and actualGPU contracts; CARLA closedloop separately reported",
        ),
        preservation=dict(
            old_files_checked=assets["old_files_checked"],
            old_files_changed=assets["old_files_changed"],
            models_and_thresholds_frozen=True,
            pretest_sha256=file_hash("reports/m5c_pretest_decision.json"),
            pretest_code_changes=unchanged,
            explanation="Only visualization report source updated to corrected validation geometry files. Training/evaluation/label code and selected checkpoints unchanged. Additive diagnostic scripts not used for test selection.",
        ),
        decisions=dict(
            expand_formal_occupancy_training=False,
            start_temporal_bev=False,
            start_sparse_query=False,
            change_depth_bins=False,
            change_splat_default=False,
            next="Continue small singleframe research: target-aware sampling/validation with honest unknown constraints; matched highresolution image sampling and geometry-resolution hypotheses; predeclare any subbin/softsplat candidate, no automatic joint training.",
        ),
        limits=[
            "Only4 heldout poses, configurations fixed across sets, maps shared with training; no population generalization certification.",
            "Known-surface scores do not certify unknown volume. Directhead exhibits persistent high absent-point scores.",
            "Actor target-only precision fields are vacuous because that submask contains positives only; use recall, known Ground FP, and separate pointabsence audit together.",
            "Weak actor detection can be satisfied by position prior; not evidence of visual object detection. Small/pedestrian cases fail or lack support.",
            "20-35m feature geometry sample n6 across3groups, test actor surface cells28 across4groups; no dense farfield claim.",
            "Ground/Obstacle reference same renderer; AABB candidate only, not Mesh truth. Mixed physical subcells remain uncertain.",
            "C/D pairloss applies to original combination output; D direct output has auxiliary surface BCE only, so weak changes are not a fair proof that pairing cannot help a directhead.",
            "Threshold objective changed from M5B signedchange to M5C absolute surface; background dominance and different data confound crossstage numerical comparisons.",
            "Runtime pressure combines CPU label processing and GPU inference; not a single-variable attribution. Most five-minute soak is postroute hold.",
            "One CPUload junction route stopped near6.57% completion under stale-plan degradation; safety protection preserved but availability failed.",
            "No purevisual closedloop; original low-speed comfort limitations remain; old controller not promoted.",
        ],
        evidence_kinds=dict(
            capture="actual_CARLA_0.9.15",
            learning="actual_RTX4090",
            geometry="offline_GPU_validation_only",
            lattice="synthetic_uniform_depth_sweep",
            labels="same_renderer_heldout_pixels_not_independentMesh",
            runtime="actual_wallpaced_CARLA_Oracle",
            tests="synthetic_unit_GPU_and_CARLA_API",
        ),
        authoritative_reports=dict(
            geometry="reports/m5c_corrected_geometry_resolution.json",
            sampling="reports/m5c_corrected_feature_sampling.json",
            historical_localization="reports/m5c_spatial_localization_historical_corrected.json",
            singleframe="reports/m5c_absolute_obstacle_detection.json",
            local_absence="reports/m5c_local_witness_scores.json",
            corrections="reports/m5c_diagnostic_corrections.json",
        ),
        reproduce="docs/M5C_IMPLEMENTATION.md",
        source_sha256=assets["source_sha256"],
        reports_sha256=assets["reports_sha256"],
    )
    atomic_json(out, result)


if __name__ == "__main__":
    main()
