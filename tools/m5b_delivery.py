"""Build final explicit independent gates from immutable experiment records."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator.recorder import atomic_json, file_hash


def main():
    acceptance = Path("reports/m5b_acceptance.json")
    if acceptance.exists():
        raise FileExistsError(acceptance)
    runtime = json.loads(Path("reports/m5b_async_final_runtime.json").read_text())
    vision = json.loads(Path("reports/m5b_counterfactual_detection.json").read_text())
    study = json.loads(Path("reports/m5b_context_ablation_corrected.json").read_text())
    safety = json.loads(Path("reports/m5b_runtime_attribution_final.json").read_text())
    quality = json.loads(Path("reports/m5b_pair_quality.json").read_text())
    geo = json.loads(Path("reports/m5b_target_geometry.json").read_text())
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for case in [
        "junction_turn",
        "static_obstacle",
        "worker_crash",
        "stalled_worker",
        "invalid_input",
    ]:
        r = next(x for x in runtime["results"] if x["scene"] == case)
        rows = [json.loads(l) for l in Path(r["log"]).read_text().splitlines()]
        t = np.array([x["sim_time"] for x in rows])
        t -= t[0]
        ax[0].plot(t, [x["measurement"]["long_speed"] for x in rows], label=case)
        ax[1].plot(t, [x["profile"]["world_tick_ms"] for x in rows])
        gaps = np.diff([x["control_apply_wall"] for x in rows]) * 1000
        ax[2].plot(t[1:], gaps)
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_ylabel("actual signed speed m/s")
    ax[1].set_ylabel("world.tick ms")
    ax[2].set_ylabel("control interval ms")
    ax[2].set_xlabel("simulation seconds")
    ax[2].axhline(50, color="r", linestyle="--")
    fig.tight_layout()
    fig.savefig("reports/m5b_figures/m5b_runtime_curves.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name in ["B_frozen", "C_context", "C_pair"]:
        rec = next(x for x in study["records"] if x["name"] == name and x["seed"] == 61)
        hist = [x for x in rec["history"] if "total" in x]
        ax[0].plot([x["step"] for x in hist], [x["total"] for x in hist], label=name)
        hist = [x for x in rec["history"] if "val_change_iou" in x]
        ax[1].plot(
            [x["step"] for x in hist], [x["val_change_iou"] for x in hist], label=name
        )
    ax[0].set_title("Actual training loss (seed61)")
    ax[1].set_title("Validation signed-change IoU")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("reports/m5b_figures/m5b_learning_curves.png", dpi=140)
    plt.close(fig)
    sources = {
        str(p): file_hash(p)
        for folder in [
            "tools",
            "models",
            "simulator",
            "training",
            "diagnostics",
            "datasets",
            "tests",
            "configs",
        ]
        for p in Path(folder).glob("*m5b*")
        if p.is_file()
    }
    reports = {
        str(p): file_hash(p)
        for p in Path("reports").glob("m5b*.json")
        if p != acceptance
    }
    final = [x for x in safety["records"] if x["run"] == "m5b_async_final_runtime"]
    atomic_json(
        acceptance,
        dict(
            stage="M5-B Counterfactual Visual Occupancy & Geometry",
            base="a252a04723a2c11f1189b3580a6d763345e1ae04",
            branch="feat/m5b-counterfactual-occupancy",
            overall="Partial",
            states={
                "engineering_interfaces": "Passed_within_tested_contract",
                "RGB_visual_learning_gain": "Observed_reproducible_on_three_test_groups_not_population_certification",
                "spatial_obstacle_generalization": "Partial",
                "runtime_realtime": "Partial",
                "Oracle_driving_safety": "Partial_tested_scenarios_collision_and_fault_protection_passed",
                "pure_visual_driving_safety": "NotValidated",
            },
            data=dict(
                total_captured_frames=180,
                independent_positions=12,
                train_frames=72,
                val_frames=36,
                formal_sealed_test_frames=36,
                unused_original_reserved_layout_frames=36,
                test_independent_positions=3,
                test_changed_witness_events=827,
                test_never_selected_training=True,
                static_actor_layouts=True,
                historical_maps=True,
            ),
            training=dict(
                models=["A_prior", "B_frozen", "C_context", "C_pair"],
                seeds=[61, 62],
                steps=400,
                matched_pair_order=True,
                depth_finetuning=False,
                original_encoder="runs/m4_expanded_joint/best.pt",
                original_encoder_sha256=file_hash("runs/m4_expanded_joint/best.pt"),
                geometry="unchanged48LID_and_HeightSplat",
                prior_tie_correction="same400step_replay_val_secondary_selection_not800step_training",
            ),
            visual={
                f"{x['name']}_{x['seed']}": x["summary"] for x in vision["records"]
            },
            runtime=dict(
                final_report="reports/m5b_async_final_runtime.json",
                max_final_control_interval_ms=max(
                    x["control_interval_ms"]["max"] for x in runtime["results"]
                ),
                final_deadline_misses=sum(
                    round(x["deadline_miss_rate"] * x["ticks"])
                    for x in runtime["results"]
                ),
                expired_normal_control=sum(
                    x["expired_plan_normal_control"] for x in final
                ),
                collisions=sum(x["collision"] for x in final),
                all_degradation_full_brake=all(
                    x["all_degradation_brake"] for x in final
                ),
                horizon_seconds=6,
                single_tick_owner=True,
                previous_tails_retained=True,
            ),
            tests=dict(
                full="reports/m5b_final_tests.xml",
                carla="reports/m5b_carla_api_tests.xml",
                types="synthetic/contracts/GPU plus actual CARLA separate reports",
            ),
            evidence_kinds=dict(
                capture="actualCARLA",
                training="actualRTX4090",
                geometry="offlineGPU_val_only",
                label_reference="same_renderer_point_witness_with_approx_AABB_gate_not_independentMesh",
                runtime="actualCARLA_wallclock",
            ),
            decisions=dict(
                expand_large_occupancy_training=False,
                start_temporal_or_sparse=False,
                next="Targeted single-frame geometry/context/witness coverage study with new sealed groups; maintain pair supervision and frozen protocol. No automatic Depth Bin/Splat change.",
            ),
            limitations=[
                "Only3 independent test locations; pixels/configurations not independent.",
                "Witness mask is partial visible surface support, not complete obstacle volume.",
                "15-27m and small targets sparse, lowest height layer zero changes.",
                "Context alone has one scene/seed regression; no claim of population statistical significance.",
                "Paired loss has substantial FP/FN; scores outside valid mask uncertified.",
                "Oracle nearest bin diagnostic is OOD and quantized; cannot uniquely identify Depth bottleneck.",
                "First geometry actor semantic mask included ego; target-only corrected audit supplied, no retraining.",
                "Final runtime run improved but earlier long tick stalls remain; no hard realtime.",
                "Static Actor oracle scenarios, no pure visual safety or dense dynamic traffic guarantee.",
            ],
            source_sha256=sources,
            reports_sha256=reports,
            reproduce="docs/M5B_IMPLEMENTATION.md",
        ),
    )
    atomic_json(
        "reports/m5b_result_registry.json",
        dict(
            authoritative_training="reports/m5b_context_ablation_corrected.json",
            authoritative_test="reports/m5b_counterfactual_detection.json",
            authoritative_target_geometry="reports/m5b_target_geometry.json",
            geometry_interventions="reports/m5b_geometry_diagnostics.json",
            authoritative_runtime="reports/m5b_async_final_runtime.json",
            runtime_attribution="reports/m5b_runtime_attribution_final.json",
            corrections=[
                "Initial prior all-tied change metric selected step0; same400step replay selects val event tie-break before test.",
                "Initial geometry feature actor tag statistics included ego; corrected target-only validation audit, no model/test selection.",
                "Pilot counterpart far-end smoothness incorrectly limited point absence; revised on training pilot before model training.",
                "BF16 exact allclose test inappropriate at isolated rounding boundaries; FP32 algebra equivalence and BF16 measured output error both supplied.",
            ],
            preserved_initial_reports=True,
        ),
    )


if __name__ == "__main__":
    main()
