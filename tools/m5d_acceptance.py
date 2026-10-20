"""Independent M5-D gates, never propagating a training success to safety."""

import json, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
from simulator.recorder import atomic_json, file_hash


def read(name):
    return json.loads(Path("reports", name + ".json").read_text())


def main():
    dest = Path("reports/m5d_acceptance.json")
    if dest.exists():
        raise FileExistsError(dest)
    digest = read("m5d_evaluation_digest")
    training = read("m5d_model_ablation")
    runtime = read("m5d_runtime_stress")
    faults = read("m5d_runtime_faults")
    asset = read("m5d_assets")
    protocol = read("m5d_pretest_decision")
    current = {k: file_hash(k) == v for k, v in protocol["checkpoints"].items()}
    code = {
        k: file_hash(k) == v
        for k, v in protocol["code_sha256"].items()
        if k
        in [
            "tools/m5d_learning.py",
            "tools/m5d_evaluate.py",
            "datasets/m5d_witness.py",
            "models/m5d_stride8.py",
            "diagnostics/m5d_metrics.py",
            "training/m5d_existence_loss.py",
        ]
    }
    tests = []
    for f in ["reports/m5d_tests.xml", "reports/m5d_carla_api_tests.xml"]:
        s = ET.parse(f).getroot().find("testsuite")
        tests.append(
            dict(
                path=f,
                **{
                    k: int(s.get(k, 0))
                    for k in ["tests", "failures", "errors", "skipped"]
                },
                sha256=file_hash(f),
            )
        )
    rr = [r for c in runtime["records"] for r in c.get("results", [])] + faults[
        "results"
    ]
    errors = {
        k: sum(r[k] for r in rr)
        for k in [
            "collisions",
            "lane_invasions",
            "road_boundary_frames",
            "normal_control_expired",
        ]
    }
    models = []
    for r in digest["records"]:
        if r["name"] not in ["B_existence", "C_stable", "S8_stable"]:
            continue
        p = all(
            s["n"] > 0 and s["presence_recall"] >= 0.8 for s in r["by_scene"].values()
        )
        a = all(s["n"] > 0 and s["absence_fpr"] <= 0.1 for s in r["by_scene"].values())
        models.append(
            dict(
                name=r["name"],
                seed=r["seed"],
                each_scene_presence=p,
                each_scene_absence=a,
                change_iou_gate=r["mean_scene_change_iou"] >= 0.5,
                far_recall_support_insufficient=True,
            )
        )
    engineering = (
        all(t["errors"] == t["failures"] == 0 for t in tests)
        and read("m5d_gpu_contracts")["passed"]
        and all(current.values())
        and all(code.values())
        and asset["old_assets_preserved"]
    )
    gates = dict(
        engineering_interfaces=dict(
            status="Passed" if engineering else "Partial",
            scope="tested masks,gradient isolation,RGB-only singleframe path,immutable historical source/checkpoints",
        ),
        research_training=dict(
            status=(
                "Passed"
                if all(
                    r["finite_gradients"] and r["frozen_gradients_absent"]
                    for r in training["records"]
                )
                else "Partial"
            ),
            scope="10 actual RTX4090 runs x400steps; cached features; not generalization",
        ),
        presence_quality=dict(
            status="Partial",
            reason="Strong nearfield gain, scene/seed and medium/far miss remains",
        ),
        absence_quality=dict(
            status="Partial",
            reason="Certified nearfield residual reduced; no training 20-35m support, test only6 repeated point events; scene thresholds fail",
        ),
        absolute_localization=dict(
            status="Partial",
            reason="Actor surface recall and weak detection remain incomplete; small pedestrian/trafficcone miss",
        ),
        geometry_generalization=dict(
            status="Partial",
            reason="TrueP8 improves selected recalls, depth distribution still misprojects; no independentMesh or unseenweather proof",
        ),
        runtime_stability=dict(
            status="Partial",
            reason="At least one maxinterval/deadline gate fails; mean20Hz insufficient",
        ),
        oracle_closed_loop=dict(
            status="Partial",
            reason="Observed safety/fault counts recorded; long routes finite budget, dynamics/continuouscollision/generaltraffic not passed",
            actual_counts=errors,
        ),
        pure_visual_closed_loop=dict(
            status="NotValidated",
            reason="Visual evidence never drives safety/controller; Oracle uses privileged route/state/actors",
        ),
    )
    atomic_json(
        dest,
        dict(
            task="M5-D",
            status="Partial",
            base="2aa91f6",
            source_commit=asset["source_commit"],
            branch=asset["branch"],
            gates=gates,
            model_quality_checks=models,
            tests=tests,
            all_training_steps=4000,
            runtime_ticks=sum(r["ticks"] for r in rr),
            checkpoint_freeze_matches=current,
            scientific_code_freeze_matches=code,
            protocol_hash_matches=file_hash("configs/m5d_protocol.json")
            == protocol["protocol_sha256"],
            old_assets_preserved=asset["old_assets_preserved"],
            expand_formal_occupancy_training=False,
            test_reused_for_tuning=False,
            source_sha256=file_hash(__file__),
            evidence={
                "supervision": "reports/m5d_supervision_audit.json",
                "labels": "reports/m5d_dataset_quality.json",
                "models": "reports/m5d_model_ablation.json",
                "predictions": "reports/m5d_presence_absence.json",
                "geometry": "reports/m5d_geometry_resolution.json",
                "runtime": "reports/m5d_runtime_stress.json",
                "runtime_attribution": "reports/m5d_runtime_attribution.json",
                "faults": "reports/m5d_runtime_faults.json",
                "assets": "reports/m5d_assets.json",
            },
            next_steps=[
                "Independent future protocol to distinguish smooth slanted far-background from occlusion discontinuities; no mask relaxation without geometric tests",
                "Add independent mid/far visiblepositive+certifiednegative target support before any scaleup",
                "Keep P8 optional; controlled depth-distribution localization study separate from architecture changes",
                "Versioned affinity/quiet-host schedule experiments; preserve 6sSafety and original timestamp expiry",
                "Do not start Temporal/SparseQuery to hide unresolved singleframe localization",
            ],
        ),
    )


if __name__ == "__main__":
    main()
