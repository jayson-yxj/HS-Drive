"""Assemble actual M5-O outcomes; absent experiments explicitly NotRun."""

import json
from pathlib import Path
from tools.m5o_common import write, asset, provenance
from datasets.m5o_benchmark import support


def read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    dataset = read("reports/m5o_dataset_support.json")
    training = read("reports/m5o_training.json")
    val = read("reports/m5o_eval_val.json")
    cal = read("reports/m5o_eval_calibration.json")
    sealed = read("reports/m5o_eval_sealed.json")
    lock = read("reports/m5o_calibration_lock.json")
    frozen = read("reports/m5o_oracle_candidate_diagnostic.json")
    prov = provenance()
    cqs = [
        dict(
            split=s,
            report=asset(f"reports/m5o_candidate_quality_{s}.json"),
            result=read(f"reports/m5o_candidate_quality_{s}.json"),
        )
        for s in ["regression", "smoke_v3", "train", "val", "calibration", "sealed"]
        if Path(f"reports/m5o_candidate_quality_{s}.json").exists()
    ]
    write(
        "reports/m5o_candidate_quality.json",
        dict(
            kind="actual_GPU_and_offline_all_candidate_audit",
            datasets=cqs,
            provenance=prov,
            physical_false_positive="NotIdentifiable_without_complete_scene_geometry",
            empty_scope="no inserted actor, not physical empty scene",
        ),
    )
    write(
        "reports/m5o_data_scaling.json",
        dict(
            status=(
                "Completed"
                if training and training["status"] == "Completed"
                else "NotRun"
            ),
            training=training,
            validation=val,
            provenance=prov,
            limits="25/50/100 nested independent position groups. Validation used for selection; no test-based budget extension. No causal data-size claim if support gate fails.",
        ),
    )
    write(
        "reports/m5o_localization_benchmarks.json",
        dict(
            status="Completed" if sealed else "Partial",
            validation=val,
            sealed=sealed,
            historical_regression=frozen,
            provenance=prov,
        ),
    )
    write(
        "reports/m5o_independent_calibration.json",
        dict(
            status="Completed" if cal else "NotRun",
            calibration=cal,
            lock=lock,
            provenance=prov,
            limits="Calibration groups disjoint from model-selection Validation; empirical cluster-heldout error proxy, not iid conformal/safety guarantee",
        ),
    )
    write(
        "reports/m5o_risk_coverage.json",
        dict(
            status="Completed" if sealed else "Partial" if cal else "NotRun",
            calibration=None if not cal else cal["risks"],
            sealed=None if not sealed else sealed["risks"],
            provenance=prov,
        ),
    )
    support_ok = bool(dataset and dataset["passed"])
    both = bool(lock and all(x["promoted"] for x in lock["calibration"].values()))
    test_ok = bool(
        sealed
        and len(sealed["curves"]) == 2
        and all(r["passed"] for r in sealed["curves"])
    )
    sealed_support = []
    if Path("runs/m5o_cache_sealed/supervision.json").exists():
        c = read("runs/m5o_cache_sealed/candidates.json")
        s = read("runs/m5o_cache_sealed/supervision.json")
        sealed_support = [
            dict(seed=seed, **support(c, s["labels"], seed)) for seed in [101, 102]
        ]
        support_ok &= all(
            r["groups"] >= 6 and r["unique_candidates"] >= 80 for r in sealed_support
        )
    else:
        support_ok = False
    tests = Path("runs/m5o_logs/full_tests_final.log")
    testtext = tests.read_text() if tests.exists() else ""
    acceptance = dict(
        stage="M5-O",
        overall="Partial",
        engineering_interfaces=(
            "Passed" if "passed" in testtext and "failed" not in testtext else "Pending"
        ),
        dataset_support="Passed_research_support_only" if support_ok else "Partial",
        finite_training=(
            "Completed" if training and training["status"] == "Completed" else "NotRun"
        ),
        visual_localization_gain=(
            "Passed_limited_offline"
            if both and test_ok and support_ok
            else "NoGo_or_unresolved"
        ),
        independent_risk_calibration=(
            "Passed_limited_offline" if both and test_ok else "NoGo_or_unresolved"
        ),
        offline_object_center_pilot=(
            "Eligible_limited_offline_only"
            if both and test_ok and support_ok
            else "NotApproved"
        ),
        direct_head_route=(
            "Continue_limited_research"
            if both and test_ok and support_ok
            else "Stop_current_route_no_automatic_head_or_loss_expansion"
        ),
        occupancy="NotValidated_NoTraining",
        contactGold="Unchanged_NotValidated",
        runtime="Inherited_M5M_Partial",
        oracle_control="Unchanged",
        pure_visual_control="Prohibited_NotValidated",
        sealed_support=sealed_support,
        main_merged=False,
        provenance=prov,
        tests=testtext.strip().splitlines()[-1] if testtext else "Pending",
    )
    write("reports/m5o_acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
