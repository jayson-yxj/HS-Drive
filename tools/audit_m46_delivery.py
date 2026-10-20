"""Aggregate immutable M4.6 evidence; keep quality failures separate from IO passes."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from simulator.recorder import atomic_json, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def motion_supplement(reports):
    results = []
    for report in reports:
        for row in read(report)["results"]:
            data = [json.loads(x) for x in Path(row["log"]).read_text().splitlines()]
            if file_hash(row["log"]) != row["log_sha256"]:
                raise ValueError("Driving log changed")
            v = np.array([x["v"] for x in data])
            t = np.array([x["t"] for x in data])
            a = np.diff(v) / np.diff(t)
            jerk = np.diff(a) / np.diff(t[:-1])
            moving = (v[:-2] > 0.2) | (v[1:-1] > 0.2) | (v[2:] > 0.2)
            brake = next(
                (
                    i
                    for i, x in enumerate(data)
                    if x["command"]["brake"] > 0
                    and ("gap" in row["settings"] or x["t"] > 1)
                ),
                None,
            )
            stop = next(
                (
                    i
                    for i in range(len(data) - 20)
                    if t[i] > 1 and np.all(v[i : i + 20] < 0.08)
                ),
                None,
            )
            poses = np.array([[x["x"], x["y"]] for x in data])
            distance = np.linalg.norm(np.diff(poses, axis=0), axis=1)
            results.append(
                dict(
                    version=row["version"],
                    scenario=row["scenario"],
                    first_stable_stop_time_s=None if stop is None else float(t[stop]),
                    distance_to_first_stable_stop_m=(
                        None if stop is None else float(distance[:stop].sum())
                    ),
                    first_brake_gap_m=None if brake is None else data[brake]["gap"],
                    jerk_abs_p95_moving_mps3=(
                        None
                        if not moving.any()
                        else float(np.quantile(abs(jerk[moving]), 0.95))
                    ),
                    jerk_abs_max_raw_mps3=float(abs(jerk).max()),
                    max_raw_decel_mps2=float(max(0, -a.min())),
                    source_log=row["log"],
                    source_sha256=file_hash(row["log"]),
                )
            )
    return dict(
        results=results,
        scope="Measured velocities and timestamps. Moving jerk excludes all-stopped triplets (<0.2m/s); report alongside full-trace jerk. First sustained stop: 20 ticks <0.08m/s; early unintended stops may qualify.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/m46_acceptance.json")
    parser.add_argument(
        "--supplement", default="reports/m46_motion_supplement_verified.json"
    )
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.supplement).exists():
        raise FileExistsError("New outputs required")
    motion = "reports/m46_closedloop_verified.json"
    routes = "reports/m46_routes_verified.json"
    occupancy = "reports/m46_label_verified.json"
    temporal = "reports/m46_temporal_verified.json"
    loader = "reports/m46_loader_release.json"
    safety = "reports/m46_safety_verified.json"
    assets = "reports/m46_asset_integrity.json"
    reports = [motion, routes, occupancy, temporal, loader, safety, assets]
    braking = read(motion)["results"]
    new = [r for r in braking if r["version"] == "m46" and "gap" in r["settings"]]
    old = [r for r in braking if r["version"] == "m45" and "gap" in r["settings"]]
    all_driving = braking + read(routes)["results"]
    policy = read(motion)["policy"]
    tests = {}
    for group, path in [
        ("gpu", "reports/m46_full_gpu_tests.xml"),
        ("sim", "reports/m46_full_sim_tests.xml"),
        ("m46_modules", "reports/m46_module_tests_release.xml"),
    ]:
        suite = ET.parse(path).getroot().find("testsuite")
        tests[group] = {**suite.attrib, "path": path, "sha256": file_hash(path)}
    supplement = motion_supplement([motion, routes])
    atomic_json(args.supplement, supplement)
    checks = dict(
        baseline_assets_preserved=read(assets)["passed"],
        labels_engineering=read(occupancy)["engineering_status"],
        labels_quality=read(occupancy)["strict_label_quality_status"],
        masked_loader_and_loss=read(loader)["passed"],
        temporal_oracle_support=read(temporal)["passed"],
        controlled_driving_passed=all(r["status"] == "Passed" for r in all_driving),
        all_driving_collision_free=all(r["collision_count"] == 0 for r in all_driving),
        moving_stop_raw_decel_within_planned_bound=all(
            r["max_decel_raw_mps2"] <= policy["comfortable_decel"] for r in new
        ),
        moving_stop_raw_jerk_within_planned_bound=all(
            r["jerk_abs_max_raw_mps3"] <= policy["jerk"] for r in new
        ),
        exact_continuous_collision_detection=False,
        temporal_learned_deployment_validated=False,
        source_stable=all(
            read(p)["source_unchanged_during_run"] for p in [motion, routes]
        ),
    )
    atomic_json(
        args.output,
        dict(
            schema_version="hsd-m46-acceptance-v1",
            overall="Partial",
            base_commit="129a4131567ac02edf88b19b9199cad979ce3d72",
            branch="feat/m46-occupancy-motion-quality",
            m4_gate_closed=False,
            m45_gate_closed=False,
            m46_gate_closed=False,
            m5_training_approved=False,
            network_trained=False,
            gated_splat_default=False,
            checks=checks,
            tests=tests,
            closedloop=dict(
                pairs=len(all_driving) // 2,
                new_stopping_gap_range_m=[
                    min(r["final_gap_m"] for r in new),
                    max(r["final_gap_m"] for r in new),
                ],
                old_obstacle_safety_ticks=sum(
                    r["safety_forced_braking_ticks"] for r in old
                ),
                new_obstacle_safety_ticks=sum(
                    r["safety_forced_braking_ticks"] for r in new
                ),
                scope="Privileged low-speed Town03 planner/controller. Green-light route subset not traffic-rule validation. Whole-stack comparison, not one-variable ablation.",
            ),
            limits=[
                "Observed false-free remains nonzero; heldout pixels are not independent 3D mesh truth.",
                "Only 36 static and 3 scripted-dynamic frames, one map. Route-group validation not blind.",
                "Actual deceleration/jerk exceed planned limits; no natural/comfortable-stop acceptance.",
                "Junction still triggers road-boundary braking and exceeds 50ms p95 planner+safety budget.",
                "Occupancy has not been connected as the obstacle source for closed-loop; Oracle actors remain privileged.",
            ],
            evidence={p: file_hash(p) for p in reports + [args.supplement]},
        ),
    )


if __name__ == "__main__":
    main()
