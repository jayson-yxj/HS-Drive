"""Independent quality, comfort, position, timing and training gates."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,yaml
from simulator.recorder import atomic_json,file_hash
from tools.m47_common import dynamics_metrics


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="reports/m47_acceptance.json")
    p.add_argument("--metrics",default="reports/m47_stopping_metrics_verified.json");a=p.parse_args()
    if Path(a.output).exists() or Path(a.metrics).exists():raise FileExistsError("New outputs required")
    gates=yaml.safe_load(Path("configs/m47_acceptance.yaml").read_text())
    driving=read("reports/m47_stopping_verified.json")["results"]+read("reports/m47_routes_verified.json")["results"]
    details=[]
    for result in driving:
        assert file_hash(result["log"])==result["log_sha256"]
        rows=[json.loads(s) for s in Path(result["log"]).read_text().splitlines()]
        measurements=[r["measurement"] for r in rows];m=dynamics_metrics(measurements)
        entry=next((r for r in rows if r["hold_entry"]),None)
        hold=None if entry is None else dict(**entry["hold_entry"],gap=entry["gap"],position=[entry["x"],entry["y"]])
        normal="gap" in result["settings"]
        gap=result["final_gap_m"];g=gates["stopping"]
        checks=dict(collision_free=result["collision_count"]==0,lane_free=result["lane_invasions"]==0,
            scenario_stop=result["status"]=="Passed",
            raw_decel=m["max_decel_raw"]<=g["normal_max_raw_decel_mps2"] if normal else None,
            moving_jerk_p95=m["jerk_p95_moving"]<=g["normal_moving_jerk_p95_mps3"] if normal else None,
            jerk_peak=m["jerk_peak_raw"]<=g["normal_jerk_peak_mps3"] if normal else None,
            stop_position=abs(gap-3)<=g["final_gap_error_abs_m"] if gap is not None else None,
            minimum_clearance=result["min_gap_m"]>=g["min_gap_m"] if gap is not None else None,
            hold_entry_low_speed=hold["speed"]<=g["hold_entry_speed_mps"] if hold else None,
            no_reverse=m["min_signed_speed"]>=-g["reverse_speed_limit_mps"],
            no_prolonged_coast=m["low_speed_moving_seconds"]<=g["low_speed_motion_limit_s"],
            p95_20hz=result["latency_p95_ms"]<=50)
        details.append(dict(version=result["version"],scenario=result["scenario"],metrics=m,checks=checks,
            hold_entry=hold,final_gap_m=gap,safety_forced_braking_events=result["safety_forced_braking_events"],
            safety_forced_braking_ticks=result["safety_forced_braking_ticks"],latency_p95_ms=result["latency_p95_ms"],
            source_log=result["log"],source_sha256=file_hash(result["log"])))
    atomic_json(a.metrics,dict(kind="offline_metrics_from_actual_carla",records=details,criteria=gates))
    physics_equal=[]
    for folder in ["reports/m47_stopping_verified_artifacts","reports/m47_routes_verified_artifacts"]:
        for p in Path(folder).glob("m46_*_physics.json"):
            other=p.with_name(p.name.replace("m46_","m47_",1))
            physics_equal.append(read(p)==read(other))
    tests={}
    for group,path in [("gpu","reports/m47_gpu_tests.xml"),("sim","reports/m47_sim_tests.xml"),("m47","reports/m47_module_tests.xml")]:
        root=ET.parse(path).getroot().find("testsuite");tests[group]={**root.attrib,"path":path}
    quality=read("reports/m47_occupancy_quality.json")
    new=[r for r in details if r["version"]=="m47"];normal=[r for r in new if r["final_gap_m"] is not None]
    sources=["reports/m47_false_free_audit.json","reports/m47_false_free_attribution_verified.json",
        "reports/m47_collision_audit.json","reports/m47_occupancy_quality.json","reports/m47_vehicle_dynamics.json",
        "reports/m47_dynamics_calibration.json","reports/m47_stopping_verified.json","reports/m47_routes_verified.json",
        "reports/m47_latency_verified.json","reports/m47_road_profile.json","reports/m47_gpu_contract_verified.json",
        "reports/m47_asset_integrity.json",a.metrics,"configs/m47_acceptance.yaml"]
    atomic_json(a.output,dict(schema="hsd-m47-acceptance-v1",overall="Partial",
        base_commit="e9e2e02097a77123138c3b802909c9111faf11b4",branch="feat/m47-label-assurance-dynamics",
        m4_gate_closed=False,m45_gate_closed=False,m46_gate_closed=False,m47_gate_closed=False,m5_training_approved=False,
        default_behavior_changed=False,optimizer_created=False,depth_bins_changed=False,
        gates=dict(all_104_provenance_reconstructed=True,whole_voxel_free_certified=False,
            independent_geometry_complete=False,reserved_scene_quality_pass=False,
            supervised_gpu_contract_pass=read("reports/m47_gpu_contract_verified.json")["unknown_gradient_zero"],
            matched_physics=all(physics_equal),all_collision_free=all(r["checks"]["collision_free"] for r in new),
            obstacle_stopping_position=all(r["checks"]["stop_position"] and r["checks"]["minimum_clearance"] for r in normal),
            comfort_decel=all(r["checks"]["raw_decel"] for r in normal),
            comfort_jerk=all(r["checks"]["moving_jerk_p95"] and r["checks"]["jerk_peak"] for r in normal),
            stop_hold_entry=all(r["checks"]["hold_entry_low_speed"] for r in normal),
            route_completion=all(r["checks"]["scenario_stop"] for r in new),
            latency_20hz=all(r["checks"]["p95_20hz"] for r in new),
            exact_rotating_mesh_collision=False,overfit_phase="Deferred: spatial quality prerequisite not met"),
        reserved_scene_collision_conflicts=sum(r["free_conflicts"] for r in quality["records"]),
        tests=tests,evidence={p:file_hash(p) for p in sources},
        decisions=["Retain M46 default; M47 controller and road corridor remain experimental.",
            "Quarantine only evidenced conflicting cells; do not interpret repair-set zero conflicts as validation.",
            "M5 formal training and conditional overfit deferred. GPU backward is not overfit success.",
            "20Hz is retained as target, not claimed achieved for junction. No untested multirate scheduling adopted."]))


if __name__=="__main__":main()
