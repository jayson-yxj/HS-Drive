"""Read-only final artifact, source, controlled-state and baseline verification."""

import argparse
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from simulator.recorder import atomic_json, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/m47_delivery_integrity.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    errors = []
    freeze = read("reports/m47_asset_freeze.json")
    for category in ("tracked_sha256", "checkpoint_sha256"):
        for name, expected in freeze[category].items():
            if not Path(name).is_file() or file_hash(name) != expected:
                errors.append([category, name])
    for name, expected in freeze["data_size_mtime"].items():
        p = Path(name)
        if not p.is_file() or [p.stat().st_size, p.stat().st_mtime_ns] != expected:
            errors.append(["old_data_size_mtime", name])
    for name, expected in read("reports/m47_data_manifest.json")["external_files"].items():
        if not Path(name).is_file() or file_hash(name) != expected:
            errors.append(["external_m47_sha256", name])
    for name, expected in read("reports/m47_acceptance.json")["evidence"].items():
        if not Path(name).is_file() or file_hash(name) != expected:
            errors.append(["acceptance_evidence", name])
    snapshots = {}
    for manifest in Path("reports").glob("m47*/**/manifest.json"):
        entries = read(manifest)
        if not isinstance(entries, dict) or not entries or not all(
            isinstance(value, str) and len(value) == 64 for value in entries.values()
        ):
            continue
        snapshots[str(manifest)] = len(entries)
        for name, expected in entries.items():
            path = manifest.parent / Path(name).name
            if not path.is_file() or file_hash(path) != expected:
                errors.append(["source_snapshot", str(path)])

    controlled = []
    for report_path in ("reports/m47_stopping_verified.json", "reports/m47_routes_verified.json"):
        report = read(report_path)
        for name, expected in report["source_hashes_at_start"].items():
            if file_hash(name) != expected:
                errors.append(["closedloop_runtime_source", name])
        results = report["results"]
        for left, right in zip(results[::2], results[1::2]):
            assert left["version"] == "m46" and right["version"] == "m47"
            assert left["scenario"] == right["scenario"]
            pair = []
            for result in (left, right):
                if file_hash(result["log"]) != result["log_sha256"]:
                    errors.append(["closedloop_log", result["log"]])
                pair.append(json.loads(Path(result["log"]).read_text().splitlines()[0]))
            a, b = pair
            gap_difference = None if a["gap"] is None else b["gap"] - a["gap"]
            actors = []
            for oa, ob in zip(a["obstacles"], b["obstacles"]):
                actors.append(dict(
                    position_delta=[y - x for x, y in zip(oa["xy"], ob["xy"])],
                    velocity_delta=[y - x for x, y in zip(oa["velocity"], ob["velocity"])],
                ))
            matches = (
                len(a["obstacles"]) == len(b["obstacles"])
                and all(abs(a[k] - b[k]) < 1e-6 for k in ("x", "y", "yaw", "v"))
                and all(abs(d) < 1e-6 for obj in actors for ds in obj.values() for d in ds)
            )
            controlled.append(dict(
                scenario=left["scenario"], initial_states_match=matches,
                ego_speed_difference=b["v"] - a["v"],
                initial_gap_difference_m=gap_difference, actors=actors,
                status="Matched" if matches else "Not strictly controlled; descriptive comparison only",
            ))
    sources = sorted(
        p for folder in ("tools", "simulator", "datasets", "tests", "configs")
        for p in Path(folder).glob("*m47*") if p.is_file()
    )
    current = {str(p): file_hash(p) for p in sources}
    deltas = {}
    for name, digest in read("reports/m47_final_source/manifest.json").items():
        if file_hash(name) != digest:
            old = Path("reports/m47_final_source") / Path(name).name
            deltas[name] = dict(
                current_sha256=file_hash(name), runtime_snapshot_sha256=digest,
                python_ast_identical=ast.dump(ast.parse(old.read_text())) == ast.dump(ast.parse(Path(name).read_text())),
            )
    tests = {}
    for group, name in (
        ("sim", "reports/m47_release_sim_tests.xml"),
        ("gpu", "reports/m47_release_gpu_tests.xml"),
        ("module", "reports/m47_review_module_tests.xml"),
    ):
        suite = ET.parse(name).getroot().find("testsuite")
        tests[group] = dict(**suite.attrib, path=name, sha256=file_hash(name))
        if int(suite.attrib["errors"]) or int(suite.attrib["failures"]):
            errors.append(["pytest", name])
    atomic_json(args.output, dict(
        kind="offline_delivery_integrity_and_controlled_state_review",
        assets_and_evidence_unchanged=not errors, mismatches=errors,
        baseline_counts={k: len(freeze[k]) for k in ("tracked_sha256", "checkpoint_sha256", "data_size_mtime")},
        external_m47_sha256_count=len(read("reports/m47_data_manifest.json")["external_files"]),
        runtime_snapshots=snapshots, source_sha256=current,
        changes_after_runtime_snapshot=deltas,
        change_explanation="Dynamics comment-only correction; visualization now replays missing M46 HOLD state. No runtime controller or label-generation changes.",
        tests=tests, controlled_initial_states=controlled,
        all_initial_states_matched=all(r["initial_states_match"] for r in controlled),
        quality_gate="Partial", m5_formal_training_approved=False,
        limitation="Old raw data checked by size/mtime, not exhaustive content hashing. Dynamic lead initialization mismatch is retained as a protocol failure.",
    ))
    print(json.dumps(dict(integrity_passed=not errors, initial_state_mismatches=[
        r["scenario"] for r in controlled if not r["initial_states_match"]
    ], errors=errors)))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
