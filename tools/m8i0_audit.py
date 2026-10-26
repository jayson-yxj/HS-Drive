"""Offline evidence inventory; no simulator, model, or third-party imports.

Run from repository root: python3 -m tools.m8i0_audit
"""
import ast
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

BASE = "69ce3158ab692387e615a781deb41aae0fb61499"
SOURCES = Path("reports/m8i0_sources")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name, data):
    Path("reports", name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    historical = []
    for stage in ["m8g", "m8h0", "m8h1"]:
        path = Path(f"reports/{stage}_release_manifest.json")
        data = json.loads(path.read_text())
        for item in data["files"]:
            assert sha(item["path"]) == item["sha256"], item["path"]
        historical.append(dict(path=str(path), sha256=sha(path), verified_entries=len(data["files"])))
    # An added review must not edit or delete any base-tracked asset or system file.
    changed = subprocess.check_output(
        ["git", "diff", "--name-status", BASE, "--diff-filter=MDTR"], text=True).strip()
    assert not changed, changed
    metadata = []
    for path in sorted(SOURCES.glob("*.meta.json")):
        item = json.loads(path.read_text())
        if item["status"] == "Fetched":
            payload = SOURCES / item["name"]
            assert sha(payload) == item["sha256"], str(payload)
            if "git_blob_sha" in item:
                b = payload.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()
                assert blob == item["git_blob_sha"], str(payload)
        metadata.append(item)
    installed = Path("/root/gpufree-data/simulators/CARLA_0.9.15/PythonAPI/examples/generate_traffic.py")
    official = SOURCES / "carlaapi__PythonAPI__examples__generate_traffic.py.txt"
    installed_check = dict(path=str(installed), exists=installed.is_file())
    if installed.is_file():
        installed_check.update(sha256=sha(installed), official_sha256=sha(official),
                               official_byte_match=installed.read_bytes() == official.read_bytes())
        assert installed_check["official_byte_match"]
    mini = json.loads((SOURCES / "b2dapi__docs__bench2drive_mini_10.json.txt").read_text())
    # Only parse literals. No import/exec/eval of the external data collector.
    tree = ast.parse((SOURCES / "b2dapi__tools__data_collect.py.txt").read_text())
    rgb = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sensors":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "sensors" for t in stmt.targets):
                    for entry in stmt.value.elts:
                        try:
                            value = ast.literal_eval(entry)
                        except (ValueError, TypeError):
                            continue
                        if value.get("type") == "sensor.camera.rgb":
                            rgb.append(value)
    assert len([x for x in rgb if x["id"] != "TOP_DOWN"]) >= 6
    catalog = []
    for kind, filename, source, risks in [
        ("FollowLeadingVehicle", "FollowLeadingVehicle", "follow_leading_vehicle.py",
         ["Waypoint-based lead +0.5m; provider adds +0.2m for non-prop; settle is unverified",
          "Lead 25m ahead is an intended opportunity, not a visibility certificate"]),
        ("DynamicObjectCrossing", "ObjectCrossing", "object_crash_vehicle.py",
         ["Up to 6 placement attempts; walker +1.2m; hidden/replaced actor and physics toggles",
          "Blocker physics disabled; post-trigger identity and natural state need evaluator"]),
        ("SignalizedJunctionLeftTurn", "SignalizedJunctionLeftTurn", "signalized_junction_left_turn.py",
         ["ActorFlow creates/destroys vehicles; TM and constant velocity used",
          "Traffic-light/sign ignore controls and signal freezing are scenario interventions"]),
    ]:
        xml = SOURCES / f"srapi__srunner__examples__{filename}.xml.txt"
        scenarios = [dict(s.attrib, ego=dict(s.find("ego_vehicle").attrib))
                     for s in ET.fromstring(xml.read_text()).findall("scenario") if s.get("type") == kind]
        catalog.append(dict(type=kind, source_file=source, xml=str(xml),
                            scenarios=scenarios, count=len(scenarios), risks=risks,
                            status="ScenarioExecutionCandidate_GTPartial", runtime="NotRun",
                            six_camera_dataset_export="NotAvailable_BuiltIn",
                            independent_evaluator="Required_NotImplemented"))
    write("m8i0_scenariorunner_scenario_catalog.json", dict(
        kind="OfficialPinnedSourceAndXMLReview", checked_date="2026-10-26",
        version="v0.9.15", commit="d12d8bb788201ebf6f03e55b9c5169f9dcb2ea87",
        installation="NotRun", scenarios=catalog))
    write("m8i0_evidence_inventory.json", dict(
        kind="ActualOfflineReadOnlyAudit", checked_at_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=BASE, historical_seals=historical, base_tracked_modifications=[],
        installed_generate_traffic=installed_check, sources=metadata,
        fetched_bytes=sum(x.get("bytes", 0) for x in metadata),
        unavailable_count=sum(x["status"] != "Fetched" for x in metadata),
        b2d_mini_declared=dict(clips=len(mini), compressed_bytes=sum(v["size"] for v in mini.values()),
                               files=mini, payloads_downloaded=0),
        b2d_camera_source_literals=rgb,
        original_raw_frames_and_timestamps="NotAvailable_InReviewedB2DSaveFunction",
        no_external_source_execution=True, new_carla_connections=0,
        new_actors=0, new_sensors=0, new_RGB=0, new_GT=0, model_forwards=0, training_steps=0))
    print("Verified historical entries:", sum(x["verified_entries"] for x in historical))
    print("Verified source payloads:", sum(x["status"] == "Fetched" for x in metadata))
    print("Catalog:", [(x["type"], x["count"]) for x in catalog])


if __name__ == "__main__":
    main()
