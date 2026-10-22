"""Finalize reports and hash-preserve historical assets after all measured results."""

import json, subprocess
from pathlib import Path
from tools.m5n_common import write, asset
from simulator.recorder import file_hash


def main():
    tests = Path("runs/m5n_logs/full_tests_final.log").read_text()
    sdk = Path("runs/m5n_logs/sdk_tests_final.log").read_text()
    assert "failed" not in tests and "passed" in tests
    assert "failed" not in sdk and "passed" in sdk
    matching = json.loads(Path("reports/m5n_matching_complete.json").read_text())
    p = Path("reports/m5n_candidate_matching.json")
    d = json.loads(p.read_text())
    d["all_set_totals"] = matching["records"]
    d["complete"] = asset("reports/m5n_matching_complete.json")
    p.write_text(json.dumps(d, indent=2) + "\n")
    p = Path("reports/m5n_actor_center_reference.json")
    d = json.loads(p.read_text())
    d["sealed"] = matching["sealed_reference"]
    d["sealed_capture_integrity"] = asset("reports/m5n_sealed_capture_quality.json")
    p.write_text(json.dumps(d, indent=2) + "\n")
    p = Path("reports/m5n_acceptance.json")
    d = json.loads(p.read_text())
    d["tests"] = dict(
        full=tests.strip().splitlines()[-1],
        sdk=sdk.strip().splitlines()[-1],
        full_log=asset("runs/m5n_logs/full_tests_final.log"),
        sdk_log=asset("runs/m5n_logs/sdk_tests_final.log"),
    )
    p.write_text(json.dumps(d, indent=2) + "\n")
    p = Path("reports/m5n_summary.md")
    p.write_text(
        p.read_text()
        + f"\n测试：{tests.strip().splitlines()[-1]}；SDK：{sdk.strip().splitlines()[-1]}。\n\n图：[训练与泛化](m5n_figures/m5n_training.png)、[风险覆盖](m5n_figures/m5n_risk_coverage.png)、[方法对比](m5n_figures/m5n_localization.png)、[参考语义](m5n_figures/m5n_reference_semantics.png)。\n"
    )
    baseline = json.loads(Path("runs/m5n_logs/baseline.json").read_text())
    bad = []
    for kind in ["tracked", "assets"]:
        for p, h in baseline[kind].items():
            if not Path(p).exists() or file_hash(p) != h:
                bad.append(p)
    assert not bad, bad
    frozen = json.loads(Path("reports/m5n_pretest_freeze.json").read_text())
    changed = [p for p, h in frozen["sources"].items() if file_hash(p) != h]
    assert not changed, changed
    paths = []
    for root in [
        "models",
        "datasets",
        "tools",
        "diagnostics",
        "configs",
        "reports",
        "docs",
        "tests",
        "runs",
    ]:
        paths.extend(
            p
            for p in Path(root).rglob("*")
            if p.is_file() and "m5n" in str(p).lower() and "__pycache__" not in str(p)
        )
    for root in ["raw", "processed"]:
        for folder in Path("/root/gpufree-data/hs-drive-data", root).glob("m5n*"):
            paths.extend(p for p in folder.rglob("*") if p.is_file())
    ix = {
        str(p): dict(sha256=file_hash(p), bytes=p.stat().st_size)
        for p in sorted(set(paths))
    }
    dst = "/root/gpufree-data/hs-drive-data/m5n_asset_sha_index.json"
    write(dst, dict(assets=ix))
    write(
        "reports/m5n_artifact_manifest.json",
        dict(
            base=baseline["base"],
            source_hashes={p: v["sha256"] for p, v in ix.items() if p.endswith(".py")},
            models=frozen["models"],
            assets=len(ix),
            index=asset(dst),
            frozen_experiment_sources_changed=changed,
        ),
    )
    write(
        "reports/m5n_integrity.json",
        dict(
            historical_tracked=len(baseline["tracked"]),
            historical_assets=len(baseline["assets"]),
            changed=bad,
            base=baseline["base"],
            index=asset(dst),
        ),
    )
    print(
        "Integrity passed", len(baseline["tracked"]), len(baseline["assets"]), len(ix)
    )


if __name__ == "__main__":
    main()
