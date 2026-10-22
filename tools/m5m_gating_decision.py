"""Lock validation gate decisions without tuning on old or fresh tests."""

import json
from pathlib import Path
from tools.m5l_common import write, asset


def decide(r):
    ix = {(v["method"], v["scope"], v["band"]): v for v in r["depth"]}
    bev = {v["method"]: v for v in r["bev"]}
    checks = []
    for mode in ["no_mobile", "mobile"]:
        for threshold in [0.5, 0.9, 0.95]:
            for seed in [101, 102]:
                name = f"{mode}_rgb{threshold}_{seed}"
                base = f"original_{seed}"

                def v(method, scope, band, key):
                    return ix[(method, scope, band)][key]

                near = v(name, "all", "near", "expected_mae")
                near0 = v(base, "all", "near", "expected_mae")
                gain = v(name, "vehicle", "20_35", "space1") - v(
                    base, "vehicle", "20_35", "space1"
                )
                near_actor = v(name, "actor", "near", "space1") - v(
                    base, "actor", "near", "space1"
                )
                bias = {
                    band: abs(v(name, "all", band, "bias"))
                    - abs(v(base, "all", band, "bias"))
                    for band in ["20_35", "35_80"]
                }
                b = bev[name]
                b0 = bev[base]
                delta = {k: b[k] - b0[k] for k in ["precision", "recall1", "fpr"]}
                pidx = {
                    (p["method"], p["scope"], p["band"], p["group"]): p
                    for p in r["by_position"]
                }
                fails = []
                for group in sorted({p["group"] for p in r["by_position"]}):
                    for band in ["near", "20_35", "35_80"]:
                        a = pidx.get((name, "all", band, group))
                        old = pidx.get((base, "all", band, group))
                        if a and old:
                            if (
                                band == "near"
                                and a["expected_mae"] > 1.05 * old["expected_mae"]
                            ):
                                fails.append([group, band, "near_mae"])
                            if (
                                band != "near"
                                and abs(a["bias"]) > abs(old["bias"]) + 0.5
                            ):
                                fails.append([group, band, "bias"])
                passed = (
                    near <= near0 * 1.05
                    and near_actor >= 0
                    and gain >= 0.01
                    and max(bias.values()) <= 0.5
                    and delta["precision"] >= 0.01
                    and delta["recall1"] >= 0.05
                    and delta["fpr"] <= 0.001
                    and not fails
                )
                checks.append(
                    dict(
                        mode=mode,
                        threshold=threshold,
                        seed=seed,
                        near_mae=near,
                        near_original=near0,
                        near_actor_space_delta=near_actor,
                        vehicle_space_delta=gain,
                        far_abs_bias_delta=bias,
                        bev_delta=delta,
                        position_failures=fails,
                        passed=passed,
                    )
                )
    eligible = [
        (m, t)
        for m in ["no_mobile", "mobile"]
        for t in [0.5, 0.9, 0.95]
        if all(c["passed"] for c in checks if c["mode"] == m and c["threshold"] == t)
    ]
    return dict(
        checks=checks,
        eligible=eligible,
        promoted=bool(eligible),
        frozen_diagnostic_threshold=0.9,
        selection="No successful candidate => RGB0.9 fixed diagnostic only, no training or default upgrade",
    )


def main():
    r = json.loads(Path("reports/m5m_gating_val.json").read_text())
    write(
        "reports/m5m_residual_lock.json",
        dict(
            kind="validation_only_fixed_residual_decision",
            **decide(r),
            source=asset("reports/m5m_gating_val.json"),
        ),
    )


if __name__ == "__main__":
    main()
