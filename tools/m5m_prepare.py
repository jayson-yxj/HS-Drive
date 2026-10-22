"""Freeze historical assets and predeclare M5-M gates before evaluation."""

import json, subprocess
from pathlib import Path
from tools.m5l_common import write, asset
from simulator.recorder import file_hash


def main():
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert base.startswith("f2f4e1a")
    old = json.loads(Path("runs/m5l_logs/baseline.json").read_text())
    assets = old["assets"]
    p = Path("/root/gpufree-data/hs-drive-data/m5l_asset_sha_index.json")
    if p.exists():
        assets.update(
            {k: v["sha256"] for k, v in json.loads(p.read_text())["assets"].items()}
        )
    tracked = {
        p: file_hash(p)
        for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    }
    bad = [p for p, h in assets.items() if file_hash(p) != h]
    assert not bad
    write(
        "runs/m5m_logs/baseline.json", dict(base=base, tracked=tracked, assets=assets)
    )
    write(
        "reports/m5m_baseline_freeze.json",
        dict(
            base=base,
            tracked=len(tracked),
            assets=len(assets),
            mismatches=bad,
            index=asset("runs/m5m_logs/baseline.json"),
        ),
    )
    write(
        "configs/m5m_protocol.json",
        dict(
            stage="M5-M",
            base=base,
            seeds=[101, 102],
            historical_tests="Both M5K and M5L tests regression only",
            contact=dict(
                gold="Multiple road collision probes agree with rendered local road (<=.10m), stable body pose, independently identifiable visible wheel/foot contact support, no occlusion/truncation/floating/penetration ambiguity. AABB/lowest rendered pixel alone never Gold.",
                head_gate=dict(
                    each_train_val=True,
                    positive_gold=50,
                    negative_reliably_unlocalizable=50,
                    independent_positions=4,
                    per_class_minimum=20,
                ),
                candidate="Visible surface and local road plane but contact anatomy or collision/render agreement incomplete",
                unknown="No stable road support or inconsistent/occluded contact; not absent/free",
            ),
            selective=dict(
                thresholds=[0, 0.2, 0.4, 0.6, 0.8],
                min_all_visible_coverage=0.30,
                min_matched=20,
                min_positions=3,
                max_p95_m=5.0,
                max_severe5m_rate=0.05,
                selection="Val only: highest coverage satisfying all gates in BOTH seeds; no admissible point => no promotion, retain threshold0 diagnostic",
                confidence="Uncalibrated RGB shape and geometric sensitivity score; no Gold => no VerifiedPosition",
                output="Unknown preserves candidate and optional conditional xy; ConditionalPosition experimental assumption only; Verified disabled",
            ),
            residual=dict(
                rgb_gate_thresholds=[0.5, 0.9, 0.95],
                near_mae_ratio=1.05,
                near_actor_space_drop=0.0,
                far_bias_increase=0.5,
                vehicle_space_gain=0.01,
                bev_precision_gain=0.01,
                bev_recall1_gain=0.05,
                fpr_increase=0.001,
                selection="Val only simultaneous two-seed gates plus per-position near/bias; no admissible gate => stop residual training",
                oracle="GT vehicle mask diagnostic only, never inference",
            ),
            plane=dict(
                rgb="Learned original depth fit below RGB component excluding q>=.1; no road GT. Candidate only; unstable fit rejected",
                fit_min_points=12,
                fit_max_rmse_m=0.1,
                max_slope_deg=15,
            ),
            shadow=dict(
                hz=2,
                ticks=160,
                warmup=20,
                repeats=2,
                max_age_s=0.25,
                order=[
                    ["alone", "loaded_idle", "shadow", "recovery"],
                    ["alone", "shadow", "loaded_idle", "recovery"],
                ],
                stop="tick P95 > max(1.5 baseline, baseline+20ms) or any tick >1000ms; stop worker and restore alone",
                safety_horizon_s=6,
                control=False,
            ),
            sealed=dict(
                positions=4,
                min_prior_m=45,
                after_freeze=True,
                configurations="empty, small/large vehicle, person, cone, occlusion, shifted orientation",
                purpose="Independent conditional geometry regression; Gold only by same predeclared evidence. No tuning.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
