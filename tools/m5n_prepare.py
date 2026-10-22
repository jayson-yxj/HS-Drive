"""Predeclare finite actor-center study and freeze M5-M assets."""

import json, subprocess
from pathlib import Path
from tools.m5n_common import write, asset
from simulator.recorder import file_hash


def main():
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert base.startswith("29ad67a")
    old = json.loads(Path("runs/m5m_logs/baseline.json").read_text())
    assets = old["assets"]
    assets.update(
        {
            p: v["sha256"]
            for p, v in json.loads(
                Path(
                    "/root/gpufree-data/hs-drive-data/m5m_asset_sha_index.json"
                ).read_text()
            )["assets"].items()
        }
    )
    # Prior index may include logs later finalized; baseline records current bytes; old immutable models/raw verified separately.
    tracked = {
        p: file_hash(p)
        for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    }
    prior_index_changes = [
        p for p, h in assets.items() if Path(p).exists() and file_hash(p) != h
    ]
    assets = {p: file_hash(p) for p in assets if Path(p).exists()}
    write(
        "runs/m5n_logs/baseline.json",
        dict(
            base=base,
            tracked=tracked,
            assets=assets,
            prior_finalized_index_changes=prior_index_changes,
        ),
    )
    write(
        "configs/m5n_protocol.json",
        dict(
            base=base,
            seeds=[101, 102],
            reference="Mean of eight transformed OBB vertices is simulator BoxCenter; not Actor origin/render surface/contact. Inverse actor transform recovers local bbox orientation/center. Frame joins and static between-ticks acquisition required.",
            matching=dict(
                iou=0.1,
                min_native_pixels=3,
                unique_intersection_pixels=3,
                train="one-to-one unmerged unsplit matched candidate only; no GT feature input",
            ),
            support=dict(
                min_train_unique_views=40,
                min_val_unique_views=30,
                min_positions_each=4,
                min_distance_bands=2,
                no_claim_for_under20_class=True,
            ),
            features="Frozen Mobile fusedP8 mask-bbox3x3 sampled grid (576) + legal K/T/region geometry (24); inference noGT. Companion geometry-only prior same data/order.",
            training=dict(
                steps=600,
                batch=32,
                lr=0.001,
                seeds=[101, 102],
                eval_steps=[0, 100, 200, 400, 600],
                loss="Huber(logZ) + Huber(normalized projected-center offset), equal weights; no NLL",
                selection="Val position-macro XY MAE minimum; no budget extension",
                stop="nonfinite loss/gradient abort; insufficient support skip",
            ),
            calibration=dict(
                method="Posthoc ridge logXYerror from legal predicted logZ, RGB component shape, predicted UV offset; leave-one-position-out Val diagnostics; final allVal fit only for sealed. Empirical90% radius, no conformal guarantee.",
                score="exp(-predicted_proxy_error_m/5)",
                thresholds=[0, 0.2, 0.4, 0.6, 0.8],
                min_coverage=0.3,
                min_matches=20,
                min_positions=3,
                p95_m=5,
                severe5m_rate=0.05,
                comparison="same accepted targets: MAE and P95 must improve over Plane AND DenseExpected AND DenseMode; both seeds",
                no_qualification="keep threshold0 research estimates, never VerifiedContact; no new training",
            ),
            baselines=[
                "frozen_M5M_plane",
                "DenseExpected_region_probability",
                "DenseMode_region_probability",
                "DirectHead",
                "GeometryPrior",
            ],
            splits="K train/val + M dev original splits (6positions each), K/L/M oldtests regression only. Actor IDs reused across worlds not identities; group-based exclusion.",
            sealed=dict(
                positions=4,
                min_prior_m=45,
                layouts="empty, near/mid/far car, side/rear, pedestrian, occlusion, cone, truncation/overlap; captured after freeze",
            ),
            runtime="offline GPU only; M5M Partial inherited, no shadow matrix/control",
        ),
    )
    write(
        "reports/m5n_baseline_freeze.json",
        dict(
            base=base,
            tracked=len(tracked),
            assets=len(assets),
            prior_finalized_index_changes=prior_index_changes,
            index=asset("runs/m5n_logs/baseline.json"),
        ),
    )


if __name__ == "__main__":
    main()
