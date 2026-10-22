"""Validation-only uncertainty calibration and frozen sealed evaluation."""

import argparse, json
from pathlib import Path
import numpy as np, torch
from tools.m5n_common import write, asset, stats
from tools.m5n_train import read
from models.m5n_direct import DirectCenterHead, reconstruct, output_contract
from diagnostics.m5n_metrics import (
    error,
    risk_features,
    fit_risk,
    predict_risk,
    aggregate,
    same_target,
    qualification,
)
from training.runtime import prepare_cuda
from simulator.recorder import file_hash


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    a = p.parse_args()
    prepare_cuda()
    torch.set_num_threads(2)
    X, cc, sup = read(a.split)
    training = json.loads(Path("reports/m5n_direct_range_training.json").read_text())
    assert training["status"] == "Completed"
    results = {}
    calibrations = {}
    curves = []
    summaries = []
    strata = []
    calibration_bins = []
    out = Path(f"runs/m5n_eval_{a.split}")
    out.mkdir(exist_ok=False)
    for seed in [101, 102]:
        selected = [c for c in cc if c["seed"] == seed]
        ids = [c["feature_index"] for c in selected]
        feats = X[ids].cuda()
        regions = torch.tensor([c["region"] for c in selected], device="cuda")
        K = torch.tensor([c["K"] for c in selected], device="cuda")
        T = torch.tensor([c["T"] for c in selected], device="cuda")
        predictions = {
            c["candidate_id"]: dict(
                candidate_id=c["candidate_id"],
                seed=seed,
                group=c["group"],
                sample_id=c["sample_id"],
                camera=c["camera"],
                region=c["region"],
                timestamp=c["timestamp"],
                positions={m: c[m] for m in ["plane", "dense_expected", "dense_mode"]},
            )
            for c in selected
        }
        for arm in ["direct", "geometry_prior"]:
            rec = next(
                r for r in training["records"] if r["seed"] == seed and r["arm"] == arm
            )
            assert file_hash(rec["selected"]["path"]) == rec["selected"]["sha256"]
            ck = torch.load(rec["selected"]["path"], weights_only=True)
            model = DirectCenterHead(ck["input_dim"]).cuda().eval()
            model.load_state_dict(ck["model"])
            pred = model(feats if arm == "direct" else feats[:, -24:])
            xyz = reconstruct(pred, regions, K, T).cpu().numpy()
            for c, point, raw in zip(selected, xyz, pred.cpu().numpy()):
                predictions[c["candidate_id"]]["positions"][arm] = point.tolist()
                predictions[c["candidate_id"]][arm + "_raw"] = raw.tolist()
        actors = [r for r in sup["actors"] if r["seed"] == seed]
        matched = [r for r in actors if r["detected"]]
        byid = {c["candidate_id"]: c for c in selected}
        features = {
            k: risk_features(byid[k], p["positions"]["direct"])
            for k, p in predictions.items()
        }
        if a.split == "val":
            good = [
                l
                for l in sup["labels"]
                if l["matched"]
                and l["train_eligible"]
                and cc[l["feature_index"]]["seed"] == seed
            ]
            fitids = [l["candidate_id"] for l in good]
            truth = {r["candidate_id"]: r for r in matched}
            rx = np.array([features[k] for k in fitids])
            y = np.array(
                [
                    error(
                        predictions[k]["positions"]["direct"],
                        truth[k]["box_center_ego"],
                    )["xy"]
                    for k in fitids
                ]
            )
            groups = np.array([truth[k]["group"] for k in fitids])
            cal = fit_risk(rx, y)
            oof = {}
            for group in sorted(set(groups)):
                mask = groups != group
                fold = fit_risk(rx[mask], y[mask])
                keys = [k for k, p in predictions.items() if p["group"] == group]
                vv = predict_risk(fold, [features[k] for k in keys])
                oof.update(dict(zip(keys, vv.tolist())))
            radius = float(np.quantile(y / np.array([oof[k] for k in fitids]), 0.9))
            cal.update(
                empirical90_multiplier=radius,
                validation_positions=len(set(groups)),
                calibration_samples=len(y),
                method="leave_position_out_ridge_logXYerror; selected head previously uses all Val for checkpoint, NOT independent conformal",
            )
            calibrations[str(seed)] = cal
        else:
            cal = json.loads(Path("reports/m5n_validation_lock.json").read_text())[
                "calibration"
            ][str(seed)]
            oof = {}
        risk = predict_risk(cal, [features[k] for k in predictions])
        for (key, pred), r in zip(predictions.items(), risk):
            e = float(oof.get(key, r))
            pred.update(
                proxy_error_m=e,
                confidence=float(np.exp(-e / 5)),
                empirical90_radius_m=e * cal["empirical90_multiplier"],
            )
            pred["output"] = output_contract(
                key,
                pred["region"],
                pred["positions"]["direct"],
                timestamp=pred["timestamp"],
                proxy_error_m=e,
            )
            pred["output"]["localization_uncertainty"]["empirical90_radius_m"] = pred[
                "empirical90_radius_m"
            ]
        results.update(predictions)
        threshold = (
            0
            if a.split == "val"
            else json.loads(Path("reports/m5n_validation_lock.json").read_text())[
                "threshold"
            ]
        )
        for t in [0, 0.2, 0.4, 0.6, 0.8]:
            m = aggregate(actors, predictions, "direct", t)
            same = same_target(actors, predictions, t)
            curves.append(
                dict(
                    seed=seed,
                    threshold=t,
                    metrics=m,
                    matched=same,
                    passed=qualification(m, same),
                )
            )
        for method in [
            "plane",
            "dense_expected",
            "dense_mode",
            "direct",
            "geometry_prior",
        ]:
            summaries.append(
                dict(
                    seed=seed,
                    method=method,
                    **aggregate(actors, predictions, method, threshold),
                )
            )
        for field, fn in {
            "type": lambda r: r["type"],
            "camera": lambda r: r["camera"],
            "group": lambda r: r["group"],
            "range": lambda r: (
                "near"
                if r["ego_xy_range_m"] < 10
                else "mid" if r["ego_xy_range_m"] < 20 else "far"
            ),
            "road": lambda r: "slope" if abs(r["pitch_deg"]) > 1 else "flat",
            "occlusion_proxy": lambda r: (
                "truncated"
                if r["truncated"]
                else (
                    "low_visible_fraction"
                    if r["visibility_ratio_proxy"] < 0.3
                    else "visible"
                )
            ),
        }.items():
            for value in sorted({fn(r) for r in actors}):
                for method in [
                    "plane",
                    "dense_expected",
                    "dense_mode",
                    "direct",
                    "geometry_prior",
                ]:
                    strata.append(
                        dict(
                            seed=seed,
                            field=field,
                            value=value,
                            method=method,
                            **aggregate(
                                [r for r in actors if fn(r) == value],
                                predictions,
                                method,
                                threshold,
                            ),
                        )
                    )
        for lo, hi in zip([0, 0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8, 1.01]):
            subset = [
                r
                for r in matched
                if lo <= predictions[r["candidate_id"]]["confidence"] < hi
            ]
            errors = [
                error(
                    predictions[r["candidate_id"]]["positions"]["direct"],
                    r["box_center_ego"],
                )["xy"]
                for r in subset
            ]
            calibration_bins.append(
                dict(
                    seed=seed,
                    confidence_lo=lo,
                    confidence_hi=hi,
                    n=len(subset),
                    xy=stats(errors),
                    within5_rate=(
                        float(np.mean(np.array(errors) <= 5)) if errors else None
                    ),
                    empirical_interval_coverage=(
                        float(
                            np.mean(
                                [
                                    e
                                    <= predictions[r["candidate_id"]][
                                        "empirical90_radius_m"
                                    ]
                                    for r, e in zip(subset, errors)
                                ]
                            )
                        )
                        if errors
                        else None
                    ),
                )
            )
    if a.split == "val":
        eligible = [
            t
            for t in [0, 0.2, 0.4, 0.6, 0.8]
            if all(
                next(r for r in curves if r["seed"] == s and r["threshold"] == t)[
                    "passed"
                ]
                for s in [101, 102]
            )
        ]
        write(
            "reports/m5n_validation_lock.json",
            dict(
                kind="validation_only_selection",
                threshold=min(eligible) if eligible else 0,
                promoted=bool(eligible),
                calibration=calibrations,
                curves=curves,
                limits="No passing point => threshold0 estimates diagnostic, no eligibility. OOF calibrator scores still optimistic because same Val checkpoint selection. No real-road coverage guarantee.",
            ),
        )
    write(out / "predictions.json", results)
    write(
        f"reports/m5n_eval_{a.split}.json",
        dict(
            kind="actual_GPU_frozen_head_plus_offline_reference_evaluation",
            split=a.split,
            aggregate=summaries,
            strata=strata,
            curves=curves,
            calibration_bins=calibration_bins,
            predictions=asset(out / "predictions.json"),
            source_sha256=file_hash(__file__),
            limits="All known visible actor-view denominator including missed detections, unlocalized never zero error. Same-target comparison reported separately. Fixed thresholds only; sealed curves diagnostic, no test selection. Unmatched candidate falseobjects unknowable without complete scene truth.",
        ),
    )


if __name__ == "__main__":
    main()
